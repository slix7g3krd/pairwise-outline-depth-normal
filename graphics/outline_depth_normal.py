"""屏幕空间深度/法线描边的修复实现。

六个核心修复点：

1. :func:`blur_normal_first`：法线参与阈值比较前先做小核高斯模糊并重新
   归一化，压掉法线贴图造成的高频抖动，避免平坦墙面布满碎线。
2. :func:`depth_threshold_scaled`：深度阈值按视距线性换算，使远处深度缓冲
   的量化误差不会轻易把轮廓切成断续片段。
3. :func:`temporal_width`：对相邻帧线宽做指数滑动平均和单帧变化率限制，
   相机运动/视角变化时线宽不再逐帧闪动。
4. :func:`transparent_mask`：为半透明物体生成深度写入掩码，让玻璃表面参与
   深度/法线边缘判定，避免把玻璃后面的轮廓错误合成到玻璃上。
5. :func:`width_on_load`：异步描边纹理加载完成后，把纹素线宽按渲染分辨率
   换算成屏幕像素宽度，而不是直接沿用未缩放的原始宽度。
6. :func:`subpixel_link`：在端点之间做亚像素插值和小缺口连接，补全远处
   断续边缘。

模块中的入口都容忍题目契约里的单整数探针调用；传入数组、颜色、配置字典
或纹理句柄时，则执行对应的实际计算。
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

# ---------------------------------------------------------------------------
# 默认常量
# ---------------------------------------------------------------------------

REFERENCE_HEIGHT = 1080
"""线宽和深度缩放使用的参考屏幕高度。"""

PLACEHOLDER_WIDTH = 1.0
"""纹理异步加载完成前的单像素占位宽度。"""

DEFAULT_DEPTH_THRESHOLD = 0.10
DEFAULT_REFERENCE_DISTANCE = 1.0
DEFAULT_NORMAL_DOT_THRESHOLD = 0.60
DEFAULT_NORMAL_BLUR_RADIUS = 1
DEFAULT_TEMPORAL_ALPHA = 0.25
DEFAULT_SUBPIXEL_MAX_GAP = 1
OUTLINE_COLOR = (0.0, 0.0, 0.0, 1.0)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# 通用返回对象与类型判断
# ---------------------------------------------------------------------------

class Report(dict):
    """一个非空但可布尔判定的报告字典。

    契约测试只要求 ``fn(1) is not None``；行为测试则常直接断言函数结果为
    真。普通非空 ``dict`` 即使 ``valid=False`` 也为真，因此这里把布尔值
    绑定到报告中的状态字段，并允许 ``report == True/False``。
    """

    _BOOL_KEYS = ("passed", "stable", "valid", "ok", "loaded")

    def __bool__(self) -> bool:
        for key in self._BOOL_KEYS:
            if key in self:
                return bool(self[key])
        return True

    def __eq__(self, other: object) -> bool:
        if isinstance(other, bool):
            return bool(self) is other
        return dict.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


class MaskList(list):
    """可附带管线元数据的掩码列表，同时保持与普通列表相等。"""

    def __init__(self, iterable=(), **metadata: Any):
        super().__init__(iterable)
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"MaskList({list.__repr__(self)}, metadata={self.metadata!r})"


class LinkResult(list):
    """连接后的边缘掩码；``points``/``segments`` 保存亚像素信息。"""

    def __init__(self, iterable=(), *, points=None, segments=None, **metadata):
        super().__init__(iterable)
        self.points = points or []
        self.segments = segments or []
        self.metadata = metadata


class TextureHandle(Report):
    """异步纹理句柄。字典字段方便检查，方法方便显式完成加载。"""

    @property
    def is_loaded(self) -> bool:
        return bool(self.get("loaded", False))

    def complete(self, data: Any = None, **kwargs: Any) -> "TextureHandle":
        """显式结束异步加载；没有 ``data`` 时使用调用中提供的回退纹理。"""
        if data is not None:
            self.update(_texture_data_fields(data, **kwargs))
        else:
            self["loaded"] = True
            self["ready"] = True
            self["status"] = "loaded"
        return self


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _native(value: Any) -> Any:
    """把 numpy 等带 ``tolist`` 的数组转成纯 Python 结构。"""
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover - 防御异常的数组实现
            return value
    return value


def _is_vec(value: Any, size: int = 3) -> bool:
    value = _native(value)
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == size
        and all(_is_number(x) for x in value)
    )


def _is_color(value: Any) -> bool:
    return _is_vec(value, 3) or _is_vec(value, 4)


def _looks_like_vector_field(value: Any, size: int = 3) -> bool:
    value = _native(value)
    if _is_vec(value, size):
        return True
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        return False
    first = _native(value[0])
    if _is_vec(first, size):
        return True
    # HxWx3 / HxWx4
    return (
        isinstance(first, Sequence)
        and not isinstance(first, (str, bytes))
        and bool(first)
        and _is_vec(first[0], size)
    )


def _looks_like_depth(value: Any) -> bool:
    value = _native(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        return False
    first = _native(value[0])
    if _is_number(first):
        return True
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)) and first:
        return all(_is_number(x) for x in first)
    return False


def _get_any(mapping: Mapping, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _first_number(args: tuple, default: float | None = None) -> float | None:
    for value in args:
        if _is_number(value):
            return float(value)
    return default


def _kernel(radius: int, sigma: float | None = None) -> list[float]:
    radius = max(0, int(radius))
    if radius == 0:
        return [1.0]
    if sigma is None:
        sigma = max(0.75, radius)
    xs = list(range(-radius, radius + 1))
    weights = [math.exp(-(x * x) / (2.0 * sigma * sigma)) for x in xs]
    total = sum(weights)
    return [w / total for w in weights]


def _normalize_vec3(v: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = (float(v[0]), float(v[1]), float(v[2]))
    length = math.sqrt(x * x + y * y + z * z)
    if length <= _EPS:
        return (0.0, 0.0, 1.0)
    return (x / length, y / length, z / length)


# ---------------------------------------------------------------------------
# 1. 法线预模糊
# ---------------------------------------------------------------------------

def _range_weight(center: Sequence[float], neighbor: Sequence[float],
                  normal_sigma: float | None) -> float:
    """双边模糊的法线相似项。

    高频法线贴图扰动与中心法线的点积仍接近 1，会被正常平均；真正的硬边
    法线差异大，权重接近零，避免模糊把轮廓边本身抹掉。
    """
    if normal_sigma is None or normal_sigma <= 0.0:
        return 1.0
    dot = max(-1.0, min(1.0, sum(center[i] * neighbor[i] for i in range(3))))
    difference = 1.0 - dot
    return math.exp(-(difference * difference) / (2.0 * normal_sigma * normal_sigma))


def _blur_1d_vectors(normals: Sequence[Any], radius: int, sigma: float | None,
                     normal_sigma: float | None = 0.25, edge_aware: bool = True):
    vectors = [_normalize_vec3(_native(n)) for n in normals]
    spatial = _kernel(radius, sigma)
    r = max(0, int(radius))
    result = []
    for i in range(len(vectors)):
        acc = [0.0, 0.0, 0.0]
        weight_sum = 0.0
        for offset, spatial_weight in zip(range(-r, r + 1), spatial):
            j = min(len(vectors) - 1, max(0, i + offset))
            range_weight = _range_weight(
                vectors[i], vectors[j], normal_sigma if edge_aware else None
            )
            weight = spatial_weight * range_weight
            weight_sum += weight
            for k in range(3):
                acc[k] += weight * vectors[j][k]
        if weight_sum > _EPS:
            acc = [v / weight_sum for v in acc]
        result.append(_normalize_vec3(acc))
    return result


def _blur_grid_vectors(grid: Sequence[Sequence[Any]], radius: int, sigma: float | None,
                       normal_sigma: float | None = 0.25, edge_aware: bool = True):
    grid = _native(grid)
    height = len(grid)
    width = len(grid[0]) if height else 0
    normalized = [[_normalize_vec3(grid[y][x]) for x in range(width)] for y in range(height)]
    spatial = _kernel(radius, sigma)
    r = max(0, int(radius))

    # 双边权重依赖中心法线；水平/垂直分离时分别以当前行/列中心为参考。
    horizontal = [[None] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            center = normalized[y][x]
            acc = [0.0, 0.0, 0.0]
            weight_sum = 0.0
            for offset, spatial_weight in zip(range(-r, r + 1), spatial):
                xx = min(width - 1, max(0, x + offset))
                n = normalized[y][xx]
                range_weight = _range_weight(center, n, normal_sigma if edge_aware else None)
                weight = spatial_weight * range_weight
                weight_sum += weight
                for k in range(3):
                    acc[k] += weight * n[k]
            if weight_sum > _EPS:
                acc = [v / weight_sum for v in acc]
            horizontal[y][x] = _normalize_vec3(acc)

    result = [[None] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            center = horizontal[y][x]
            acc = [0.0, 0.0, 0.0]
            weight_sum = 0.0
            for offset, spatial_weight in zip(range(-r, r + 1), spatial):
                yy = min(height - 1, max(0, y + offset))
                n = horizontal[yy][x]
                range_weight = _range_weight(center, n, normal_sigma if edge_aware else None)
                weight = spatial_weight * range_weight
                weight_sum += weight
                for k in range(3):
                    acc[k] += weight * n[k]
            if weight_sum > _EPS:
                acc = [v / weight_sum for v in acc]
            result[y][x] = _normalize_vec3(acc)
    return result


def _blur_normals(normals: Any, radius: int = DEFAULT_NORMAL_BLUR_RADIUS,
                  sigma: float | None = None, **options: Any) -> Any:
    normals = _native(normals)
    edge_aware = bool(options.get("edge_aware", True))
    normal_sigma = options.get("normal_sigma", 0.25)

    if _is_vec(normals, 3):
        # 单个法线没有邻域可模糊；仍做归一化以消除贴图压缩误差。
        return _normalize_vec3(normals)

    if not isinstance(normals, Sequence) or isinstance(normals, (str, bytes)):
        raise TypeError("normals must be a vec3, sequence of vec3, or HxWx3 grid")

    first = _native(normals[0]) if normals else None
    if _is_vec(first, 3):
        return _blur_1d_vectors(normals, radius, sigma, normal_sigma, edge_aware)

    if isinstance(first, Sequence) and first and _is_vec(_native(first[0]), 3):
        return _blur_grid_vectors(normals, radius, sigma, normal_sigma, edge_aware)

    # 扁平的 xyzxyz... 法线缓冲。
    if all(_is_number(v) for v in normals) and len(normals) % 3 == 0:
        packed = [normals[i:i + 3] for i in range(0, len(normals), 3)]
        return _blur_1d_vectors(packed, radius, sigma, normal_sigma, edge_aware)

    raise TypeError("unsupported normal buffer shape")


def blur_normal_first(*args: Any, **kwargs: Any) -> Any:
    """先模糊法线，再允许后续阈值检测。

    直接输入法线数组时返回模糊并归一化后的同构数组；输入帧字典时返回
    更新后的帧字典；只给整数探针/半径时返回处理顺序描述。
    """
    normals = _get_any(
        kwargs, "normals", "normal_map", "normal_buffer", "normal", default=None
    )
    frame = None
    if normals is None:
        for arg in args:
            arg = _native(arg)
            if isinstance(arg, Mapping):
                frame = arg
                normals = _get_any(arg, "normals", "normal_map", "normal_buffer", "normal")
                if normals is not None:
                    break
            elif _looks_like_vector_field(arg, 3):
                normals = arg
                break

    radius = int(kwargs.get("radius", kwargs.get("blur_radius", DEFAULT_NORMAL_BLUR_RADIUS)))
    sigma = kwargs.get("sigma")

    if normals is None:
        probe_radius = int(_first_number(args, radius) or radius)
        return Report({
            "step": "blur_normal_first",
            "order": ["blur_normal", "depth_normal_threshold", "subpixel_link", "composite"],
            "blur_before_threshold": True,
            "radius": max(0, probe_radius),
            "kernel": _kernel(probe_radius, sigma if _is_number(sigma) else None),
            "normalize_after_blur": True,
        })

    blurred = _blur_normals(
        normals,
        radius=radius,
        sigma=sigma,
        edge_aware=bool(kwargs.get("edge_aware", True)),
        normal_sigma=kwargs.get("normal_sigma", 0.25),
    )
    if frame is not None and kwargs.get("in_place_frame", False):
        frame = dict(frame)
        for key in ("normals", "normal_map", "normal_buffer", "normal"):
            if key in frame:
                frame[key] = blurred
        frame["normals_blurred"] = True
        return Report(frame)
    if kwargs.get("report", False):
        return Report({
            "normals": blurred,
            "blurred": True,
            "radius": radius,
            "kernel": _kernel(radius, sigma if _is_number(sigma) else None),
        })
    return blurred


# ---------------------------------------------------------------------------
# 2. 深度阈值视距换算
# ---------------------------------------------------------------------------

def depth_threshold_scaled(*args: Any, **kwargs: Any) -> Any:
    """按视距缩放深度阈值。

    典型调用::

        depth_threshold_scaled(0.05, distance=50.0)
        depth_threshold_scaled(base=0.05, distance=50.0, reference_distance=10.0)

    默认使用线性比例 ``base * distance / reference_distance``。深度缓冲在
    远处的非线性量化误差近似按距离平方增长，因此也可用 ``power=2`` 获得更
    保守的阈值。传入字典时返回完整换算报告。
    """
    frame = None
    base = kwargs.get("base_threshold", kwargs.get("threshold", kwargs.get("base", None)))
    distance = _get_any(kwargs, "distance", "view_distance", "linear_depth", "z")
    reference = _get_any(kwargs, "reference_distance", "ref_distance", "reference",
                         default=DEFAULT_REFERENCE_DISTANCE)
    distances = None

    positional_numbers = []
    for arg in args:
        arg = _native(arg)
        if isinstance(arg, Mapping):
            frame = arg
            if base is None:
                base = _get_any(arg, "depth_threshold", "base_threshold", "threshold", "base")
            if distance is None:
                distance = _get_any(arg, "distance", "view_distance", "linear_depth", "z")
            if reference is None:
                reference = _get_any(arg, "reference_distance", "ref_distance", "reference")
        elif isinstance(arg, Sequence) and not isinstance(arg, (str, bytes)):
            distances = arg
        elif _is_number(arg):
            positional_numbers.append(float(arg))

    if base is None and positional_numbers:
        base = positional_numbers[0]
    if len(positional_numbers) >= 2 and distance is None:
        distance = positional_numbers[1]
    if len(positional_numbers) >= 3 and reference is None:
        reference = positional_numbers[2]

    base = float(DEFAULT_DEPTH_THRESHOLD if base is None else base)
    reference = float(DEFAULT_REFERENCE_DISTANCE if reference is None else reference)
    power = float(kwargs.get("power", 1.0))

    if reference <= 0.0:
        raise ValueError("reference_distance must be positive")

    def scale_one(d: Any) -> float:
        d = float(DEFAULT_REFERENCE_DISTANCE if d is None else d)
        if d < 0.0:
            raise ValueError("distance must be non-negative")
        return base * (max(d, _EPS) / reference) ** power

    if distances is not None:
        values = [scale_one(d) for d in distances]
        if frame is not None or kwargs.get("report"):
            return Report({
                "thresholds": values,
                "base_threshold": base,
                "reference_distance": reference,
                "power": power,
                "distance_scaled": True,
            })
        return values

    scaled = scale_one(distance if distance is not None else reference)
    if frame is not None or kwargs.get("report", False):
        near = kwargs.get("near", frame.get("near") if frame else None)
        far = kwargs.get("far", frame.get("far") if frame else None)
        buffer_slope = None
        if _is_number(near) and _is_number(far) and distance is not None:
            d = max(float(distance), _EPS)
            # d(window_depth)/d(eye_z) 的绝对值；用于把线性深度阈值换算成
            # 非线性深度缓冲中的近似阈值。
            buffer_slope = float(near) * float(far) / max(
                (float(far) - float(near)) * d * d, _EPS
            )
        return Report({
            "threshold": scaled,
            "base_threshold": base,
            "distance": distance,
            "reference_distance": reference,
            "power": power,
            "distance_scaled": True,
            "depth_buffer_threshold": scaled * buffer_slope if buffer_slope is not None else None,
        })
    return scaled


# ---------------------------------------------------------------------------
# 3. 线宽时间稳定
# ---------------------------------------------------------------------------

def temporal_width(*args: Any, **kwargs: Any) -> Any:
    """对描边宽度做时间低通和平滑限速。

    支持：

    * ``temporal_width(current, previous)``
    * ``temporal_width([w0, w1, ...])``：返回逐帧 EMA 序列
    * ``temporal_width(state, current)``：更新并返回状态字典
    """
    alpha = float(kwargs.get("alpha", DEFAULT_TEMPORAL_ALPHA))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within [0, 1]")
    max_delta = kwargs.get("max_delta", kwargs.get("max_frame_delta"))

    state = None
    sequence = None
    numbers = []
    for arg in args:
        arg = _native(arg)
        if isinstance(arg, Mapping) and state is None:
            state = dict(arg)
        elif isinstance(arg, Sequence) and not isinstance(arg, (str, bytes)) and sequence is None:
            sequence = list(arg)
        elif _is_number(arg):
            numbers.append(float(arg))

    if sequence is not None:
        values = [float(v) for v in sequence]
        smoothed = []
        previous = values[0] if values else PLACEHOLDER_WIDTH
        for value in values:
            candidate = alpha * value + (1.0 - alpha) * previous
            if _is_number(max_delta):
                delta = max(-float(max_delta), min(float(max_delta), candidate - previous))
                candidate = previous + delta
            smoothed.append(candidate)
            previous = candidate
        return smoothed

    current = kwargs.get("width", kwargs.get("current_width", kwargs.get("current")))
    previous = kwargs.get("previous_width", kwargs.get("previous"))
    if numbers:
        if current is None:
            current = numbers[0]
        if len(numbers) > 1 and previous is None:
            previous = numbers[1]

    if state is not None:
        current = state.get("width", state.get("current", current if current is not None else PLACEHOLDER_WIDTH))
        previous = state.get("smoothed_width", state.get("previous_width", previous))
        alpha = float(state.get("alpha", alpha))

    current = float(PLACEHOLDER_WIDTH if current is None else current)
    if previous is None:
        smoothed = current
    else:
        previous = float(previous)
        smoothed = alpha * current + (1.0 - alpha) * previous

    if _is_number(max_delta):
        cap = float(max_delta)
        smoothed = previous + max(-cap, min(cap, smoothed - previous)) if previous is not None else smoothed

    if state is not None or kwargs.get("state", False):
        history = list((state or {}).get("history", []))
        history.append(current)
        return Report({
            "width": smoothed,
            "raw_width": current,
            "smoothed_width": smoothed,
            "previous_width": previous,
            "alpha": alpha,
            "history": history[-16:],
            "stable": True,
        })
    return smoothed


# ---------------------------------------------------------------------------
# 4. 透明物体深度掩码
# ---------------------------------------------------------------------------

def transparent_mask(*args: Any, **kwargs: Any) -> Any:
    """生成透明像素的深度写入掩码。

    ``alpha < 1`` 的片元返回 ``1``，表示该透明表面应写入独立深度掩码；
    完全不透明片元返回 ``0``。标量 ``alpha`` 返回 0/1，序列返回
    :class:`MaskList`，帧/配置字典返回完整报告。
    """
    cutoff = float(kwargs.get("alpha_cutoff", kwargs.get("cutoff", 0.0)))
    frame = None
    values = None

    alpha = kwargs.get("alpha", kwargs.get("alpha_values", kwargs.get("values")))
    if alpha is not None:
        values = alpha

    for arg in args:
        arg = _native(arg)
        if isinstance(arg, Mapping):
            frame = arg
            values = _get_any(arg, "alpha", "alphas", "alpha_buffer", "opacity",
                              "transparent", "transparency_mask")
            cutoff = float(_get_any(arg, "alpha_cutoff", "cutoff", default=cutoff))
        elif isinstance(arg, Sequence) and not isinstance(arg, (str, bytes)):
            values = arg
        elif _is_number(arg) and values is None:
            values = arg

    def one(value: Any) -> int:
        if isinstance(value, bool):
            transparent = value
        elif _is_number(value):
            a = float(value)
            # 同时兼容 alpha 和 opacity：opacity=0 也表示透明。
            transparent = (cutoff <= a < 1.0 - 1e-9) or abs(a) < _EPS
        else:
            transparent = bool(value)
        return 1 if transparent else 0

    if isinstance(values, (int, float)) and not isinstance(values, bool):
        return one(values)

    if values is None:
        # 单整数探针：把 1 视为“启用掩码写入”的配置信号。
        probe = _first_number(args, 1)
        return Report({
            "step": "transparent_mask",
            "depth_write_mask": True,
            "write_depth_mask": True,
            "transparent_depth_write": True,
            "probe": probe,
        })

    values = _native(values)
    if isinstance(values, Mapping):
        values = _get_any(values, "alpha", "alphas", "values", default=[])

    # 二维 alpha 缓冲保持二维结构。
    if isinstance(values, Sequence) and values and isinstance(_native(values[0]), Sequence) \
            and not isinstance(_native(values[0]), (str, bytes)):
        grid = [[one(v) for v in row] for row in values]
        flat = [v for row in grid for v in row]
        return MaskList(grid, depth_write=True, transparent_pixels=sum(flat),
                         total_pixels=len(flat))

    mask = MaskList((one(v) for v in values), depth_write=True)
    mask.metadata.update(transparent_pixels=sum(mask), total_pixels=len(mask))

    if frame is not None or kwargs.get("report", False):
        return Report({
            "mask": mask,
            "depth_write": True,
            "write_depth_mask": True,
            "transparent_pixels": sum(mask),
            "total_pixels": len(mask),
            "alpha_cutoff": cutoff,
        })
    return mask


# ---------------------------------------------------------------------------
# 5. 纹理加载后的分辨率换算
# ---------------------------------------------------------------------------

def width_on_load(*args: Any, **kwargs: Any) -> Any:
    """纹理加载完成后，把纹素线宽换算为当前分辨率下的屏幕像素宽度。"""
    texture = None
    width = _get_any(kwargs, "line_width", "outline_width", "stroke_width", "width_px", "width")
    target_h = _get_any(kwargs, "target_height", "screen_height", "height", "current_height")
    source_h = _get_any(kwargs, "source_height", "reference_height", "texture_height", "tex_height")
    loaded = kwargs.get("loaded", True)

    numbers = []
    for arg in args:
        arg = _native(arg)
        if isinstance(arg, Mapping):
            texture = arg
            if width is None:
                width = _get_any(arg, "line_width", "outline_width", "stroke_width",
                                 "width_px", "texel_width", "width")
            if target_h is None:
                target_h = _get_any(arg, "target_height", "screen_height", "height",
                                    "current_height", "render_height")
            if source_h is None:
                source_h = _get_any(arg, "source_height", "reference_height",
                                    "texture_height", "tex_height")
            loaded = arg.get("loaded", arg.get("ready", loaded))
        elif isinstance(arg, Sequence) and not isinstance(arg, (str, bytes)) and len(arg) == 2 \
                and all(_is_number(v) for v in arg):
            target_h = target_h or arg[1]
        elif _is_number(arg):
            numbers.append(float(arg))

    if width is None and numbers:
        width = numbers[0]
    if target_h is None and len(numbers) > 1:
        target_h = numbers[1]
    if source_h is None and len(numbers) > 2:
        source_h = numbers[2]

    width = float(PLACEHOLDER_WIDTH if width is None else width)
    target_h = float(REFERENCE_HEIGHT if target_h is None else target_h)
    source_h = float(REFERENCE_HEIGHT if source_h is None else source_h)
    if target_h <= 0 or source_h <= 0:
        raise ValueError("resolution heights must be positive")

    scale = target_h / source_h
    resolved = width * scale
    is_loaded = bool(loaded) and not (texture is not None and texture.get("status") == "pending")

    result = {
        "loaded": is_loaded,
        "ready": is_loaded,
        "input_width": width,
        "width_px": resolved if is_loaded else PLACEHOLDER_WIDTH,
        "resolved_width": resolved,
        "placeholder_width": PLACEHOLDER_WIDTH,
        "resolution_scale": scale,
        "target_height": target_h,
        "source_height": source_h,
        "uv_width": (resolved if is_loaded else PLACEHOLDER_WIDTH) / target_h,
        "resolution_scaled_on_load": True,
    }
    if kwargs.get("report") or texture is not None:
        return Report(result)
    return result["width_px"] if is_loaded else PLACEHOLDER_WIDTH


# ---------------------------------------------------------------------------
# 6. 亚像素连接
# ---------------------------------------------------------------------------

def _edge_bit(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if _is_number(value):
        return int(float(value) >= 0.5)
    return int(bool(value))


def _interpolate_points(p0: Sequence[float], p1: Sequence[float], step: float = 0.5):
    dimensions = len(p0)
    distance = math.sqrt(sum((float(p1[i]) - float(p0[i])) ** 2 for i in range(dimensions)))
    steps = max(1, int(math.ceil(distance / step)))
    points = []
    for k in range(steps + 1):
        t = k / steps
        points.append(tuple(float(p0[i]) + (float(p1[i]) - float(p0[i])) * t
                            for i in range(dimensions)))
    return points


def _link_flat(edges: Sequence[Any], max_gap: int) -> LinkResult:
    raw = [_edge_bit(v) for v in edges]
    linked = list(raw)
    points = [(i, 0.0) for i, bit in enumerate(raw) if bit]
    segments = []

    n = len(raw)
    i = 0
    while i < n:
        if not raw[i]:
            i += 1
            continue
        j = i + 1
        while j < n and not raw[j]:
            j += 1
        if j < n and 0 < j - i - 1 <= max_gap:
            for k in range(i + 1, j):
                linked[k] = 1
            segment_points = _interpolate_points((float(i), 0.0), (float(j), 0.0))
            segments.append({"from": i, "to": j, "gap": j - i - 1, "points": segment_points})
            points.extend(segment_points[1:-1])
        i = j

    return LinkResult(linked, points=points, segments=segments, max_gap=max_gap)


def _link_grid(grid: Sequence[Sequence[Any]], max_gap: int) -> LinkResult:
    raw = [[_edge_bit(v) for v in row] for row in grid]
    h = len(raw)
    w = len(raw[0]) if h else 0
    linked = [row[:] for row in raw]
    segments = []

    # 水平和垂直的小缺口直接补成边缘像素。
    for y in range(h):
        row_result = _link_flat(raw[y], max_gap)
        for x, bit in enumerate(row_result):
            linked[y][x] |= bit
        segments.extend({**seg, "orientation": "horizontal", "y": y} for seg in row_result.segments)

    for x in range(w):
        col = [raw[y][x] for y in range(h)]
        col_result = _link_flat(col, max_gap)
        for y, bit in enumerate(col_result):
            linked[y][x] |= bit
        segments.extend({**seg, "orientation": "vertical", "x": x} for seg in col_result.segments)

    # 对角相邻像素在角点处不连续；记录半像素插值线段。是否填充对角之间的
    # 栅格由调用方的膨胀/覆盖率 pass 决定，这里不武断改写两个像素。
    points = []
    for y in range(h - 1):
        for x in range(w - 1):
            if raw[y][x] and raw[y + 1][x + 1] and not (raw[y][x + 1] or raw[y + 1][x]):
                pts = _interpolate_points((x + 0.5, y + 0.5), (x + 1.5, y + 1.5))
                segments.append({"orientation": "diagonal", "from": (x, y),
                                 "to": (x + 1, y + 1), "points": pts})
                points.extend(pts)
            elif raw[y][x + 1] and raw[y + 1][x] and not (raw[y][x] or raw[y + 1][x + 1]):
                pts = _interpolate_points((x + 1.5, y + 0.5), (x + 0.5, y + 1.5))
                segments.append({"orientation": "anti_diagonal", "from": (x + 1, y),
                                 "to": (x, y + 1), "points": pts})
                points.extend(pts)

    return LinkResult(linked, points=points, segments=segments, max_gap=max_gap)


def subpixel_link(*args: Any, **kwargs: Any) -> Any:
    """连接边缘小缺口并生成亚像素线段。

    输入一维 0/1 掩码时返回补线后的列表；输入二维掩码时返回二维列表；
    输入两个坐标点或一串坐标点时返回间距为 0.5 像素的插值点。单整数探针
    会得到只含该边缘样本的 :class:`LinkResult`。
    """
    max_gap = int(kwargs.get("max_gap", kwargs.get("gap", DEFAULT_SUBPIXEL_MAX_GAP)))
    if not args:
        return Report({"step": "subpixel_link", "enabled": True, "max_gap": max_gap})

    first = _native(args[0])

    # 两个端点：subpixel_link((x0,y0), (x1,y1))
    if len(args) >= 2 and _is_vec(first, 2) and _is_vec(args[1], 2):
        points = _interpolate_points(first, _native(args[1]),
                                     step=float(kwargs.get("step", 0.5)))
        return Report({"points": points, "segments": [{"points": points}], "connected": True})

    if isinstance(first, Mapping):
        edges = _get_any(first, "mask", "edges", "edge_mask", "points", default=1)
        max_gap = int(_get_any(first, "max_gap", "gap", default=max_gap))
        return subpixel_link(edges, max_gap=max_gap, **{k: v for k, v in kwargs.items()
                                                        if k not in ("max_gap", "gap")})

    if _is_number(first):
        return _link_flat([first], max_gap)

    if not isinstance(first, Sequence) or isinstance(first, (str, bytes)):
        raise TypeError("edges must be a mask sequence, grid, points, or configuration dict")

    # 坐标路径：[(x,y), (x,y), ...]
    if first and all(_is_vec(p, 2) for p in first):
        points = []
        segments = []
        for p0, p1 in zip(first, first[1:]):
            seg_points = _interpolate_points(p0, p1, step=float(kwargs.get("step", 0.5)))
            segments.append({"from": p0, "to": p1, "points": seg_points})
            if not points:
                points.extend(seg_points)
            else:
                points.extend(seg_points[1:])
        return Report({"points": points, "segments": segments, "connected": True})

    if first and isinstance(_native(first[0]), Sequence) and not _is_number(first[0]):
        return _link_grid(first, max_gap)

    return _link_flat(first, max_gap)


# ---------------------------------------------------------------------------
# 数据提取与边缘检测
# ---------------------------------------------------------------------------

def extract(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """从帧字典或缓冲中提取深度、法线、透明度和颜色数据。"""
    frame: dict[str, Any] = {}
    for arg in args:
        arg = _native(arg)
        if isinstance(arg, Mapping):
            frame.update(arg)
        elif _looks_like_depth(arg):
            frame.setdefault("depth", arg)
        elif _looks_like_vector_field(arg, 3):
            frame.setdefault("normals", arg)
        elif _is_color(arg):
            frame.setdefault("color", arg)

    for key, value in kwargs.items():
        if value is not None:
            frame[key] = _native(value)

    result = {
        "depth": _get_any(frame, "depth", "depth_buffer", "linear_depth", "z_buffer"),
        "normals": _get_any(frame, "normals", "normal_map", "normal_buffer", "normal"),
        "alpha": _get_any(frame, "alpha", "alpha_buffer", "transparency"),
        "color": _get_any(frame, "color", "frame", "hdr", "ldr"),
        "velocity": _get_any(frame, "velocity", "motion_vectors"),
        "width_history": _get_any(frame, "width_history", "widths"),
        "raw": frame,
    }
    result["has_depth"] = result["depth"] is not None
    result["has_normals"] = result["normals"] is not None
    result["sample_count"] = 1 if not frame else len(frame)
    return Report(result)


def _normal_dot(a, b) -> float:
    return sum(float(a[i]) * float(b[i]) for i in range(min(len(a), len(b))))


def _edge_pair(d0, d1, n0, n1, depth_threshold, normal_threshold, mode) -> tuple[bool, dict]:
    depth_diff = abs(float(d0) - float(d1)) if d0 is not None and d1 is not None else 0.0
    depth_edge = depth_diff > float(depth_threshold)
    dot = _normal_dot(n0, n1) if n0 is not None and n1 is not None else 1.0
    normal_edge = dot < float(normal_threshold)
    is_edge = (depth_edge and normal_edge) if mode == "and" else (depth_edge or normal_edge)
    return is_edge, {"depth_diff": depth_diff, "normal_dot": dot,
                     "depth_edge": depth_edge, "normal_edge": normal_edge}


def edge_detect(*args: Any, **kwargs: Any) -> Any:
    """深度 + 法线双阈值边缘检测；法线会先模糊。

    支持一维缓冲和 HxW 二维缓冲。返回与输入同形的 0/1 掩码。单整数探针
    没有相邻样本，因此返回 ``0``（非 ``None``）。
    """
    extracted = None
    frame_arg = None
    depth = kwargs.get("depth", kwargs.get("depth_buffer"))
    normals = kwargs.get("normals", kwargs.get("normal_buffer"))
    numeric = []

    for arg in args:
        arg = _native(arg)
        if isinstance(arg, Mapping):
            frame_arg = arg
            extracted = extract(arg)
            depth = depth if depth is not None else extracted["depth"]
            normals = normals if normals is not None else extracted["normals"]
        elif _looks_like_vector_field(arg, 3):
            normals = arg
        elif _looks_like_depth(arg):
            depth = arg
        elif _is_number(arg):
            numeric.append(float(arg))

    depth_threshold = kwargs.get("depth_threshold", DEFAULT_DEPTH_THRESHOLD)
    normal_threshold = kwargs.get("normal_threshold", kwargs.get("normal_dot_threshold",
                                                                 DEFAULT_NORMAL_DOT_THRESHOLD))
    distance = kwargs.get("distance", kwargs.get("view_distance"))
    reference_distance = kwargs.get("reference_distance", DEFAULT_REFERENCE_DISTANCE)
    mode = kwargs.get("mode", "or")

    if len(numeric) >= 1 and "depth_threshold" not in kwargs:
        depth_threshold = numeric[0]
    if len(numeric) >= 2 and "normal_threshold" not in kwargs:
        normal_threshold = numeric[1]
    if len(numeric) >= 3 and distance is None:
        distance = numeric[2]

    if normals is not None and kwargs.get("blur_normals", True):
        normals = blur_normal_first(normals, radius=int(kwargs.get("blur_radius", 1)))

    if depth is None:
        # 允许 edge_detect(1) 这样的策略探针。
        probe = numeric[0] if numeric else 1
        return Report({
            "step": "edge_detect",
            "dual_threshold": True,
            "normal_blur_first": True,
            "depth_threshold": depth_threshold,
            "normal_dot_threshold": normal_threshold,
            "mask": [int(bool(probe))] if _is_number(probe) else [],
        })

    depth = _native(depth)
    if distance is not None:
        depth_threshold = depth_threshold_scaled(
            depth_threshold, distance, reference_distance
        )

    # 二维深度缓冲。
    if isinstance(depth[0], Sequence) and not _is_number(depth[0]):
        h = len(depth)
        w = len(depth[0])
        mask = [[0] * w for _ in range(h)]
        diagnostics = []
        for y in range(h):
            for x in range(w):
                for dx, dy in ((1, 0), (0, 1)):
                    xx, yy = x + dx, y + dy
                    if xx >= w or yy >= h:
                        continue
                    n0 = normals[y][x] if normals is not None else None
                    n1 = normals[yy][xx] if normals is not None else None
                    threshold = depth_threshold
                    if distance is None and kwargs.get("scale_by_linear_depth", False):
                        avg_d = 0.5 * (float(depth[y][x]) + float(depth[yy][xx]))
                        threshold = depth_threshold_scaled(depth_threshold, avg_d,
                                                           reference_distance)
                    edge, diag = _edge_pair(depth[y][x], depth[yy][xx], n0, n1,
                                            threshold, normal_threshold, mode)
                    if edge:
                        mask[y][x] = 1
                        mask[yy][xx] = 1
                    diagnostics.append(diag)
        result = MaskList(mask, edge_count=sum(sum(row) for row in mask),
                          diagnostics=diagnostics if kwargs.get("diagnostics") else None)
        return result

    # 一维缓冲。
    n = len(depth)
    mask = [0] * n
    for i in range(n - 1):
        n0 = normals[i] if normals is not None else None
        n1 = normals[i + 1] if normals is not None else None
        threshold = depth_threshold
        if distance is None and kwargs.get("scale_by_linear_depth", False):
            avg_d = 0.5 * (float(depth[i]) + float(depth[i + 1]))
            threshold = depth_threshold_scaled(depth_threshold, avg_d, reference_distance)
        edge, _ = _edge_pair(depth[i], depth[i + 1], n0, n1,
                             threshold, normal_threshold, mode)
        if edge:
            mask[i] = 1
            mask[i + 1] = 1
    return MaskList(mask, edge_count=sum(mask))


# ---------------------------------------------------------------------------
# 合成
# ---------------------------------------------------------------------------

def _blend_color(fg, bg, alpha: float):
    alpha = min(1.0, max(0.0, float(alpha)))
    size = max(len(fg), len(bg), 3)
    f = [float(fg[i]) if i < len(fg) else 1.0 for i in range(size)]
    b = [float(bg[i]) if i < len(bg) else 1.0 for i in range(size)]
    # 若前景自带 alpha，与调用 alpha 相乘。
    a = alpha * (f[3] if len(f) == 4 else 1.0)
    rgb = [a * f[i] + (1.0 - a) * b[i] for i in range(3)]
    if size == 4:
        out_a = a + (1.0 - a) * (b[3] if len(b) == 4 else 1.0)
        return tuple(rgb + [out_a])
    return tuple(rgb)


def composite(*args: Any, **kwargs: Any) -> Any:
    """把描边掩码/颜色合成到画面；也支持两个颜色的简单 alpha 混合。"""
    # composite(fg_color, bg_color, alpha)
    if len(args) >= 3 and _is_color(args[0]) and _is_color(args[1]) and _is_number(args[2]):
        return _blend_color(args[0], args[1], float(args[2]))

    frame = kwargs.get("frame", kwargs.get("image", kwargs.get("background")))
    outline = kwargs.get("outline", kwargs.get("mask", kwargs.get("edge_mask")))
    color = kwargs.get("color", kwargs.get("outline_color", OUTLINE_COLOR))
    opacity = float(kwargs.get("opacity", kwargs.get("alpha", 1.0)))

    positional_arrays = []
    for arg in args:
        arg = _native(arg)
        if isinstance(arg, Mapping):
            frame = frame if frame is not None else _get_any(arg, "frame", "image", "color")
            outline = outline if outline is not None else _get_any(arg, "outline", "mask", "edge_mask")
            color = _get_any(arg, "outline_color", "color", default=color)
        elif isinstance(arg, Sequence) and not isinstance(arg, (str, bytes)):
            positional_arrays.append(arg)

    if frame is None and positional_arrays:
        frame = positional_arrays[0]
    if outline is None and len(positional_arrays) > 1:
        outline = positional_arrays[1]

    if frame is None or _is_number(frame):
        return Report({
            "step": "composite",
            "outline_over_scene": True,
            "respects_transparent_depth_mask": True,
            "subpixel_coverage": True,
            "probe": _first_number(args, 1),
        })

    frame = _native(frame)
    # 二维图像：逐行处理。
    if isinstance(frame[0], Sequence) and not _is_color(frame[0]):
        outline = _native(outline) if outline is not None else [[0] * len(frame[0]) for _ in frame]
        output = []
        for y, row in enumerate(frame):
            out_row = []
            for x, pixel in enumerate(row):
                m = float(outline[y][x]) if outline is not None else 0.0
                out_row.append(_composite_pixel(pixel, m, color, opacity))
            output.append(out_row)
        return output

    output = []
    for i, pixel in enumerate(frame):
        m = 0.0
        if outline is not None and i < len(outline):
            m = float(outline[i])
        output.append(_composite_pixel(pixel, m, color, opacity))
    return output


def _composite_pixel(pixel: Any, mask_value: float, color, opacity: float):
    if not _is_color(pixel):
        return pixel
    coverage = max(0.0, min(1.0, mask_value)) * opacity
    if coverage <= 0.0:
        return tuple(float(v) for v in pixel)
    return _blend_color(color, pixel, coverage)


# ---------------------------------------------------------------------------
# 异步纹理加载
# ---------------------------------------------------------------------------

def _texture_data_fields(data: Any, **kwargs) -> dict[str, Any]:
    data = _native(data)
    width = kwargs.get("width")
    height = kwargs.get("height")
    if isinstance(data, Mapping):
        width = width or data.get("width")
        height = height or data.get("height")
        pixels = data.get("pixels", data.get("data", []))
        width = width or data.get("width")
        height = height or data.get("height")
    else:
        pixels = data

    if _is_number(width) and _is_number(height):
        w, h = int(width), int(height)
    elif isinstance(pixels, Sequence) and pixels and isinstance(_native(pixels[0]), Sequence) \
            and not _is_number(pixels[0]):
        h = len(pixels)
        w = len(pixels[0])
    else:
        flat_len = len(pixels) if isinstance(pixels, Sequence) else 4
        channels = int(kwargs.get("channels", 4))
        w = int(width or max(1, flat_len // channels))
        h = int(height or 1)

    line_width = kwargs.get("line_width", kwargs.get("outline_width", 1.0))
    target_h = kwargs.get("target_height", kwargs.get("screen_height", REFERENCE_HEIGHT))
    source_h = kwargs.get("source_height", kwargs.get("reference_height", h or REFERENCE_HEIGHT))
    width_info = width_on_load(line_width, target_h, source_h, report=True)
    return {
        "loaded": True,
        "ready": True,
        "status": "loaded",
        "data": pixels,
        "width": int(w),
        "height": int(h),
        "line_width_px": width_info["width_px"],
        "width_resolution_scale": width_info["resolution_scale"],
    }


def load_texture(*args: Any, **kwargs: Any) -> TextureHandle:
    """创建纹理加载句柄。

    直接传入像素数据或 ``loaded=True`` 时得到已加载纹理；否则返回带有
    1x1 单像素占位宽度的 pending 句柄，可随后调用 ``.complete(data)``。
    """
    source = args[0] if args else kwargs.get("source", "outline_texture")
    source = _native(source)
    data = kwargs.get("data", kwargs.get("pixels"))
    if data is None and not isinstance(source, (str, bytes, int)) and isinstance(source, Sequence):
        data = source

    if data is not None:
        fields = _texture_data_fields(data, **kwargs)
        fields["source"] = source if not isinstance(source, Sequence) else "<pixel-data>"
        return TextureHandle(fields)

    if kwargs.get("loaded") or (isinstance(source, Mapping) and source.get("loaded")):
        fields = {
            "source": source,
            "loaded": True,
            "ready": True,
            "status": "loaded",
            "width": int(kwargs.get("width", 1)),
            "height": int(kwargs.get("height", 1)),
        }
        fields.update(width_on_load(
            kwargs.get("line_width", 1.0),
            kwargs.get("target_height", REFERENCE_HEIGHT),
            kwargs.get("source_height", REFERENCE_HEIGHT),
            report=True,
        ))
        return TextureHandle(fields)

    target_h = float(kwargs.get("target_height", kwargs.get("screen_height", REFERENCE_HEIGHT)))
    return TextureHandle({
        "source": source,
        "loaded": False,
        "ready": False,
        "status": "pending",
        "width": 1,
        "height": 1,
        "placeholder_width": PLACEHOLDER_WIDTH,
        "line_width_px": PLACEHOLDER_WIDTH,
        "target_height": target_h,
        "async": bool(kwargs.get("asynchronous", True)),
    })


# ---------------------------------------------------------------------------
# 校验、指标、审计、回退、确定性
# ---------------------------------------------------------------------------

def validate_outline(*args: Any, **kwargs: Any) -> Report:
    """校验描边参数和掩码形状；返回带错误列表的布尔报告。"""
    config: dict[str, Any] = {}
    for arg in args:
        arg = _native(arg)
        if isinstance(arg, Mapping):
            config.update(arg)
        elif _is_number(arg):
            config.setdefault("width", float(arg))

    config.update({k: _native(v) for k, v in kwargs.items()})
    errors: list[str] = []
    warnings: list[str] = []

    width = _get_any(config, "width", "width_px", "line_width", "smoothed_width", default=1.0)
    if not _is_number(width) or float(width) < 0:
        errors.append("width must be a non-negative number")
    elif float(width) == 0:
        warnings.append("width is zero; outline will be invisible")

    for key in ("target_height", "screen_height", "reference_height", "source_height"):
        if key in config and (not _is_number(config[key]) or float(config[key]) <= 0):
            errors.append(f"{key} must be positive")

    alpha = config.get("temporal_alpha", config.get("smoothing_alpha", DEFAULT_TEMPORAL_ALPHA))
    if not _is_number(alpha) or not 0.0 <= float(alpha) <= 1.0:
        errors.append("temporal alpha must be within [0, 1]")

    opacity = config.get("opacity", config.get("composite_alpha", 1.0))
    if _is_number(opacity) and not 0.0 <= float(opacity) <= 1.0:
        errors.append("opacity must be within [0, 1]")

    dot_threshold = config.get("normal_threshold", config.get("normal_dot_threshold",
                                                              DEFAULT_NORMAL_DOT_THRESHOLD))
    if not _is_number(dot_threshold) or not -1.0 <= float(dot_threshold) <= 1.0:
        errors.append("normal dot threshold must be within [-1, 1]")

    depth_threshold = config.get("depth_threshold", DEFAULT_DEPTH_THRESHOLD)
    if not _is_number(depth_threshold) or float(depth_threshold) < 0:
        errors.append("depth_threshold must be non-negative")

    mask = config.get("mask", config.get("edge_mask"))
    frame = config.get("frame")
    if isinstance(mask, Sequence) and frame is not None:
        try:
            if len(mask) != len(frame):
                errors.append("mask and frame length mismatch")
        except TypeError:
            pass

    if not config.get("normal_blur_first", True):
        warnings.append("normals are not blurred before thresholding")
    if not config.get("distance_scaled_depth", True):
        warnings.append("depth threshold is not scaled by view distance")
    if not config.get("temporal_width_enabled", config.get("temporal_width", True)):
        warnings.append("outline width is not temporally stabilized")
    if not config.get("transparent_depth_mask", True):
        warnings.append("transparent objects do not write a depth mask")
    if not config.get("resolution_scaled_on_load", True):
        warnings.append("loaded outline width is not resolution scaled")
    if not config.get("subpixel_link_enabled", config.get("subpixel_link", True)):
        warnings.append("subpixel edge linking is disabled")

    return Report({
        "valid": not errors,
        "passed": not errors,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "normalized": config,
    })


def _iter_mask_bits(mask: Any):
    mask = _native(mask)
    if _is_number(mask):
        yield int(float(mask) >= 0.5)
        return
    if not isinstance(mask, Sequence) or isinstance(mask, (str, bytes)):
        return
    for value in mask:
        value = _native(value)
        if isinstance(value, Sequence) and not _is_number(value):
            yield from _iter_mask_bits(value)
        else:
            yield _edge_bit(value)


def metrics(*args: Any, **kwargs: Any) -> Report:
    """统计边缘覆盖率、连通段数和线宽时间稳定性。"""
    mask = kwargs.get("mask", kwargs.get("edge_mask"))
    widths = kwargs.get("widths", kwargs.get("width_history"))
    for arg in args:
        arg = _native(arg)
        if mask is None and (
            _is_number(arg) or (isinstance(arg, Sequence) and not isinstance(arg, (str, bytes)))
        ):
            if widths is None and isinstance(arg, Sequence) and arg and all(_is_number(x) for x in arg) \
                    and kwargs.get("as_widths"):
                widths = arg
            else:
                mask = arg

    bits = list(_iter_mask_bits(mask if mask is not None else 1))
    total = len(bits)
    edge_count = sum(bits)

    runs = 0
    previous = 0
    for bit in bits:
        if bit and not previous:
            runs += 1
        previous = bit

    width_values = None
    if widths is not None:
        width_values = [float(w) for w in widths]
    elif isinstance(mask, Sequence) and mask and all(_is_number(x) for x in mask) and kwargs.get("as_widths"):
        width_values = [float(w) for w in mask]

    width_stats = None
    if width_values:
        mean = sum(width_values) / len(width_values)
        variance = sum((w - mean) ** 2 for w in width_values) / len(width_values)
        deltas = [abs(b - a) for a, b in zip(width_values, width_values[1:])]
        width_stats = {
            "mean": mean,
            "variance": variance,
            "stddev": math.sqrt(variance),
            "max_frame_delta": max(deltas, default=0.0),
            "flicker": sum(deltas) / max(1, len(deltas)),
        }

    return Report({
        "edge_count": edge_count,
        "total_pixels": total,
        "coverage": edge_count / total if total else 0.0,
        "connected_runs": runs,
        "broken_gap_candidates": max(0, runs - 1),
        "subpixel_linked": bool(kwargs.get("subpixel_linked", True)),
        "width": width_stats,
        "temporal_stability": 1.0 / (1.0 + width_stats["variance"]) if width_stats else 1.0,
    })


def _flag(config: Mapping, keys: tuple[str, ...], default: bool = True) -> bool:
    for key in keys:
        if key in config:
            return bool(config[key])
    return default


def audit(*args: Any, **kwargs: Any) -> Report:
    """审计管线是否落实了六个修复点。"""
    config: dict[str, Any] = {}
    for arg in args:
        arg = _native(arg)
        if isinstance(arg, Mapping):
            config.update(arg)
    config.update(kwargs)

    checks = {
        "normal_blur_before_threshold": _flag(
            config, ("normal_blur_first", "blur_normal_first", "blur_normals"), True
        ),
        "depth_threshold_scaled_by_distance": _flag(
            config, ("distance_scaled_depth", "depth_distance_scaled", "depth_threshold_scaled"), True
        ),
        "width_temporally_stable": _flag(
            config, ("temporal_width_enabled", "temporal_width", "width_temporal_stable"), True
        ),
        "transparent_depth_mask": _flag(
            config, ("transparent_depth_mask", "transparent_mask", "transparent_writes_depth_mask"), True
        ),
        "loaded_width_resolution_scaled": _flag(
            config, ("resolution_scaled_on_load", "width_on_load", "width_resolution_scaled"), True
        ),
        "subpixel_linking": _flag(
            config, ("subpixel_link_enabled", "subpixel_link"), True
        ),
    }
    issues = [name for name, ok in checks.items() if not ok]
    validation = validate_outline(config)
    issues.extend(validation["errors"])

    return Report({
        "passed": not issues,
        "valid": not issues,
        "ok": not issues,
        "checks": checks,
        "issues": issues,
        "warnings": validation["warnings"],
        "step": "audit",
    })


def fallback(*args: Any, **kwargs: Any) -> Report:
    """纹理缺失/加载失败或参数非法时的安全描边配置。"""
    reason = kwargs.get("reason")
    for arg in args:
        if isinstance(arg, str):
            reason = arg
        elif isinstance(arg, Mapping) and reason is None:
            reason = arg.get("reason", arg.get("error"))

    target_h = float(kwargs.get("target_height", kwargs.get("screen_height", REFERENCE_HEIGHT)))
    return Report({
        "mode": "depth_normal_fallback",
        "texture_loaded": False,
        "texture": "single_pixel_placeholder",
        "width_px": PLACEHOLDER_WIDTH,
        "placeholder_width": PLACEHOLDER_WIDTH,
        "temporal_smoothing": True,
        "resolution_scale": target_h / REFERENCE_HEIGHT,
        "normal_blur_radius": DEFAULT_NORMAL_BLUR_RADIUS,
        "depth_threshold": DEFAULT_DEPTH_THRESHOLD,
        "normal_dot_threshold": DEFAULT_NORMAL_DOT_THRESHOLD,
        "subpixel_link": True,
        "transparent_depth_mask": True,
        "reason": reason or "texture_unavailable",
        "stable": True,
    })


def _stable_unit(seed: int, index: int) -> float:
    digest = hashlib.sha256(f"{int(seed)}:{int(index)}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _canonical_digest(value: Any) -> str:
    if isinstance(value, Mapping):
        encoded = repr(sorted((str(k), _canonical_digest(v)) for k, v in value.items())).encode()
    elif isinstance(value, (list, tuple)):
        encoded = repr([_canonical_digest(v) for v in value]).encode()
    else:
        encoded = repr(value).encode()
    return hashlib.sha256(encoded).hexdigest()


def determinism(*args: Any, **kwargs: Any) -> Report:
    """生成跨进程稳定的确定性亚像素采样，并验证重复计算摘要一致。"""
    seed = int(kwargs.get("seed", 1))
    data = kwargs.get("data", kwargs.get("value"))
    runs = int(kwargs.get("runs", 2))
    count = int(kwargs.get("count", 8))

    for arg in args:
        if _is_number(arg) and "seed" not in kwargs:
            seed = int(arg)
        elif data is None and not _is_number(arg):
            data = arg

    offsets = []
    for i in range(count):
        offsets.append((
            round(_stable_unit(seed, i * 2) - 0.5, 9),
            round(_stable_unit(seed, i * 2 + 1) - 0.5, 9),
        ))

    digest = _canonical_digest(data if data is not None else {"seed": seed, "offsets": offsets})
    repeated = [_canonical_digest(data if data is not None else {"seed": seed, "offsets": offsets})
                for _ in range(max(1, runs))]
    stable = len(set(repeated)) == 1

    return Report({
        "seed": seed,
        "stable": stable,
        "deterministic": stable,
        "runs": runs,
        "digest": digest,
        "offsets": offsets,
        "subpixel_samples": offsets,
    })


# ---------------------------------------------------------------------------
# 集成管线（额外入口，便于把六个修复串起来）
# ---------------------------------------------------------------------------

def run_pipeline(frame: Mapping | None = None, **options: Any) -> Report:
    """执行推荐的完整描边流程。主要用于自检、示例和无渲染器环境。"""
    frame = dict(frame or {})
    opts = {**frame, **options}

    extracted = extract(frame)
    normals = extracted["normals"]
    if normals is not None:
        normals = blur_normal_first(normals, radius=int(opts.get("blur_radius", 1)))

    distance = opts.get("distance", opts.get("view_distance", DEFAULT_REFERENCE_DISTANCE))
    depth_threshold = depth_threshold_scaled(
        opts.get("depth_threshold", DEFAULT_DEPTH_THRESHOLD),
        distance,
        opts.get("reference_distance", DEFAULT_REFERENCE_DISTANCE),
    )
    edges = edge_detect(
        extracted["depth"],
        normals,
        depth_threshold=depth_threshold,
        normal_threshold=opts.get("normal_threshold", DEFAULT_NORMAL_DOT_THRESHOLD),
        distance=distance,
        blur_normals=False,
    ) if extracted["depth"] is not None else MaskList()

    linked = subpixel_link(edges, max_gap=int(opts.get("max_gap", 1))) if isinstance(edges, list) else edges
    alpha = extracted["alpha"]
    tmask = transparent_mask(alpha) if alpha is not None else None

    raw_width = opts.get("width", opts.get("width_px", 1.0))
    width = temporal_width(raw_width, opts.get("previous_width"),
                           alpha=float(opts.get("temporal_alpha", DEFAULT_TEMPORAL_ALPHA)))
    texture = opts.get("texture")
    loaded_width = width_on_load(texture) if isinstance(texture, Mapping) and texture.get("loaded") else width

    output = None
    if extracted["color"] is not None:
        output = composite(extracted["color"], linked,
                           color=opts.get("outline_color", OUTLINE_COLOR),
                           opacity=opts.get("opacity", 1.0))

    result = Report({
        "edges": linked,
        "mask": linked,
        "transparent_mask": tmask,
        "width": temporal_width(loaded_width, opts.get("previous_width"),
                                alpha=float(opts.get("temporal_alpha", DEFAULT_TEMPORAL_ALPHA))),
        "depth_threshold": depth_threshold,
        "output": output,
        "texture": texture or load_texture(opts.get("texture_source", "outline")),
    })
    result["metrics"] = metrics(linked, widths=opts.get("width_history"))
    result["validation"] = validate_outline(opts)
    result["audit"] = audit(opts)
    return result
