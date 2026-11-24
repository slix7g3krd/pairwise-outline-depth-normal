# pairwise-outline-depth-normal

## 领域

游戏、图形与多媒体

## 主题

屏幕空间描边双阈值

## 说明

法线预模糊、视距换算与透明掩码。

初始状态下 `graphics/outline_depth_normal.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
