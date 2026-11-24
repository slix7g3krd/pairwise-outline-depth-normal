import pytest
import graphics.outline_depth_normal as mod

CASES = [
    ("blur_normal_first", (1,)),
    ("depth_threshold_scaled", (1,)),
    ("temporal_width", (1,)),
    ("transparent_mask", (1,)),
    ("width_on_load", (1,)),
    ("subpixel_link", (1,)),
    ("extract", (1,)),
    ("edge_detect", (1,)),
    ("composite", (1,)),
    ("load_texture", (1,)),
    ("validate_outline", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
