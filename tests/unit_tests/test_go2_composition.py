from scripts.evaluate_go2_composition import blend_alpha


def test_direct_switch_and_b_alone_alpha():
    assert blend_alpha("direct", 0.0, -0.01) == 0.0
    assert blend_alpha("direct", 0.0, 0.0) == 1.0
    assert blend_alpha("b_alone", None, -1.0) == 1.0


def test_blend_alpha_clamps_at_one():
    assert blend_alpha("blend_050", 0.5, 0.0) == 0.0
    assert blend_alpha("blend_050", 0.5, 0.25) == 0.5
    assert blend_alpha("blend_050", 0.5, 1.0) == 1.0
