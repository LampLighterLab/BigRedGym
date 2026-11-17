import torch

from scripts.evaluate_go2height_readiness import first_sustained


def test_first_sustained_requires_the_full_window():
    mask = torch.tensor([False, True, True, False, True, True, True])
    assert first_sustained(mask, sustain_steps=3, dt=0.1) == 0.4


def test_first_sustained_returns_none_when_window_is_missing():
    assert first_sustained(torch.tensor([True, False, True]), 2, 0.1) is None
