"""Analysis functions of scripts/evaluate_symmetry.py on synthetic gaits."""

import math

import pytest
import torch

from gym.envs.go2.go2_config import GO2_DOF_NAMES
from learning.utils.symmetry import GO2_SIDE_PAIRS, GO2_SIGN_FLIP_KEYWORDS
from learning.utils.symmetry import Mirror, joint_mirror
from scripts.evaluate_symmetry import (
    GAIT_TEMPLATES,
    classify_gait,
    components_for,
    mirror_shift_error,
    pca_explained,
    relative_phases,
    stride_frequency,
)

DT = 0.01
FREQ = 2.0


def _contacts(lags, duty=0.5, seconds=6.0):
    """[time, 4] contact signals; foot i lags FL by lags[i] cycles."""
    t = torch.arange(int(seconds / DT)) * DT
    phase = (FREQ * t[:, None] - torch.tensor(lags)) % 1.0
    return phase < duty


@pytest.mark.parametrize("gait", list(GAIT_TEMPLATES))
def test_gait_templates_are_recovered(gait):
    contact = _contacts((0.0,) + GAIT_TEMPLATES[gait])
    f = stride_frequency(contact, DT)
    assert f == pytest.approx(FREQ, abs=0.2)
    name, dist = classify_gait(relative_phases(contact, DT, f))
    assert name == gait
    assert dist < 0.05


def test_irregular_phases_are_not_forced_into_a_template():
    assert classify_gait([0.25, 0.25, 0.75])[0] == "irregular"


def test_relative_phase_sign_is_lag():
    contact = _contacts((0.0, 0.25, 0.5, 0.75))
    rel = relative_phases(contact, DT, FREQ)
    assert rel == pytest.approx([0.25, 0.5, 0.75], abs=0.02)


def _mirror():
    return Mirror(*joint_mirror(GO2_DOF_NAMES, GO2_SIDE_PAIRS, GO2_SIGN_FLIP_KEYWORDS))


def _trot_joints(limp=0.0, seconds=6.0):
    """Joint angles of an ideal trot; ``limp`` scales the right legs' swing."""
    t = torch.arange(int(seconds / DT)) * DT
    lag = {"FL": 0.0, "FR": 0.5, "RL": 0.5, "RR": 0.0}
    cols = []
    for name in GO2_DOF_NAMES:
        leg, joint = name.split("_")[0], name.split("_")[1]
        s = torch.sin(2 * math.pi * (FREQ * t - lag[leg]))
        amp = {"hip": 0.05, "thigh": 0.3, "calf": 0.4}[joint]
        if leg in ("FR", "RR"):
            amp *= 1.0 - limp
        sign = -1.0 if (joint == "hip" and leg in ("FR", "RR")) else 1.0
        cols.append(sign * amp * s + {"hip": 0.0, "thigh": 0.7, "calf": -1.4}[joint])
    return torch.stack(cols, dim=1)


def test_symmetric_trot_matches_mirror_after_half_stride():
    q = _trot_joints()
    half = round(0.5 / (FREQ * DT))
    m = _mirror()
    assert mirror_shift_error(q, m, half) < 1e-3
    assert mirror_shift_error(q, m, 0) > 0.5


def test_limp_is_asymmetric_for_every_shift():
    q = _trot_joints(limp=0.5)
    half = round(0.5 / (FREQ * DT))
    m = _mirror()
    assert min(mirror_shift_error(q, m, 0), mirror_shift_error(q, m, half)) > 0.05


def test_pca_counts_independent_signals():
    t = torch.arange(1000) * DT
    a, b = torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * 1.7 * t)
    x = torch.stack([a, 2 * a, b, -b, a + b], dim=1)
    ratios = pca_explained(x)
    assert sum(ratios[:2]) == pytest.approx(1.0, abs=1e-9)
    assert components_for(ratios, 0.9) <= 2
