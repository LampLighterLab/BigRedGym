import math
from types import SimpleNamespace

import torch

from gym.envs.go2.go2osc_config import Go2OscCfg, Go2OscRunnerCfg
from go2_deploy.custom_implementation.phase1.go2osc_deploy import (
    CustomDeployConfig,
    advance_oscillators,
    get_custom_buffers_dict,
    get_custom_obs_pieces,
    normalize_foot_forces,
)


def test_oscillator_update_matches_simulation():
    controller = SimpleNamespace(**get_custom_buffers_dict())
    initial_phase = controller.phase.clone()
    expected_offsets = 2 * math.pi * torch.tensor([0.0, 0.25, 0.5, 0.75])
    assert torch.allclose(
        torch.remainder(initial_phase - initial_phase[0], 2 * math.pi),
        expected_offsets,
        atol=1e-6,
    )
    grf = torch.tensor([0.2, 0.3, 0.4, 0.5])
    dt = 0.02
    advance_oscillators(controller, grf, dt=dt)
    expected = torch.remainder(
        initial_phase + 2 * math.pi * Go2OscCfg.osc.frequency * dt,
        2 * math.pi,
    )
    assert torch.allclose(controller.phase, expected)
    assert torch.allclose(controller.phase_frequency, torch.full((4,), 2.0))
    assert torch.allclose(controller.measured_grf, grf)
    assert torch.allclose(
        controller.phase_obs, torch.cat((expected.sin(), expected.cos()))
    )


def test_grf_does_not_change_phase_rate():
    first = SimpleNamespace(**get_custom_buffers_dict())
    second = SimpleNamespace(**get_custom_buffers_dict())
    second.phase = first.phase.clone()
    advance_oscillators(first, torch.zeros(4), 0.02)
    advance_oscillators(second, torch.ones(4), 0.02)
    assert torch.allclose(first.phase, second.phase)
    assert not torch.equal(first.measured_grf, second.measured_grf)


def test_observation_contract():
    assert CustomDeployConfig.obs_vector == list(Go2OscRunnerCfg.actor.obs)
    assert CustomDeployConfig.obs_vector[-3:] == [
        "phase_obs", "phase_frequency", "measured_grf"
    ]
    sizes = CustomDeployConfig.custom_obs_sizes
    assert sum(sizes.values()) == 16
    assert "target_grf" not in get_custom_obs_pieces()
    controller = SimpleNamespace(**get_custom_buffers_dict())
    for name, getter in get_custom_obs_pieces().items():
        assert getter(controller, None).shape == (sizes[name],)


def test_force_normalization():
    grf = normalize_foot_forces(
        raw=[20, 40, 60, 80],
        offsets=[10, 20, 30, 40],
        gains=[1, 1, 1, 1],
        order=[1, 0, 3, 2],
        mass_kg=10.0,
    )
    expected = torch.tensor([20, 10, 40, 30]) / 98.1
    assert torch.allclose(grf, expected)
