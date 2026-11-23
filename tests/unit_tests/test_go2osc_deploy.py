import math
from types import SimpleNamespace

import pytest
import torch

from gym.envs.go2.go2osc_config import Go2OscCfg, Go2OscRunnerCfg

from go2_deploy.deploy_config import DeployConfig
from go2_deploy.utility.import_custom_impl import load_custom_impl


def test_go2osc_deploy_observation_contract():
    cfg = DeployConfig(load_custom_impl("go2osc"))

    assert cfg.task_name == "go2osc"
    assert cfg.obs_vector == list(Go2OscRunnerCfg.actor.obs)
    assert cfg.obs_dim() == 57
    assert cfg.custom_impl.custom_obs_sizes == {
        "phase_obs": 8,
        "phase_frequency": 4,
    }
    assert "measured_grf" not in cfg.obs_vector
    assert "measured_grf" in Go2OscRunnerCfg.critic.obs
    assert cfg.default_dof_pos.tolist() == pytest.approx([0.0, 0.66, -1.36] * 4)

    for name in cfg.obs_vector:
        assert getattr(cfg.DeployScaling, name, 1.0) == getattr(
            Go2OscCfg.scaling, name, 1.0
        )


def test_go2osc_buffers_advance_and_expose_observations():
    custom = load_custom_impl("go2osc")
    controller = SimpleNamespace(**custom.get_custom_buffers_dict())
    initial_phase = controller.phase.clone()
    offsets = 2 * math.pi * torch.tensor([0.0, 0.25, 0.5, 0.75])
    assert torch.allclose(
        torch.remainder(initial_phase - initial_phase[0], 2 * math.pi),
        offsets,
        atol=1e-6,
    )

    custom.update_custom_buffers(controller, 10.0)
    custom.update_custom_buffers(controller, 10.02)
    expected_phase = torch.remainder(initial_phase + 2 * math.pi * 2.0 * 0.02, 2 * math.pi)
    assert torch.allclose(controller.phase, expected_phase, atol=1e-5)

    pieces = custom.get_custom_obs_pieces()
    for name, getter in pieces.items():
        assert getter(controller, None).shape == (
            custom.custom_obs_sizes[name],
        )
    assert torch.allclose(
        pieces["phase_obs"](controller, None),
        torch.cat((expected_phase.sin(), expected_phase.cos())),
    )

    controller.phase_frequency.zero_()
    custom.update_custom_buffers(controller, 10.04)
    assert torch.allclose(controller.phase, expected_phase, atol=1e-5)
