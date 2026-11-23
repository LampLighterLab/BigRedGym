import math
import threading
from types import SimpleNamespace

import pytest
import torch

from gym.envs.go2.go2osc_config import Go2OscCfg, Go2OscRunnerCfg

from go2_deploy.deploy_config import DeployConfig
from go2_deploy.keyboard_handler import KeyboardHandler
from go2_deploy.utility.import_custom_impl import load_custom_impl


def test_go2osc_deploy_observation_contract():
    cfg = DeployConfig(load_custom_impl("go2osc"))

    assert cfg.task_name == "go2osc"
    assert cfg.obs_vector == list(Go2OscRunnerCfg.actor.obs)
    assert cfg.obs_dim() == 61
    assert cfg.custom_impl.custom_obs_sizes == {
        "phase_obs": 8,
        "phase_frequency": 4,
        "measured_grf": 4,
    }
    assert "measured_grf" in cfg.obs_vector
    assert "measured_grf" in Go2OscRunnerCfg.critic.obs
    assert cfg.default_dof_pos.tolist() == pytest.approx([0.0, 0.66, -1.36] * 4)

    for name in cfg.obs_vector:
        assert getattr(cfg.DeployScaling, name, 1.0) == getattr(
            Go2OscCfg.scaling, name, 1.0
        )


def test_go2osc_buffers_advance_and_expose_observations():
    custom = load_custom_impl("go2osc")
    custom.force_calibration = {
        "offsets": [10.0, 20.0, 30.0, 40.0],
        "gains": [1.0, 1.0, 1.0, 1.0],
        "order": [1, 0, 3, 2],
        "mass_kg": 10.0,
    }
    controller = SimpleNamespace(
        **custom.get_custom_buffers_dict(),
        last_lowstate_msg_lock=threading.Lock(),
        last_lowstate_msg=SimpleNamespace(foot_force=[20, 40, 60, 80]),
    )
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
    assert torch.allclose(
        controller.measured_grf, torch.tensor([20.0, 10.0, 40.0, 30.0]) / 98.1
    )

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

    assert custom.adjust_frequency(controller, 1) == 0.25
    custom.update_custom_buffers(controller, 10.06)
    expected_phase = torch.remainder(
        expected_phase + 2 * math.pi * 0.25 * 0.02, 2 * math.pi
    )
    assert torch.allclose(controller.phase, expected_phase, atol=1e-5)


def test_go2osc_requires_foot_force_calibration():
    custom = load_custom_impl("go2osc")
    custom.force_calibration = None
    controller = SimpleNamespace(**custom.get_custom_buffers_dict())
    with pytest.raises(RuntimeError, match="calibration"):
        custom.update_custom_buffers(controller, 10.0)


def test_go2osc_default_force_baseline():
    custom = load_custom_impl("go2osc")
    assert custom.force_calibration["mass_kg"] == 15.0
    normalized = custom.normalize_foot_forces(
        [15.0 * 9.81 / 4] * 4, custom.force_calibration
    )
    assert torch.allclose(normalized, torch.full((4,), 0.25))


def test_deployment_keyboard_changes_all_leg_frequencies(monkeypatch):
    custom = load_custom_impl("go2osc")
    controller = SimpleNamespace(
        **custom.get_custom_buffers_dict(),
        cfg=SimpleNamespace(custom_impl=custom),
    )
    handler = object.__new__(KeyboardHandler)
    handler.controller = controller
    initial_phase = controller.phase.clone()

    for key, expected in (("u", 2.25), ("d", 2.0)):
        monkeypatch.setattr("builtins.input", lambda: key)
        handler._process_input()
        assert torch.equal(
            controller.phase_frequency, torch.full((4,), expected)
        )

    controller.phase_frequency.fill_(5.0)
    monkeypatch.setattr("builtins.input", lambda: "u")
    handler._process_input()
    assert torch.equal(controller.phase_frequency, torch.full((4,), 5.0))

    controller.phase_frequency.zero_()
    monkeypatch.setattr("builtins.input", lambda: "d")
    handler._process_input()
    assert torch.equal(controller.phase_frequency, torch.zeros(4))
    assert torch.equal(controller.phase, initial_phase)
