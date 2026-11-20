"""Offline adapter checks; force readings here are calibrated synthetic data."""

import math
import threading
from types import SimpleNamespace

import pytest
import torch

from go2_deploy.custom_implementation import pd_emergent as adapter
from go2_deploy.utility import deploy_utility
from gym.envs.go2.go2osc import Go2Osc
from gym.envs.go2.go2osc_config import Go2OscCfg


@pytest.fixture(autouse=True)
def clear_calibration_environment(monkeypatch):
    for name in (
        "GO2_FORCE_SENSOR_ORDER",
        "GO2_FORCE_NEWTONS_PER_UNIT",
        "GO2_FORCE_ZERO_OFFSET",
    ):
        monkeypatch.delenv(name, raising=False)


def _controller(monkeypatch, speed=1.0):
    monkeypatch.setenv("GO2_FORCE_SENSOR_ORDER", "RL,FR,RR,FL")
    monkeypatch.setenv("GO2_FORCE_NEWTONS_PER_UNIT", "1")
    return SimpleNamespace(
        **adapter.get_custom_buffers_dict(),
        cfg=SimpleNamespace(ctrl_freq=100),
        last_lowstate_msg=SimpleNamespace(foot_force=[60, 30, 50, 40]),
        last_lowstate_msg_lock=threading.Lock(),
        last_command=torch.tensor([speed, 0.0, 0.0]),
        last_command_lock=threading.Lock(),
    )


def test_unconfirmed_force_calibration_is_required_before_startup(monkeypatch):
    with pytest.raises(ValueError, match="GO2_FORCE_SENSOR_ORDER"):
        adapter.get_custom_buffers_dict()
    monkeypatch.setenv("GO2_FORCE_SENSOR_ORDER", "FR,FL,RR,RL")
    with pytest.raises(ValueError, match="GO2_FORCE_NEWTONS_PER_UNIT"):
        adapter.get_custom_buffers_dict()


@pytest.mark.parametrize("order", ["FR,FR,RR,RL", "FR,FL,RR", "FR,FL,RR,UNKNOWN"])
def test_force_order_rejects_missing_or_duplicate_legs(monkeypatch, order):
    monkeypatch.setenv("GO2_FORCE_SENSOR_ORDER", order)
    monkeypatch.setenv("GO2_FORCE_NEWTONS_PER_UNIT", "1")
    with pytest.raises(ValueError, match="GO2_FORCE_SENSOR_ORDER"):
        adapter.get_custom_buffers_dict()


@pytest.mark.parametrize("conversion", ["0", "1,2", "1,nan,1,1", "1,0,1,1"])
def test_force_conversion_rejects_invalid_values(monkeypatch, conversion):
    monkeypatch.setenv("GO2_FORCE_SENSOR_ORDER", "FR,FL,RR,RL")
    monkeypatch.setenv("GO2_FORCE_NEWTONS_PER_UNIT", conversion)
    with pytest.raises(ValueError, match="GO2_FORCE_NEWTONS_PER_UNIT"):
        adapter.get_custom_buffers_dict()


def test_force_conversion_preserves_per_sensor_sign_offsets_and_leg_mapping(
    monkeypatch,
):
    controller = _controller(monkeypatch)
    monkeypatch.setenv("GO2_FORCE_NEWTONS_PER_UNIT", "-2,1,-0.5,0.5")
    monkeypatch.setenv("GO2_FORCE_ZERO_OFFSET", "0,10,0,20")
    for name, value in adapter.get_custom_buffers_dict().items():
        setattr(controller, name, value)
    controller.last_lowstate_msg.foot_force = [-30, 40, -100, 100]

    adapter.update_custom_buffers(controller, 0.0)

    expected = torch.tensor([30.0, 40.0, 50.0, 60.0]) / (16.087 * 9.81)
    torch.testing.assert_close(controller.grf, expected)


@pytest.mark.parametrize("speed", [0.0, 0.5, 1.0, -1.0, 1.23, 2.0, 5.0])
def test_phase_dynamics_match_training_under_identical_held_forces(monkeypatch, speed):
    controller = _controller(monkeypatch, speed)
    # Exercise training equations without constructing a simulator or SDK client.
    reference = Go2Osc.__new__(Go2Osc)
    reference.cfg = SimpleNamespace(osc=Go2OscCfg.osc)
    reference.commands = controller.last_command.reshape(1, 3)
    state_names = (
        "oscillators",
        "oscillators_vel",
        "oscillator_obs",
        "grf",
        "osc_omega",
        "osc_coupling",
        "osc_offset",
    )
    for name in state_names:
        setattr(reference, name, getattr(controller, name).clone().unsqueeze(0))
    normalized_force = torch.tensor([[30.0, 40.0, 50.0, 60.0]]) / (16.087 * 9.81)
    reference._compute_grf = lambda: normalized_force.clone()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        reference._update_oscillator_parameters()
        for _ in range(5):
            reference._step_oscillators(dt=0.002)
        torch.manual_seed(17)
        adapter.update_custom_buffers(controller, 0.0)

    for name in state_names:
        torch.testing.assert_close(
            getattr(controller, name),
            getattr(reference, name).squeeze(0),
            atol=2e-6,
            rtol=2e-6,
        )


def test_actor_observation_layout_scaling_and_previous_action(monkeypatch):
    controller = _controller(monkeypatch)
    custom = adapter.CustomDeployConfig
    controller.cfg = SimpleNamespace(
        ctrl_freq=100,
        obs_vector=custom.obs_vector,
        obs_sizes={
            "base_ang_vel": 3,
            "projected_gravity": 3,
            "commands": 3,
            "dof_pos_obs": 12,
            "dof_vel": 12,
            "oscillator_obs": 8,
            "dof_pos_target": 12,
        },
        DeployScaling=custom.DeployScaling,
        default_dof_pos=custom.default_dof_pos,
    )
    controller.obs_vec_size = 53
    controller.last_action = torch.tensor([[0.2, 0.3, 0.3] * 4, [0.0] * 12])
    message = controller.last_lowstate_msg
    message.imu_state = SimpleNamespace(
        gyroscope=[0.3, 0.6, 0.3], quaternion=[1.0, 0.0, 0.0, 0.0]
    )
    # Identical per-leg joint positions still exercise hip/thigh/calf scales.
    message.motor_state = [
        SimpleNamespace(q=q + 0.05, dq=dq)
        for q, dq in zip([0.0, 0.66, -1.36] * 4, [2.0, 2.0, 4.0] * 4)
    ]
    controller.oscillators.copy_(
        torch.tensor([0.0, math.pi / 2, math.pi, 1.5 * math.pi])
    )
    controller.oscillator_obs.copy_(
        torch.cat((controller.oscillators.cos(), controller.oscillators.sin()))
    )
    monkeypatch.setattr(deploy_utility, "custom_impl", adapter)

    actual = deploy_utility.lowstate_to_obs(controller, message)

    expected = torch.cat(
        (
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([0.0, 0.0, -1.0]),
            torch.tensor([1.0 / 3, 0.0, 0.0]),
            torch.tensor([0.05 / 0.2, 0.05 / 0.3, 0.05 / 0.3] * 4),
            torch.ones(12),
            torch.tensor([1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0]),
            torch.ones(12),
        )
    )
    assert actual.shape == (53,)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
