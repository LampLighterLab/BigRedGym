"""The course-deploy custom implementation must match the go2payload policy.

The file is self-contained (it is run inside the course deploy tree, whose
Go2TrotCfg differs), so these tests pin every value to the training config.
"""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gym.envs.go2.go2payload_config import Go2PayloadCfg, Go2PayloadRunnerCfg

IMPL_PATH = (
    Path(__file__).resolve().parents[2]
    / "go2_deploy"
    / "custom_implementation"
    / "heavy_weights.py"
)


@pytest.fixture(scope="module")
def impl():
    spec = importlib.util.spec_from_file_location(IMPL_PATH.stem, IMPL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_observation_layout_matches_trained_actor(impl):
    cfg = impl.CustomDeployConfig
    assert cfg.obs_vector == Go2PayloadRunnerCfg.actor.obs
    standard = {
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "commands": 3,
        "dof_pos_obs": 12,
        "dof_vel": 12,
        "dof_pos_target": 12,
    }
    sizes = standard | cfg.custom_obs_sizes
    assert sum(sizes[name] for name in cfg.obs_vector) == 48


def test_scaling_matches_training_for_every_observation_and_the_action(impl):
    scaling = impl.CustomDeployConfig.DeployScaling
    for name in impl.CustomDeployConfig.obs_vector + ["dof_pos_target"]:
        trained = getattr(Go2PayloadCfg.scaling, name, 1.0)
        assert getattr(scaling, name) == trained, name


def test_listed_training_settings_match_the_task(impl):
    cfg = impl.CustomDeployConfig
    control = Go2PayloadCfg.control
    assert cfg.trained_ctrl_freq == control.ctrl_frequency
    assert set(control.stiffness.values()) == {cfg.trained_kp}
    assert set(control.damping.values()) == {cfg.trained_kd}
    angles = Go2PayloadCfg.init_state.default_joint_angles
    assert cfg.trained_default_joint_angles == {
        "hip": angles["hip_joint"],
        "thigh": angles["FL_thigh_joint"],
        "calf": angles["calf_joint"],
    }
    ranges = Go2PayloadCfg.commands.ranges
    limits = cfg.command_limits
    assert limits["lin_vel_x"] <= min(-min(ranges.lin_vel_x), max(ranges.lin_vel_x))
    assert limits["lin_vel_y"] <= ranges.lin_vel_y
    assert limits["yaw_vel"] <= ranges.yaw_vel
    low, high = control.gait_freq
    assert low <= impl.get_custom_buffers_dict()["phase_frequency"] <= high


def test_phase_buffers_and_observation_pieces(impl):
    controller = SimpleNamespace(**impl.get_custom_buffers_dict())
    pieces = impl.get_custom_obs_pieces()
    assert set(pieces) == set(impl.CustomDeployConfig.custom_obs_sizes)

    # A quarter of a 2 Hz cycle is 0.125 s.
    impl.update_custom_buffers(controller, 10.125)
    assert controller.phase == pytest.approx(math.pi / 2)
    torch.testing.assert_close(
        pieces["phase_obs"](controller, None),
        torch.tensor([1.0, 0.0]),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        pieces["phase_frequency"](controller, None), torch.tensor([2.0])
    )
    for name, piece in pieces.items():
        assert piece(controller, None).shape == (
            impl.CustomDeployConfig.custom_obs_sizes[name],
        )
