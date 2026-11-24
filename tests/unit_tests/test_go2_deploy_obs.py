"""Current deployment utilities, independent of the Unitree SDK/DDS stack.

These tests check deployment observation and action utilities with synthetic
messages; they do not compare deployment against a simulated task.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from go2_deploy.deploy_config import DeployConfig
from go2_deploy.utility import deploy_utility
from go2_deploy.utility.import_custom_impl import load_custom_impl

SDK_IN_QGYM_ORDER = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]


def _controller(cfg=None):
    if cfg is None:
        cfg = DeployConfig(load_custom_impl("go2trot"))
    return SimpleNamespace(
        cfg=cfg,
        obs_vec_size=sum(cfg.obs_sizes[name] for name in cfg.obs_vector),
        last_command=torch.tensor([0.5, -0.25, 0.75]),
        last_action=torch.zeros(2, 12),
        phase=0.4,
        phase_frequency=cfg.custom_impl.get_custom_buffers_dict()["phase_frequency"],
    )


def _message():
    return SimpleNamespace(
        motor_state=[
            SimpleNamespace(q=float(i), dq=-2.0 * i, ddq=3.0 * i) for i in range(12)
        ],
        imu_state=SimpleNamespace(
            quaternion=[math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0],
            gyroscope=[0.1, -0.2, 0.3],
        ),
    )


def test_joint_mapping_for_positions_velocities_and_accelerations():
    controller, message = _controller(), _message()
    controller.cfg.default_dof_pos = torch.linspace(0.1, 1.2, 12)
    expected = torch.tensor(SDK_IN_QGYM_ORDER, dtype=torch.float)
    torch.testing.assert_close(
        deploy_utility._get_obs_dof_pos_obs(controller, message),
        expected - controller.cfg.default_dof_pos,
    )
    torch.testing.assert_close(
        deploy_utility._get_obs_dof_vel(controller, message), -2 * expected
    )
    torch.testing.assert_close(
        deploy_utility._get_obs_dof_accel(controller, message), 3 * expected
    )
    canonical = torch.arange(12)
    sdk = canonical[deploy_utility.QGYM_TO_UNITREE_JOINT_IDX]
    assert sdk.tolist() == SDK_IN_QGYM_ORDER
    assert torch.equal(sdk[deploy_utility.UNITREE_TO_QGYM_JOINT_IDX], canonical)


def test_zero_action_and_nominal_posture_have_zero_relative_observations():
    controller, message = _controller(), _message()
    nominal = torch.tensor([0.0, 0.96, -1.36] * 4)
    torch.testing.assert_close(
        deploy_utility.action_to_target_pos(controller, torch.zeros(12)), nominal
    )
    for motor, position in zip(message.motor_state, nominal[SDK_IN_QGYM_ORDER]):
        motor.q = position.item()
    torch.testing.assert_close(
        deploy_utility._get_obs_dof_pos_obs(controller, message), torch.zeros(12)
    )


def test_wxyz_quaternion_projects_gravity_but_gyro_is_already_body_local():
    controller, message = _controller(), _message()
    # A positive quarter-turn about X maps world-down to body negative Y.
    torch.testing.assert_close(
        deploy_utility._get_obs_projected_gravity(controller, message),
        torch.tensor([0.0, -1.0, 0.0]),
        atol=2e-7,
        rtol=0,
    )
    torch.testing.assert_close(
        deploy_utility._get_obs_base_ang_vel(controller, message),
        torch.tensor([0.1, -0.2, 0.3]),
    )


def test_observation_dispatch_preserves_configured_order_sizes_and_scaling():
    cfg = DeployConfig(load_custom_impl("go2trot"))
    cfg.obs_vector = [
        "phase_frequency",
        "dof_vel",
        "base_ang_vel",
        "dof_pos_target",
        "commands",
        "projected_gravity",
        "dof_pos_obs",
        "dof_accel",
        "phase_obs",
    ]
    cfg.DeployScaling = SimpleNamespace(
        phase_frequency=4.0,
        dof_vel=2.0,
        base_ang_vel=0.5,
        dof_pos_target=0.25,
        commands=[2.0, 4.0, 8.0],
        projected_gravity=2.0,
        dof_pos_obs=list(range(1, 13)),
        dof_accel=3.0,
        phase_obs=2.0,
    )
    controller = _controller(cfg)
    controller.last_action[0] = torch.linspace(-6.0, 5.0, 12)
    ordered = torch.tensor(SDK_IN_QGYM_ORDER, dtype=torch.float)
    expected = torch.cat(
        (
            torch.tensor([controller.phase_frequency / 4]),
            -ordered,
            torch.tensor([0.2, -0.4, 0.6]),
            controller.last_action[0] / 0.25,
            torch.tensor([0.25, -0.0625, 0.09375]),
            torch.tensor([0.0, -0.5, 0.0]),
            (ordered - cfg.default_dof_pos) / torch.arange(1, 13),
            ordered,
            torch.tensor([math.sin(0.4), math.cos(0.4)]) / 2,
        )
    )

    actual = deploy_utility.lowstate_to_obs(controller, _message())

    assert actual.shape == (controller.obs_vec_size,)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-6)


def test_frequency_observation_tracks_active_controller_frequency():
    cfg = DeployConfig(load_custom_impl("go2trot"))
    cfg.obs_vector = ["phase_frequency"]
    cfg.DeployScaling = SimpleNamespace(phase_frequency=4.0)
    controller = _controller(cfg)

    # The oscillator's current frequency can differ from both config defaults.
    for frequency in (3.0, 4.5):
        controller.phase_frequency = frequency
        torch.testing.assert_close(
            deploy_utility.lowstate_to_obs(controller, _message()),
            torch.tensor([frequency / 4.0]),
        )


@pytest.mark.parametrize("task_name", ["go2trot", "go2"])
def test_action_target_roundtrip_uses_static_default_pose(task_name):
    controller = _controller()
    controller.cfg.task_name = task_name
    controller.cfg.default_dof_pos = torch.linspace(0.1, 1.2, 12)
    action = torch.linspace(-0.2, 0.2, 12)

    target = deploy_utility.action_to_target_pos(controller, action)

    torch.testing.assert_close(
        target,
        action + controller.cfg.default_dof_pos,
    )
    torch.testing.assert_close(
        deploy_utility.target_pos_to_action(controller, target), action
    )
    controller.phase += math.pi
    torch.testing.assert_close(
        deploy_utility.action_to_target_pos(controller, action), target
    )


def _lowcmd_controller(cfg=None):
    controller = _controller(cfg)
    controller.default_lowcmd = SimpleNamespace(
        motor_cmd=[SimpleNamespace(q=0.0, kp=0.0, kd=0.0) for _ in range(20)]
    )
    return controller


def test_existing_lowcmd_receives_permuted_targets_and_scaled_gains():
    controller = _lowcmd_controller()
    # Distinct in-range targets, so any permutation error is visible.
    canonical_target = controller.cfg.default_dof_pos + torch.linspace(-0.3, 0.3, 12)
    action = canonical_target - controller.cfg.default_dof_pos

    command = deploy_utility.action_to_lowcmd(
        controller, action, kp_mult=0.5, kd_mult=0.25
    )

    expected = canonical_target[SDK_IN_QGYM_ORDER]
    assert command is controller.default_lowcmd
    torch.testing.assert_close(
        torch.tensor([motor.q for motor in command.motor_cmd[:12]]), expected
    )
    assert all(motor.kp == controller.cfg.kp * 0.5 for motor in command.motor_cmd[:12])
    assert all(motor.kd == controller.cfg.kd * 0.25 for motor in command.motor_cmd[:12])
    assert all(motor.q == 0.0 for motor in command.motor_cmd[12:])


def test_deploy_controller_matches_training_controller():
    from gym.envs.go2.go2payload_config import Go2PayloadCfg

    cfg = DeployConfig(load_custom_impl("go2payload"))
    assert cfg.task_name == "go2payload"
    assert cfg.ctrl_freq == Go2PayloadCfg.control.ctrl_frequency == 50
    assert set(Go2PayloadCfg.control.stiffness.values()) == {cfg.kp}
    assert set(Go2PayloadCfg.control.damping.values()) == {cfg.kd}
    # Training applies no action smoothing, and the actor observes the result.
    assert not cfg.exp_moving_avg


@pytest.mark.parametrize(
    "name, limit",
    [
        # Symmetric joystick: 2.0 forward also means -2.0 in reverse, and
        # training only reverses to -1.0 m/s.
        ("lin_vel_x", 2.0),
        ("lin_vel_y", 1.0),
        ("yaw_vel", 3.0),
    ],
)
def test_command_limits_outside_training_are_rejected(name, limit):
    cfg = DeployConfig(load_custom_impl("go2payload"))
    cfg.command_limits[name] = limit
    with pytest.raises(ValueError, match=name):
        cfg.validate()


def test_gait_frequency_outside_training_is_rejected(monkeypatch):
    impl = load_custom_impl("go2payload")
    monkeypatch.setattr(
        impl, "get_custom_buffers_dict", lambda: {"phase_frequency": 4.0}
    )
    with pytest.raises(ValueError, match="gait_freq"):
        DeployConfig(impl)


def test_lowcmd_clamps_targets_to_urdf_limits():
    controller = _lowcmd_controller(DeployConfig(load_custom_impl("go2payload")))
    limits = controller.cfg.dof_pos_limits
    action = torch.full((12,), 10.0)
    action[::2] = -10.0

    command = deploy_utility.action_to_lowcmd(controller, action)

    expected = torch.where(action > 0, limits[:, 1], limits[:, 0])
    torch.testing.assert_close(
        torch.tensor([motor.q for motor in command.motor_cmd[:12]]),
        expected[SDK_IN_QGYM_ORDER],
    )


def test_urdf_limits_are_canonical_and_contain_nominal_pose():
    from gym import GYM_ROOT_DIR
    from gym.envs.base.urdf_limits import parse_urdf_position_limits
    from gym.envs.go2.go2payload_config import Go2PayloadCfg

    cfg = DeployConfig(load_custom_impl("go2payload"))
    lower, upper = cfg.dof_pos_limits.unbind(1)
    asset = Go2PayloadCfg.asset
    limits = parse_urdf_position_limits(asset.file.format(GYM_ROOT_DIR=GYM_ROOT_DIR))
    expected = torch.tensor([limits[name] for name in asset.robot_layout.dof_names])
    torch.testing.assert_close(cfg.dof_pos_limits, expected)
    assert ((lower < cfg.default_dof_pos) & (cfg.default_dof_pos < upper)).all()


def test_intermediate_ramp_starts_at_measured_pose_and_ends_at_nominal():
    controller, message = _controller(), _message()
    # Unitree stand pose, in SDK motor order (FR, FL, RR, RL).
    for motor, position in zip(message.motor_state, [0.0, 0.67, -1.30] * 4):
        motor.q = position
    start = deploy_utility.get_dof_pos(message)
    nominal = controller.cfg.default_dof_pos

    def commanded(elapsed):
        target = deploy_utility.ramp_target(start, nominal, elapsed, 2.0)
        action = deploy_utility.target_pos_to_action(controller, target)
        return deploy_utility.action_to_target_pos(controller, action)

    # The first target is where the joints are, not an offset from nominal.
    torch.testing.assert_close(commanded(0.0), torch.tensor([0.0, 0.67, -1.30] * 4))
    torch.testing.assert_close(commanded(1.0), (start + nominal) / 2)
    torch.testing.assert_close(commanded(2.0), nominal)
    torch.testing.assert_close(commanded(5.0), nominal)
