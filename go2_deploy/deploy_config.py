import gym.envs  # noqa: F401
from gym import GYM_ROOT_DIR
from gym.envs.base.urdf_limits import parse_urdf_position_limits
from gym.utils.task_registry import task_registry
import torch

# Scaling, nominal posture and actor inputs all come from this task's config.
TASK_NAME = "go2payload"
# Registered configs are instances; DeployScaling subclasses the class itself.
_ENV_CFG = type(task_registry.env_cfgs[TASK_NAME])
_TRAIN_CFG = type(task_registry.train_cfgs[TASK_NAME])
_URDF_POS_LIMITS = parse_urdf_position_limits(
    _ENV_CFG.asset.file.format(GYM_ROOT_DIR=GYM_ROOT_DIR)
)


def _single_gain(gains):
    """Unitree takes one kp/kd here, so training must use one value per gain."""
    values = set(gains.values())
    if len(values) != 1:
        raise ValueError(f"deploy expects one gain for every joint, got {gains}")
    return values.pop()


class DeployConfig:
    task_name = TASK_NAME

    # Control rate and PD gains must match training, so take them from the task.
    ctrl_freq = _ENV_CFG.control.ctrl_frequency  # Hz
    kp = _single_gain(_ENV_CFG.control.stiffness)  # Stiffness constant
    kd = _single_gain(_ENV_CFG.control.damping)  # Damping constant

    phase_frequency = 2.0  # Go2Trot gait frequency in Hz (cycles/s)

    # Apply exponential moving average to actions. Training applies none, and
    # the actor observes the smoothed action, so leave this off.
    exp_moving_avg = False
    ema_smoothing_factor = 0.1

    # Must match the trained actor exactly, so take it from the task config.
    obs_vector = list(_TRAIN_CFG.actor.obs)

    # Scale observations
    class DeployScaling(_ENV_CFG.scaling):
        pass

    # Full joystick deflection commands +/- these values. __init__ rejects any
    # limit outside the training command ranges.
    command_limits = {
        "lin_vel_x": 1.0,  # m/s; training covers -1.0 to 2.5
        "lin_vel_y": 0.5,  # m/s
        "yaw_vel": 1.5,  # rad/s
    }

    # Specify size of observation vector. should not have to modify this
    obs_sizes = {
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "commands": 3,
        "dof_pos_obs": 12,
        "dof_vel": 12,
        "dof_accel": 12,
        "dof_pos_target": 12,
        "phase_obs": 2,
        "phase_frequency": 1,
    }

    # Canonical FL, FR, RL, RR order; hip, thigh, calf within each leg.
    default_dof_pos = torch.tensor(
        [
            angle
            for leg in ("FL", "FR", "RL", "RR")
            for angle in (
                _ENV_CFG.init_state.default_joint_angles["hip_joint"],
                _ENV_CFG.init_state.default_joint_angles[f"{leg}_thigh_joint"],
                _ENV_CFG.init_state.default_joint_angles["calf_joint"],
            )
        ]
    )

    # URDF [lower, upper] joint limits (rad), same order. Position targets are
    # clamped to these so the motors never drive into the hard stops.
    dof_pos_limits = torch.tensor(
        [_URDF_POS_LIMITS[name] for name in _ENV_CFG.asset.robot_layout.dof_names]
    )

    # s: INTERMEDIATE ramps linearly from the measured pose to default_dof_pos
    # over this time; CUSTOM_CTRL can only start once the ramp has finished.
    intermediate_ramp_time = 2.0

    # s: how often terminal should print ctrl freq, instructions
    terminal_log_period = 2.0

    # Check for unsafe config
    def __init__(self):
        if self.ctrl_freq < 20:
            raise ValueError("ctrl_freq should be > 20")
        if self.kp <= 0:
            raise ValueError("kp should be > 0")
        if self.kd <= 0:
            raise ValueError("kd should be > 0")
        if self.intermediate_ramp_time <= 0:
            raise ValueError("intermediate_ramp_time should be > 0")
        ranges = _ENV_CFG.commands.ranges
        # The joystick is symmetric, so x is bounded by the slower direction.
        trained = {
            "lin_vel_x": min(-min(ranges.lin_vel_x), max(ranges.lin_vel_x)),
            "lin_vel_y": ranges.lin_vel_y,
            "yaw_vel": ranges.yaw_vel,
        }
        for name, limit in self.command_limits.items():
            if not 0 <= limit <= trained[name]:
                raise ValueError(
                    f"command_limits[{name!r}] = {limit} is outside the "
                    f"trained range [0, {trained[name]}]"
                )
        if hasattr(_ENV_CFG.control, "gait_freq"):
            low, high = _ENV_CFG.control.gait_freq
            if not low <= self.phase_frequency <= high:
                raise ValueError(
                    f"phase_frequency {self.phase_frequency} Hz is outside the "
                    f"trained gait_freq [{low}, {high}]"
                )
        for obs in self.obs_vector:
            if obs not in self.obs_sizes:
                raise KeyError("observation " + obs + " not supported")
