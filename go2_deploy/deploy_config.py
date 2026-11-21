import gym.envs  # noqa: F401
from gym.utils.task_registry import task_registry
import torch

# Scaling, nominal posture and actor inputs all come from this task's config.
TASK_NAME = "go2payload"
# Registered configs are instances; DeployScaling subclasses the class itself.
_ENV_CFG = type(task_registry.env_cfgs[TASK_NAME])
_TRAIN_CFG = type(task_registry.train_cfgs[TASK_NAME])


class DeployConfig:
    task_name = TASK_NAME

    ctrl_freq = 100  # Hz

    kp = 30.0  # Stiffness constant
    kd = 2.0  # Damping constant

    phase_frequency = 2.0  # Go2Trot gait frequency in Hz (cycles/s)

    # Apply exponential moving average to actions
    exp_moving_avg = True
    ema_smoothing_factor = 0.1

    # Must match the trained actor exactly, so take it from the task config.
    obs_vector = list(_TRAIN_CFG.actor.obs)

    # Scale observations
    class DeployScaling(_ENV_CFG.scaling):
        pass

    # Set ranges of joystick commands
    command_limits = {
        "lin_vel_x": 2.0,  # m/s
        "lin_vel_y": 1.0,  # m/s
        "yaw_vel": 3.0,  # rad/s
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
        for obs in self.obs_vector:
            if obs not in self.obs_sizes:
                raise KeyError("observation " + obs + " not supported")
