import math

import torch
import numpy as np
from go2_deploy.custom_implementation.custom_config import CustomConfig


class DeployConfig:
    # custom_impl: CustomConfig instance, from load_custom_impl(task_name)
    def __init__(self, custom_impl):
        if not isinstance(custom_impl, CustomConfig):
            raise TypeError(
                f"custom_impl must be a CustomConfig instance, got {custom_impl!r}"
            )
        self.custom_impl = custom_impl

        self.task_name = custom_impl.task_name

        self.ctrl_freq = getattr(custom_impl, "trained_ctrl_freq", 100)
        self.kp = getattr(custom_impl, "trained_kp", 30.0)
        self.kd = getattr(custom_impl, "trained_kd", 2.0)

        # Apply exponential moving average to actions
        self.exp_moving_avg = getattr(custom_impl, "trained_exp_moving_avg", True)
        self.ema_time_const = 0.095  # seconds

        self.obs_vector = list(self.custom_impl.obs_vector)

        # Scale observations
        class DeployScaling(self.custom_impl.DeployScaling):
            pass

        self.DeployScaling = DeployScaling

        # Set ranges of joystick commands
        course_limits = {
            "lin_vel_x": 2.0,  # m/s
            "lin_vel_y": 1.0,  # m/s
            "yaw_vel": 3.0,  # rad/s
        }
        self.command_limits = dict(
            getattr(custom_impl, "command_limits", course_limits)
        )
        self.intermediate_ramp_time = getattr(
            custom_impl, "intermediate_ramp_time", None
        )
        limits = getattr(custom_impl, "dof_pos_limits", None)
        self.dof_pos_limits = torch.tensor(limits) if limits is not None else None

        # Specify size of observation vector pieces
        self.obs_sizes = {
            "base_ang_vel": 3,
            "projected_gravity": 3,
            "commands": 3,
            "dof_pos_obs": 12,
            "dof_vel": 12,
            "dof_accel": 12,
            "dof_pos_target": 12,
        }

        self.obs_sizes.update(self.custom_impl.custom_obs_sizes)

        # Static default posture the actions are offsets from (canonical
        # FL, FR, RL, RR order; hip, thigh, calf within each leg)
        self.default_dof_pos = torch.tensor(
            self.custom_impl.default_dof_pos, dtype=torch.float
        )
        # Training run the policy came from; set by setup_actor()
        self.run_dir = None

        # s: how often terminal should print ctrl freq, instructions
        self.terminal_log_period = 2.0

        # Set by policy_bundle.load_bundle(): the bundle's deploy_meta.json
        self.bundle_meta = None

        self.validate()

    def obs_dim(self):
        return sum(self.obs_sizes[obs] for obs in self.obs_vector)

    @property
    def ema_smoothing_factor(self):
        return -np.expm1(-1 / (self.ctrl_freq * self.ema_time_const))

    # Check for unsafe config
    def validate(self):
        if not math.isfinite(self.ctrl_freq) or self.ctrl_freq < 20:
            raise ValueError("ctrl_freq must be finite and at least 20 Hz")
        for name in ("kp", "kd", "ema_time_const"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        for name in ("ctrl_freq", "kp", "kd", "exp_moving_avg"):
            trained = getattr(self.custom_impl, f"trained_{name}", None)
            if trained is not None and getattr(self, name) != trained:
                raise ValueError(f"{name} must match the trained value {trained}")
        bounds = getattr(self.custom_impl, "command_limits", self.command_limits)
        for name, maximum in bounds.items():
            if not 0 <= self.command_limits[name] <= maximum:
                raise ValueError(f"{name} command limit must be within [0, {maximum}]")
        gait_range = getattr(self.custom_impl, "trained_gait_freq_range", None)
        if gait_range is not None:
            low, high = gait_range
            hz = self.custom_impl.get_custom_buffers_dict()["phase_frequency"]
            if not low <= hz <= high:
                raise ValueError(
                    f"phase_frequency is outside trained gait_freq {gait_range}"
                )
        ramp_time = self.intermediate_ramp_time
        if ramp_time is not None and (not math.isfinite(ramp_time) or ramp_time <= 0):
            raise ValueError("intermediate_ramp_time must be finite and > 0")
        for obs in self.obs_vector:
            if obs not in self.obs_sizes:
                raise KeyError("observation " + obs + " not supported")
        if self.default_dof_pos.shape != (12,):
            raise ValueError(
                f"default_dof_pos should have 12 joint angles, "
                f"got shape {tuple(self.default_dof_pos.shape)}"
            )
        if self.dof_pos_limits is not None:
            if self.dof_pos_limits.shape != (12, 2):
                raise ValueError("dof_pos_limits must contain 12 [lower, upper] pairs")
            lower, upper = self.dof_pos_limits.unbind(1)
            if not (
                (lower < self.default_dof_pos) & (self.default_dof_pos < upper)
            ).all():
                raise ValueError("default_dof_pos must be within dof_pos_limits")
