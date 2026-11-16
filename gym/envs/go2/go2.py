import torch

from gym.envs.base.legged_robot import LeggedRobot
# from learning.utils.logger.SaveStates import (
#     init_env_log_buffers,
# )


class Go2(LeggedRobot):
    def __init__(self, cfg, device, headless, backend):
        super().__init__(cfg, device, headless, backend)

    def dof_pos_target_bounds(self):
        """Per-env, per-joint [lower, upper] for dof_pos_target (rad offsets).

        The narrowest interval that still lets the PD law reach the torque
        limit in either direction from anywhere inside the joint range:
        target + default must reach (joint limit +- tau_max / Kp). Wider
        targets only saturate the torque harder, so clipping to this interval
        changes no reachable torque; it only stops unbounded policy outputs
        from re-entering the observation as huge dof_pos_target values.
        """
        idx = self.actuated_dof_indices
        default = self.default_dof_pos.index_select(1, idx)
        limits = self.dof_pos_limits.index_select(0, idx)
        margin = self.actuated_torque_limits / self.p_gains.clamp_min(1e-6)
        lower = limits[:, 0] - default - margin
        upper = limits[:, 1] - default + margin
        return lower, upper

    def _pre_decimation_step(self):
        super()._pre_decimation_step()
        if getattr(self.cfg.control, "clip_dof_pos_target", False):
            lower, upper = self.dof_pos_target_bounds()
            torch.clamp(self.dof_pos_target, lower, upper, out=self.dof_pos_target)
        if self._latency_range() is not None:
            self._push_latency_buffer()

    # * Action latency ---------------------------------------------------------
    # The policy observes the target it commanded (dof_pos_target), as on the
    # robot, but the PD law acts on the target from `delay` control steps
    # earlier. `delay` is sampled per env at reset, uniformly from
    # control.action_latency_steps_range = [min, max] (inclusive).

    def _latency_range(self):
        return getattr(self.cfg.control, "action_latency_steps_range", None)

    def _init_latency(self):
        low, high = self._latency_range()
        self._latency_steps = torch.randint(
            low, high + 1, (self.num_envs,), device=self.device
        )
        self._target_history = torch.zeros(
            high + 1, *self.dof_pos_target.shape, device=self.device
        )

    def _push_latency_buffer(self):
        if not hasattr(self, "_target_history"):
            self._init_latency()
        self._target_history = torch.roll(self._target_history, 1, dims=0)
        self._target_history[0] = self.dof_pos_target
        env = torch.arange(self.num_envs, device=self.device)
        self._applied_target = self._target_history[self._latency_steps, env]

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        if self._latency_range() is None:
            return
        if not hasattr(self, "_target_history"):
            self._init_latency()
        low, high = self._latency_range()
        n = int(reset_mask.sum())
        if n:
            self._latency_steps[reset_mask] = torch.randint(
                low, high + 1, (n,), device=self.device
            )
            # Until new commands arrive, the delayed target holds the default
            # posture (zero offset).
            self._target_history[:, reset_mask] = 0.0

    def _compute_torques(self):
        if self._latency_range() is None or not hasattr(self, "_applied_target"):
            return super()._compute_torques()
        commanded = self.dof_pos_target
        self.dof_pos_target = self._applied_target
        try:
            return super()._compute_torques()
        finally:
            self.dof_pos_target = commanded

    def _reward_lin_vel_z(self):
        """Penalize z axis base linear velocity with squared exp"""
        return self._sqrdexp(self.base_lin_vel[:, 2] / self.scales["base_lin_vel"])

    def _reward_ang_vel_xy(self):
        """Penalize xy axes base angular velocity"""
        error = self._sqrdexp(self.base_ang_vel[:, :2] / self.scales["base_ang_vel"])
        return torch.sum(error, dim=1)

    def _reward_orientation(self):
        """Penalize non-flat base orientation"""
        error = (
            torch.square(self.projected_gravity[:, :2])
            / self.cfg.reward_settings.tracking_sigma
        )
        return torch.sum(torch.exp(-error), dim=1)

    def _reward_min_base_height(self):
        """Squared exponential saturating at base_height target"""
        error = self.base_height - self.cfg.reward_settings.base_height_target
        error /= self.scales["base_height"]
        error = torch.clamp(error, max=0, min=None).flatten()
        return self._sqrdexp(error)

    def _reward_tracking_lin_vel(self):
        """Tracking of linear velocity commands (xy axes)"""
        # just use lin_vel?
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        # * scale by (1+|cmd|): if cmd=0, no scaling.
        error *= 1.0 / (1.0 + torch.abs(self.commands[:, :2]))
        error = torch.sum(torch.square(error), dim=1)
        return torch.exp(-error / self.cfg.reward_settings.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """Tracking of angular velocity commands (yaw)"""
        ang_vel_error = torch.square(
            (self.commands[:, 2] - self.base_ang_vel[:, 2]) / 2.5
        )
        return self._sqrdexp(ang_vel_error)

    def _reward_dof_vel(self):
        """Penalize dof velocities"""
        return torch.sum(self._sqrdexp(self.dof_vel / self.scales["dof_vel"]), dim=1)

    def _reward_dof_near_home(self):
        return torch.sum(
            self._sqrdexp(
                (self.dof_pos - self.default_dof_pos) / self.scales["dof_pos_obs"]
            ),
            dim=1,
        )
