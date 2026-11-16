import torch

from gym.envs.go2.go2 import Go2
from gym.utils import masked_update
from gym.utils.torch_quat import quat_rotate_inverse


class Go2SE(Go2):
    """Go2 with the states needed for a concurrently trained state estimator
    (Ji et al., "Concurrent Training of a Control Policy and a State Estimator
    for Dynamic and Robust Legged Locomotion", RA-L 2022).

    Estimator targets (sim-only ground truth):
        base_lin_vel  -- already provided by LeggedRobot
        foot_height   -- height of each foot above the ground [m]
        foot_contact  -- 1.0 if the foot is in contact, else 0.0

    Extra proprioceptive observations (available on hardware):
        dof_pos_error_history -- (q_des - q) at t - k * history_dt, k = 1..N
        dof_vel_history       -- dq at t - k * history_dt, k = 1..N
        foot_pos_base         -- foot positions in the base frame (kinematics)
    """

    def _init_buffers(self):
        super()._init_buffers()
        n = self.num_actuators
        self.num_feet = len(self.feet_indices)
        self.foot_height = torch.zeros(self.num_envs, self.num_feet, device=self.device)
        self.foot_contact = torch.zeros(
            self.num_envs, self.num_feet, device=self.device
        )
        self.foot_pos_base = torch.zeros(
            self.num_envs, 3 * self.num_feet, device=self.device
        )
        # * ring buffers of every control step; histories are sampled from
        # * them every history_stride steps (paper: t-0.02, t-0.04, t-0.06 s)
        num_samples = self.cfg.env.history_length
        self.history_stride = max(1, round(self.cfg.env.history_dt / self.dt))
        self._history_idx = self.history_stride * torch.arange(
            1, num_samples + 1, device=self.device
        )
        buffer_length = self.history_stride * num_samples + 1
        self._dof_pos_error_buffer = torch.zeros(
            self.num_envs, buffer_length, n, device=self.device
        )
        self._dof_vel_buffer = torch.zeros_like(self._dof_pos_error_buffer)
        self.dof_pos_error_history = torch.zeros(
            self.num_envs, num_samples * n, device=self.device
        )
        self.dof_vel_history = torch.zeros_like(self.dof_pos_error_history)
        # * reward bookkeeping (Ji et al. Table II)
        self.foot_vel_xy = torch.zeros(self.num_envs, self.num_feet, device=self.device)
        self.feet_air_time = torch.zeros_like(self.foot_height)  # T_a
        self.feet_stance_time = torch.zeros_like(self.foot_height)  # T_s
        self._last_dof_vel = torch.zeros(self.num_envs, n, device=self.device)
        self.dof_vel_delta = torch.zeros_like(self._last_dof_vel)
        self._update_foot_states()

    def _sample_histories(self):
        self.dof_pos_error_history[:] = self._dof_pos_error_buffer[
            :, self._history_idx
        ].flatten(1)
        self.dof_vel_history[:] = self._dof_vel_buffer[:, self._history_idx].flatten(1)

    def _actuated_dof_pos_error(self):
        pos = self.dof_pos.index_select(1, self.actuated_dof_indices)
        default_pos = self.default_dof_pos.index_select(1, self.actuated_dof_indices)
        return self.dof_pos_target + default_pos - pos

    def _update_foot_states(self):
        foot_pos = self._rigid_body_pos[:, self.feet_indices, :]
        # * flat terrain only: ground is at z=0
        self.foot_height[:] = (
            foot_pos[..., 2] - self.cfg.asset.foot_radius
        ).clip(min=0.0)
        self.foot_contact[:] = (
            torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
            > self.cfg.reward_settings.contact_force_threshold
        ).float()
        rel_pos = foot_pos - self.root_states[:, None, 0:3]
        quat = self.base_quat.repeat_interleave(self.num_feet, dim=0)
        self.foot_pos_base[:] = quat_rotate_inverse(
            quat, rel_pos.reshape(-1, 3)
        ).view(self.num_envs, -1)
        if hasattr(self, "_rigid_body_lin_vel"):
            self.foot_vel_xy[:] = torch.norm(
                self._rigid_body_lin_vel[:, self.feet_indices, :2], dim=-1
            )

    def _update_gait_timers(self):
        contact = self.foot_contact.bool()
        zero = torch.zeros_like(self.feet_air_time)
        self.feet_air_time[:] = torch.where(contact, zero, self.feet_air_time + self.dt)
        self.feet_stance_time[:] = torch.where(
            contact, self.feet_stance_time + self.dt, zero
        )

    def _post_decimation_step(self):
        super()._post_decimation_step()
        vel = self.dof_vel.index_select(1, self.actuated_dof_indices)
        self.dof_vel_delta[:] = vel - self._last_dof_vel
        self._last_dof_vel[:] = vel
        self._dof_pos_error_buffer = self._dof_pos_error_buffer.roll(1, dims=1)
        self._dof_pos_error_buffer[:, 0] = self._actuated_dof_pos_error()
        self._dof_vel_buffer = self._dof_vel_buffer.roll(1, dims=1)
        self._dof_vel_buffer[:, 0] = self.dof_vel.index_select(
            1, self.actuated_dof_indices
        )
        self._sample_histories()
        self._update_foot_states()
        self._update_gait_timers()

    # * ------------------------------------------------------------------
    # * commands: curriculum on forward speed + standing-still commands
    # * ------------------------------------------------------------------

    def update_command_curriculum(self, iteration):
        """Ji et al. eq. (1): V_x,max = v0 + gain / (1 + exp(-rate (t - mid))).
        The backward limit scales with it (-0.5 -> -1.75 when 1.0 -> 3.5)."""
        c = self.cfg.commands.curriculum
        if not c.toggle:
            return
        x = torch.tensor(-c.rate * (iteration - c.midpoint))
        v_max = c.v_start + c.v_gain / (1.0 + torch.exp(x).item())
        self.command_ranges["lin_vel_x"] = [-c.backward_ratio * v_max, v_max]

    def _resample_commands(self, command_mask):
        super()._resample_commands(command_mask)
        # * a fraction of commands is exactly zero, to learn standing still
        stand = command_mask & (
            torch.rand(self.num_envs, device=self.device)
            < self.cfg.commands.zero_command_prob
        )
        self.commands.masked_fill_(stand.unsqueeze(1), 0.0)

    def _reset_idx(self, reset_mask):
        # * with some probability, a timed-out env keeps its final state and
        # * only gets a new command (learns to handle sudden command changes)
        keep = (
            reset_mask
            & self.timed_out
            & ~self.terminated
            & (
                torch.rand(self.num_envs, device=self.device)
                < self.cfg.init_state.continue_prob
            )
        )
        if keep.any():
            self.episode_length_buf.masked_fill_(keep, 0)
            self._resample_commands(keep)
        reset_mask = reset_mask & ~keep
        if not reset_mask.any():
            return
        super()._reset_idx(reset_mask)
        # * targets are offsets from default_dof_pos (see _compute_torques), so
        # * the base class reset to default_dof_pos would double the posture
        self.dof_pos_target.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        masked_update(
            self.dof_pos_history,
            self.dof_pos_target.tile(1, 3),
            reset_mask,
        )
        mask = reset_mask.view(-1, 1, 1)
        self._dof_pos_error_buffer[:] = torch.where(
            mask, self._actuated_dof_pos_error().unsqueeze(1), self._dof_pos_error_buffer
        )
        self._dof_vel_buffer[:] = torch.where(
            mask,
            self.dof_vel.index_select(1, self.actuated_dof_indices).unsqueeze(1),
            self._dof_vel_buffer,
        )
        self._sample_histories()
        self._update_foot_states()
        vel = self.dof_vel.index_select(1, self.actuated_dof_indices)
        masked_update(self._last_dof_vel, vel, reset_mask)
        self.dof_vel_delta.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        self.feet_air_time.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        self.feet_stance_time.masked_fill_(reset_mask.unsqueeze(1), 0.0)

    # * ------------------------------------------------------------------
    # * Ji et al. Table II reward terms. These return raw (unweighted)
    # * quantities; the paper's coefficients (negative for penalties) are
    # * the weights in Go2SERunnerCfg, and EstimatorRunner combines them as
    # * r_pos * exp(0.2 * r_neg).
    # * ------------------------------------------------------------------

    def _reward_lin_vel_exp(self):
        # * paper: sigma = 1. Smaller sigma makes standing still under a
        # * moving command pay much less (at |cmd|=1: 37% -> 2% for 0.25)
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        sigma = self.cfg.reward_settings.lin_vel_tracking_sigma
        return torch.exp(-error.square().sum(dim=1) / sigma)

    def _reward_ang_vel_exp(self):
        return torch.exp(-1.5 * (self.commands[:, 2] - self.base_ang_vel[:, 2]).square())

    def _reward_air_time(self):
        t_a, t_s = self.feet_air_time, self.feet_stance_time
        stand = torch.clip(t_s - t_a, -0.3, 0.3)
        t_max = torch.maximum(t_a, t_s)
        move = torch.where(t_max < 0.25, t_max.clamp(max=0.2), torch.zeros_like(t_max))
        stance_cmd = (torch.norm(self.commands, dim=1) < 1e-6).unsqueeze(1)
        return torch.where(stance_cmd, stand, move).sum(dim=1)

    def _reward_foot_slip(self):
        return (self.foot_contact * self.foot_vel_xy.square()).sum(dim=1)

    def _reward_foot_clearance(self):
        target = self.cfg.reward_settings.foot_clearance_target
        error = (self.foot_height - target).square()
        return (error * self.foot_vel_xy.sqrt()).sum(dim=1)

    def _reward_orientation_angle(self):
        cos = (-self.projected_gravity[:, 2]).clamp(-1.0, 1.0)
        return torch.acos(cos).square()

    def _reward_joint_torque(self):
        return self.torques.square().sum(dim=1)

    def _reward_joint_pos(self):
        return (self.dof_pos - self.default_dof_pos).square().sum(dim=1)

    def _reward_joint_speed(self):
        return self.dof_vel.square().sum(dim=1)

    def _reward_joint_acc(self):
        return self.dof_vel_delta.square().sum(dim=1)

    def _reward_action_smoothness1(self):
        n = self.num_actuators
        h = self.dof_pos_history
        return (h[:, :n] - h[:, n : 2 * n]).square().sum(dim=1)

    def _reward_action_smoothness2(self):
        n = self.num_actuators
        h = self.dof_pos_history
        return (h[:, :n] - 2 * h[:, n : 2 * n] + h[:, 2 * n :]).square().sum(dim=1)

    def _reward_base_motion(self):
        return (
            0.8 * self.base_lin_vel[:, 2].square()
            + 0.2 * self.base_ang_vel[:, 0].abs()
            + 0.2 * self.base_ang_vel[:, 1].abs()
        )
