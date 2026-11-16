import math

import torch

from gym.envs.go2.go2 import Go2
from gym.utils.sampling import masked_update


class Go2Osc(Go2):

    def _init_buffers(self):
        super()._init_buffers()

        foot_names = self.robot_layout.body_groups["feet"]
        self._phase_offsets = 2 * math.pi * torch.tensor(
            [self.cfg.osc.phase_offsets[name] for name in foot_names],
            dtype=torch.float,
            device=self.device,
        )
        body_names = self._backend.body_names
        self.knee_indices = torch.tensor(
            [body_names.index(name.replace("_foot", "_calf")) for name in foot_names],
            dtype=torch.long,
            device=self.device,
        )
        self.phase = torch.zeros(self.num_envs, 4, device=self.device)
        self.phase_obs = torch.zeros(self.num_envs, 8, device=self.device)

        self.phase_frequency = torch.full_like(self.phase, self.cfg.osc.frequency)
        self.measured_grf = torch.zeros_like(self.phase)

    def _vertical_grndforce(self):
        weight = 9.81 * self._backend.link_mass.sum(
            dim=1, keepdim=True
        )

        forces = self.contact_forces[:, self.feet_indices, 2]

        return forces.clamp_min(0.0) / weight

    def _update_oscillator_observations(self):
        self.measured_grf = self._vertical_grndforce()
        self.phase_obs = torch.cat(
            (torch.sin(self.phase), torch.cos(self.phase)), dim=1
        )

    def _post_physics_step(self):
        super()._post_physics_step()

        dt = self.dt / self.cfg.control.decimation
        self.phase = torch.remainder(
            self.phase + 2 * math.pi * self.phase_frequency * dt,
            2 * math.pi,
        )

        self._update_oscillator_observations()

    def _post_decimation_step(self):
        super()._post_decimation_step()
        if self.cfg.osc.randomize_frequency:
            interval = round(self.cfg.osc.frequency_resampling_time / self.dt)
            resample_mask = (
                (self.episode_length_buf > 0)
                & (self.episode_length_buf < self.max_episode_length)
                & (self.episode_length_buf % interval == 0)
            )
            self._resample_frequency(resample_mask)

    def _check_terminations_and_timeouts(self):
        super()._check_terminations_and_timeouts()
        self.terminated |= self.root_states[:, 2] < self.cfg.env.termination_height

    def _resample_frequency(self, resample_mask):
        if self.cfg.osc.randomize_frequency:
            low, high = self.cfg.osc.frequency_range
            frequencies = low + (high - low) * torch.rand(
                self.num_envs, 1, device=self.device
            )
            if low == 0.0:
                zero_frequency = (
                    torch.rand(self.num_envs, 1, device=self.device)
                    < self.cfg.osc.zero_frequency_fraction
                )
                frequencies.masked_fill_(zero_frequency, 0.0)
        else:
            frequencies = torch.full(
                (self.num_envs, 1), self.cfg.osc.frequency, device=self.device
            )
        masked_update(
            self.phase_frequency, frequencies.expand(-1, 4), resample_mask
        )

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)

        cycle_phase = 2 * math.pi * torch.rand(
            self.num_envs, 1, device=self.device
        )
        phases = torch.remainder(cycle_phase + self._phase_offsets, 2 * math.pi)
        masked_update(self.phase, phases, reset_mask)
        self._resample_frequency(reset_mask)

        self.dof_pos_target.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        self.dof_pos_history.masked_fill_(reset_mask.unsqueeze(1), 0.0)

        self._update_oscillator_observations()

    def _reward_gait_contact(self):
        # Negative sine marks stance. Missing a stance contact is penalized,
        # so keeping every foot off the ground cannot cancel out the reward.
        contact = (self.measured_grf > self.cfg.osc.contact_threshold).to(
            self.phase.dtype
        )
        stance = torch.sin(self.phase) < 0
        gait_reward = torch.where(stance, 2 * contact - 1, -contact).mean(dim=1)
        # At 0 Hz every foot should support a stationary robot. Keep the
        # maximum at 0.5, matching the two-stance-foot moving gait.
        standing_reward = contact.mean(dim=1) - 0.5
        return torch.where(
            self.phase_frequency[:, 0] == 0.0, standing_reward, gait_reward
        )

    def _reward_orientation(self):
        # The parent reward uses only gravity's x/y components, which also
        # score an upside-down but level robot highly.
        return -2.0 * self.projected_gravity[:, 2]

    def _reward_knee_clearance(self):
        # The calf link origin is at the knee joint. Penalize approaching the
        # floor before the calf actually makes contact with it.
        clearance = self.cfg.osc.knee_clearance
        knee_height = self._rigid_body_pos[:, self.knee_indices, 2]
        shortfall = ((clearance - knee_height) / clearance).clamp_min(0.0)
        return -shortfall.square().mean(dim=1)
