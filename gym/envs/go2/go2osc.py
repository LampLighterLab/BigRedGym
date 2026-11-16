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
            self.phase + 2 * math.pi * self.cfg.osc.frequency * dt,
            2 * math.pi,
        )

        self._update_oscillator_observations()

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)

        cycle_phase = 2 * math.pi * torch.rand(
            self.num_envs, 1, device=self.device
        )
        phases = torch.remainder(cycle_phase + self._phase_offsets, 2 * math.pi)
        masked_update(self.phase, phases, reset_mask)

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
        return torch.where(stance, 2 * contact - 1, -contact).mean(dim=1)

    def _reward_orientation(self):
        # The parent reward uses only gravity's x/y components, which also
        # score an upside-down but level robot highly.
        return -2.0 * self.projected_gravity[:, 2]
