import torch
import math

from gym.envs.go2.go2 import Go2
from gym.envs.go2.go2_oscillators import phase_rate, step, desired_force
from gym.utils.sampling import masked_update

class Go2Osc(Go2):

    def _init_buffers(self):
        super()._init_buffers()

        self.phase = torch.zeros(self.num_envs, 4, device=self.device)
        self.phase_obs = torch.zeros(self.num_envs, 8, device=self.device)

        self.phase_frequency = torch.zeros_like(self.phase)
        self.measured_grf = torch.zeros_like(self.phase)
        self.target_grf = torch.zeros_like(self.phase)

    def _vertical_grndforce(self):

        weight = 9.81 * self._backend.link_mass.sum(
            dim=1, keepdim=True
        )

        forces = self.contact_forces[:, self.feet_indices, 2]

        return forces.clamp_min(0.0) / weight

    def _update_oscillator_observations(self):
            
        self.measured_grf = self._vertical_grndforce()

        self.target_grf = desired_force(self.phase, self.cfg.osc.amplitude)

        self.phase_obs = torch.cat((torch.sin(self.phase), torch.cos(self.phase)),dim=1)

        self.phase_frequency = phase_rate(self.phase,self.measured_grf.clamp(max=1.0),self.cfg.osc.omega,self.cfg.osc.sigma,self.cfg.osc.xi) / (2 * math.pi)

    def _post_physics_step(self):
        super()._post_physics_step()

        grf = self._vertical_grndforce().clamp(max=1.0)

        dt = self.dt / self.cfg.control.decimation

        self.phase = step(
            self.phase,
            grf,
            self.cfg.osc.omega,
            self.cfg.osc.sigma,
            self.cfg.osc.xi,
            dt
        )

        self._update_oscillator_observations()

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)

        random_phases = 2 * math.pi * torch.rand_like(self.phase)

        masked_update(self.phase, random_phases, reset_mask)

        self.dof_pos_target.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        self.dof_pos_history.masked_fill_(reset_mask.unsqueeze(1), 0.0)

        self._update_oscillator_observations()

    def _reward_force_tracking(self):
        error = self.measured_grf - self.target_grf
        return -torch.mean(error**2, dim=1)