import torch

from gym.envs.go2.go2 import Go2
from gym.utils.torch_quat import quat_apply


class Go2Stand(Go2):
    """Go2 rearing up onto its two hind legs."""

    def _init_buffers(self):
        super()._init_buffers()
        rs = self.cfg.reward_settings

        target = torch.tensor(rs.gravity_target, dtype=torch.float, device=self.device)
        self._gravity_target = target / target.norm()

        self._hind_feet_indices = self._body_indices(rs.hind_feet_names)
        self._front_feet_indices = self._body_indices(rs.front_feet_names)

        # Body positions are link-frame origins; add each link's CoM offset.
        mjm = getattr(self._backend, "_mjm", None)
        order = getattr(self._backend, "_canonical_to_native_body_np", None)
        if mjm is not None and order is not None:
            offsets = torch.tensor(
                mjm.body_ipos[order], dtype=torch.float, device=self.device
            )
        else:
            offsets = torch.zeros(self.num_bodies, 3, device=self.device)
        self._link_com_offsets = offsets.unsqueeze(0).expand(self.num_envs, -1, -1)
        self.last_base_ang_vel = torch.zeros_like(self.base_ang_vel)

    def _pre_decimation_step(self):
        super()._pre_decimation_step()
        self.last_base_ang_vel[:] = self.base_ang_vel

    def _check_terminations_and_timeouts(self):
        super()._check_terminations_and_timeouts()
        # Pitch: 0 = level, 90 = nose straight up, >90 = tipping onto its back.
        g = self.projected_gravity
        pitch = torch.rad2deg(torch.atan2(-g[:, 0], -g[:, 2]))
        self.terminated |= pitch > self.cfg.reward_settings.max_pitch_deg

    def _body_indices(self, names):
        return torch.tensor(
            [self._backend.find_body_index(name) for name in names],
            dtype=torch.long,
            device=self.device,
        )

    def _whole_body_com(self):
        link_com = self._rigid_body_pos + quat_apply(
            self._rigid_body_quat, self._link_com_offsets
        )
        mass = self._backend.link_mass.unsqueeze(-1)
        return (mass * link_com).sum(dim=1) / mass.sum(dim=1)

    def _reward_gravity_target(self):
        error = torch.sum(
            torch.square(self.projected_gravity - self._gravity_target), dim=1
        )
        return torch.exp(-error / self.cfg.reward_settings.gravity_sigma)

    def _reward_com_over_hind_feet(self):
        """CoM ground projection on the segment between the hind feet."""
        com_xy = self._whole_body_com()[:, :2]
        feet_xy = self._rigid_body_pos[:, self._hind_feet_indices, :2]
        a, b = feet_xy[:, 0], feet_xy[:, 1]
        ab = b - a
        t = torch.sum((com_xy - a) * ab, dim=1) / torch.sum(ab * ab, dim=1).clamp(
            min=1e-6
        )
        closest = a + t.clamp(0.0, 1.0).unsqueeze(1) * ab
        distance_sq = torch.sum(torch.square(com_xy - closest), dim=1)
        return torch.exp(-distance_sq / self.cfg.reward_settings.com_line_sigma**2)

    def _reward_base_ang_acc(self):
        """Penalize base angular acceleration (body frame)."""
        acc = (self.base_ang_vel - self.last_base_ang_vel) / self.dt
        return -torch.sum(
            torch.square(acc / self.cfg.reward_settings.ang_acc_scale), dim=1
        )

    def _reward_stand_height(self):
        """Saturates at 1 once the base reaches stand_height_target."""
        rs = self.cfg.reward_settings
        shortfall = torch.clamp(
            self.base_height.flatten() - rs.stand_height_target, max=0.0
        )
        return torch.exp(-torch.square(shortfall / rs.stand_height_sigma))

    def _reward_front_feet_height(self):
        """Saturates at 1 once the front feet reach the target height."""
        rs = self.cfg.reward_settings
        height = self._rigid_body_pos[:, self._front_feet_indices, 2]
        shortfall = torch.clamp(height - rs.front_feet_height_target, max=0.0)
        return torch.exp(-torch.square(shortfall / rs.front_feet_height_sigma)).mean(
            dim=1
        )
