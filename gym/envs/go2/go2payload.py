import torch

from gym.envs.base.domain_randomization import payload_mass_range
from gym.envs.go2.go2trot import Go2Trot


class Go2Payload(Go2Trot):
    """Go2Trot carrying a welded payload resampled every episode."""

    def _init_buffers(self):
        super()._init_buffers()
        # Privileged critic observation: [mass, COM offset from mount_pos].
        # Stays zero when the payload is disabled.
        self.payload_obs = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device
        )
        self._payload_enabled = payload_mass_range(self.cfg) is not None
        # Configs saved before the payload existed have no payload block.
        payload_cfg = getattr(self.cfg.domain_randomization, "payload", None)
        self._payload_mount_pos = torch.tensor(
            payload_cfg.mount_pos if payload_cfg else [0.0, 0.0, 0.0],
            dtype=torch.float,
            device=self.device,
        )

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        self._update_payload_observation()

    def _update_payload_observation(self):
        """Payloads only change at reset, so refresh the observation there."""
        if not self._payload_enabled:
            return
        self.payload_obs[:, 0] = self._backend.payload_mass
        self.payload_obs[:, 1:] = self._backend.payload_com - self._payload_mount_pos

    def _foot_contact_strength(self):
        """Map upward foot load smoothly from zero to robot-plus-payload weight."""
        total_mass = self._backend.link_mass.sum(dim=1, keepdim=True)
        total_mass = total_mass + self._backend.payload_mass.unsqueeze(1)
        load = torch.clamp(
            self.contact_forces[:, self.feet_indices, 2] / (9.81 * total_mass),
            min=0.0,
            max=1.0,
        )
        return load.square() * (3.0 - 2.0 * load)
