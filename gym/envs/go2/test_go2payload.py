from types import SimpleNamespace

import torch

from gym.envs.go2.go2payload import Go2Payload


def test_contact_strength_counts_payload_as_body_weight():
    task = Go2Payload.__new__(Go2Payload)
    task.feet_indices = torch.arange(4)
    task.contact_forces = torch.zeros(2, 4, 3)
    task.contact_forces[:, :, 2] = 98.1
    # Same 20 kg total load either way, whether robot or payload.
    task._backend = SimpleNamespace(
        link_mass=torch.tensor([[10.0, 10.0], [5.0, 10.0]]),
        payload_mass=torch.tensor([0.0, 5.0]),
    )

    strength = task._foot_contact_strength()

    torch.testing.assert_close(strength[0], strength[1])
    torch.testing.assert_close(strength[1], torch.full((4,), 0.5))
