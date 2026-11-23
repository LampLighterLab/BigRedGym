from types import SimpleNamespace
import torch

from go2_deploy.custom_implementation.phase1.go2osc_deploy import (
    get_custom_buffers_dict,
    advance_oscillators,
)

from go2_deploy.custom_implementation.phase1.go2osc_deploy import (
    normalize_foot_forces,
)


def test_oscillator_update():
    controller = SimpleNamespace(**get_custom_buffers_dict())
    initial_phase = controller.phase.clone()

    grf = torch.tensor([0.2, 0.3, 0.4, 0.5])

    advance_oscillators(controller, grf, dt=0.005)

    assert controller.phase.shape == (4,)
    assert controller.phase_obs.shape == (8,)
    assert controller.phase_frequency.shape == (4,)
    assert controller.target_grf.shape == (4,)
    assert torch.isfinite(controller.phase).all()
    assert not torch.allclose(initial_phase, controller.phase)

    def test_force_normalization():
        grf = normalize_foot_forces(
            raw=[20, 40, 60, 80],
            offsets=[10, 20, 30, 40],
            gains=[1, 1, 1, 1],
            order=[1, 0, 3, 2],
            mass_kg=10.0,
        )

        expected = torch.tensor([20, 10, 40, 30]) / 98.1

        assert torch.allclose(grf, expected)