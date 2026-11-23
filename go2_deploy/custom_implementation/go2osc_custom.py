"""Go2 oscillator observations and buffers for the saved go2osc policy."""

import math

import torch

from go2_deploy.custom_implementation.custom_config import CustomConfig


class Go2OscCustom(CustomConfig):
    task_name = "go2osc"

    # Actor observation order for the force-free go2osc policy (57 values).
    obs_vector = [
        "base_ang_vel",
        "projected_gravity",
        "commands",
        "dof_pos_obs",
        "dof_vel",
        "dof_pos_target",
        "phase_obs",
        "phase_frequency",
    ]

    custom_obs_sizes = {
        "phase_obs": 8,
        "phase_frequency": 4,
    }

    # FL, FR, RL, RR; hip, thigh, calf within each leg, in radians.
    default_dof_pos = [0.0, 0.66, -1.36] * 4

    class DeployScaling:
        base_ang_vel = 0.3
        commands = [3.0, 1.0, 3.0]
        dof_pos_obs = [1.0472, 2.53075, 0.94247] * 4
        dof_vel = [2.0, 2.0, 4.0] * 4
        dof_pos_target = [0.2, 0.3, 0.3] * 4
        # projected_gravity, phase_obs, and phase_frequency
        # have unit scale in training and in the deployment observation builder.

    def get_custom_buffers_dict(self):
        cycle_phase = 2 * math.pi * torch.rand(())
        phase = torch.remainder(
            cycle_phase + 2 * math.pi * torch.tensor([0.0, 0.25, 0.5, 0.75]),
            2 * math.pi,
        )
        return {
            "phase": phase,
            "phase_obs": torch.cat((phase.sin(), phase.cos())),
            "phase_frequency": torch.full((4,), 2.0),
            "last_osc_update_t": None,
        }

    def update_custom_buffers(self, main_controller, t):
        if not math.isfinite(t):
            raise ValueError("Invalid oscillator update time")

        last_t = main_controller.last_osc_update_t
        dt = 0.0 if last_t is None else t - last_t
        if not 0.0 <= dt <= 0.1:
            raise ValueError("Invalid oscillator update interval")

        frequency = main_controller.phase_frequency
        if (
            frequency.shape != (4,)
            or not torch.isfinite(frequency).all()
            or not (frequency == frequency[0]).all()
            or not (0.0 <= frequency).all()
            or not (frequency <= 5.0).all()
        ):
            raise ValueError("Expected one common oscillator frequency in [0, 5] Hz")

        main_controller.phase = torch.remainder(
            main_controller.phase + 2 * math.pi * frequency * dt, 2 * math.pi
        )
        main_controller.phase_obs = torch.cat(
            (main_controller.phase.sin(), main_controller.phase.cos())
        )
        main_controller.last_osc_update_t = t

    def get_custom_obs_pieces(self):
        return {
            "phase_obs": self._get_phase_obs,
            "phase_frequency": self._get_phase_frequency,
        }

    @staticmethod
    def _get_phase_obs(main_controller, lowstate_msg):
        return main_controller.phase_obs

    @staticmethod
    def _get_phase_frequency(main_controller, lowstate_msg):
        return main_controller.phase_frequency

