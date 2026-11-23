"""Go2 oscillator observations and buffers for the saved go2osc policy."""

import math

import torch

from go2_deploy.custom_implementation.custom_config import CustomConfig


class Go2OscCustom(CustomConfig):
    task_name = "go2osc"

    # Uncalibrated Go2 EDU baseline. Only the approximate 15 kg mass comes
    # from the model specification. The SDK's foot_force values are integers,
    # not documented as newtons, and their FL/FR/RL/RR order is unverified.
    # Replace offsets, gains, and order with measurements from this robot.
    force_calibration = {
        "offsets": [0.0, 0.0, 0.0, 0.0],
        "gains": [1.0, 1.0, 1.0, 1.0],
        "order": [0, 1, 2, 3],
        "mass_kg": 15.0,
    }

    # Actor observation order from the old go2osc training run (61 values).
    obs_vector = [
        "base_ang_vel",
        "projected_gravity",
        "commands",
        "dof_pos_obs",
        "dof_vel",
        "dof_pos_target",
        "phase_obs",
        "phase_frequency",
        "measured_grf",
    ]

    custom_obs_sizes = {
        "phase_obs": 8,
        "phase_frequency": 4,
        "measured_grf": 4,
    }

    # FL, FR, RL, RR; hip, thigh, calf within each leg, in radians.
    default_dof_pos = [0.0, 0.66, -1.36] * 4

    class DeployScaling:
        base_ang_vel = 0.3
        commands = [3.0, 1.0, 3.0]
        dof_pos_obs = [1.0472, 2.53075, 0.94247] * 4
        dof_vel = [2.0, 2.0, 4.0] * 4
        dof_pos_target = [0.2, 0.3, 0.3] * 4
        # projected_gravity, phase_obs, phase_frequency, and measured_grf
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
            "measured_grf": torch.zeros(4),
            "last_osc_update_t": None,
        }

    def update_custom_buffers(self, main_controller, t):
        if self.force_calibration is None:
            raise RuntimeError("go2osc foot-force calibration is required")
        if not math.isfinite(t):
            raise ValueError("Invalid oscillator update time")

        with main_controller.last_lowstate_msg_lock:
            msg = main_controller.last_lowstate_msg
            if msg is None:
                raise RuntimeError("No LowState foot-force measurements received")
            raw_forces = tuple(msg.foot_force)

        grf = self.normalize_foot_forces(raw_forces, self.force_calibration)
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

        main_controller.measured_grf = grf
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
            "measured_grf": self._get_measured_grf,
        }

    @staticmethod
    def _get_phase_obs(main_controller, lowstate_msg):
        return main_controller.phase_obs

    @staticmethod
    def _get_phase_frequency(main_controller, lowstate_msg):
        return main_controller.phase_frequency

    @staticmethod
    def _get_measured_grf(main_controller, lowstate_msg):
        return main_controller.measured_grf

    @staticmethod
    def normalize_foot_forces(raw, calibration):
        """Return FL, FR, RL, RR vertical forces divided by robot weight."""
        try:
            offsets = torch.as_tensor(calibration["offsets"], dtype=torch.float32)
            gains = torch.as_tensor(calibration["gains"], dtype=torch.float32)
            order = list(calibration["order"])
            mass_kg = float(calibration["mass_kg"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid foot-force calibration") from exc

        raw = torch.as_tensor(raw, dtype=torch.float32)
        if any(x.shape != (4,) for x in (raw, offsets, gains)):
            raise ValueError("Expected four foot-force readings, offsets, and gains")
        if sorted(order) != [0, 1, 2, 3]:
            raise ValueError("Foot-force order must contain indices 0, 1, 2, 3")
        if not torch.isfinite(torch.cat((raw, offsets, gains))).all():
            raise ValueError("Foot-force readings and calibration must be finite")
        if not math.isfinite(mass_kg) or mass_kg <= 0 or (gains <= 0).any():
            raise ValueError("Foot-force gains and robot mass must be positive")

        force_newtons = (raw - offsets) * gains
        return force_newtons[order].clamp_min(0.0) / (mass_kg * 9.81)
