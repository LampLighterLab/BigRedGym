"""Deploy the saved ORC111 forward4096 policy through the custom hooks.

Select with GO2_CUSTOM_IMPLEMENTATION=pd_emergent.py before starting deployment.
The TA must supply GO2_FORCE_SENSOR_ORDER (comma-separated leg names) and
GO2_FORCE_NEWTONS_PER_UNIT (one value or four comma-separated values).
GO2_FORCE_ZERO_OFFSET optionally supplies offsets in the same sensor order.
Those calibration values can also be set on CustomDeployConfig below.

Oscillators use FR, FL, RR, RL order; joint vectors use FL, FR, RL, RR order.
Scalar hardware force approximates the full 3D force norm used in simulation.
Training commands were [1, 0, 0]; the joystick remains under operator control.
"""

import math
import os

import torch


class CustomDeployConfig:
    task_name = "go2_osc"
    ctrl_freq = 100
    kp = 20.0
    kd = 0.5
    default_dof_pos = torch.tensor([0.0, 0.66, -1.36] * 4)
    exp_moving_avg = False
    command_limits = {"lin_vel_x": 1.0, "lin_vel_y": 0.0, "yaw_vel": 0.0}
    load_run = "Nov16_06-26-08_ORC111_forward4096_reference_smoothness"
    checkpoint = 2000

    obs_vector = [
        "base_ang_vel",
        "projected_gravity",
        "commands",
        "dof_pos_obs",
        "dof_vel",
        "oscillator_obs",
        "dof_pos_target",
    ]
    custom_obs_sizes = {"oscillator_obs": 8}

    class DeployScaling:
        base_ang_vel = [0.3, 0.3, 0.1]
        projected_gravity = 1.0
        commands = [3.0, 1.0, 3.0]
        dof_pos_obs = [0.2, 0.3, 0.3] * 4
        dof_vel = [2.0, 2.0, 4.0] * 4
        oscillator_obs = 1.0
        # Also used to multiply the actor's raw output into radian offsets.
        dof_pos_target = [0.2, 0.3, 0.3] * 4

    # Set these from measured/confirmed hardware conventions, not assumed units.
    # Conversion and offsets may be scalar or one value per sensor channel.
    # Signed conversion supports sensors whose compression readings decrease.
    force_sensor_order = None
    force_newtons_per_unit = None
    force_zero_offset = 0.0

    # Nominal saved Go2 model: URDF inertial masses sum to 16.087 kg.
    # This denominator matches simulated robot weight, not per-leg weight.
    training_weight_newtons = 16.087 * 9.81

    # Saved training configuration: parameter noise per control tick and
    # independent process noise per leg at each 500 Hz integration substep.
    omega_var = 0.25
    coupling_var = 0.25
    process_noise_std = 0.25


def _force_calibration():
    """Resolve explicit calibration before the controller opens robot clients."""
    cfg = CustomDeployConfig
    expected_legs = ("FR", "FL", "RR", "RL")
    sensor_order = os.environ.get("GO2_FORCE_SENSOR_ORDER", cfg.force_sensor_order)
    if isinstance(sensor_order, str):
        sensor_order = tuple(leg.strip().upper() for leg in sensor_order.split(","))
    if (
        sensor_order is None
        or len(sensor_order) != 4
        or set(sensor_order) != set(expected_legs)
    ):
        raise ValueError(
            "The TA must set GO2_FORCE_SENSOR_ORDER to the four verified "
            "LowState foot_force leg names, e.g. FR,FL,RR,RL only if confirmed."
        )

    def sensor_values(environment_name, configured_value):
        value = os.environ.get(environment_name, configured_value)
        if value is None:
            raise ValueError(
                f"The TA must set {environment_name} from the hardware calibration."
            )
        if isinstance(value, str):
            value = [float(part.strip()) for part in value.split(",")]
        values = torch.as_tensor(value, dtype=torch.float32).flatten()
        if values.numel() == 1:
            values = values.expand(4)
        if values.shape != (4,) or not torch.isfinite(values).all():
            raise ValueError(
                f"{environment_name} must contain one or four finite values."
            )
        return values.clone()

    conversion = sensor_values("GO2_FORCE_NEWTONS_PER_UNIT", cfg.force_newtons_per_unit)
    if (conversion == 0).any():
        raise ValueError("GO2_FORCE_NEWTONS_PER_UNIT must be nonzero for every sensor.")
    offsets = sensor_values("GO2_FORCE_ZERO_OFFSET", cfg.force_zero_offset)
    weight = float(cfg.training_weight_newtons)
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("training_weight_newtons must be positive and finite.")
    indices = torch.tensor(
        [sensor_order.index(leg) for leg in expected_legs], dtype=torch.long
    )
    return indices, conversion, offsets, weight


def get_custom_buffers_dict():
    """Initialize one robot's phases and validate the force conversion."""
    indices, conversion, zero_offset, weight = _force_calibration()
    phases = torch.rand(4) * (2 * math.pi)
    return {
        "oscillators": phases,
        "oscillators_vel": torch.full((4,), 2 * math.pi),
        "oscillator_obs": torch.cat((phases.cos(), phases.sin())),
        "grf": torch.zeros(4),
        "osc_omega": torch.tensor([1.0]),
        "osc_coupling": torch.tensor([4.0]),
        "osc_offset": torch.tensor([1.0]),
        "osc_force_indices": indices,
        "osc_force_zero_offset": zero_offset,
        "osc_force_conversion": conversion,
        "osc_training_weight": weight,
    }


def update_custom_buffers(main_controller, t):
    """Advance one control interval, using the latest received foot forces.

    The template calls this before refreshing last_command. That command is
    therefore the one held over the preceding interval being integrated here.
    Five numerical substeps use the same latest force sample; this differs
    from the five fresh physics force samples available during training.
    """
    del t  # Integrate one nominal control interval, rather than wall-clock time.
    mc = main_controller
    with mc.last_lowstate_msg_lock:
        msg = mc.last_lowstate_msg
        if msg is None:
            return
        raw_force = torch.as_tensor(list(msg.foot_force), dtype=torch.float32)

    force_newtons = (
        (raw_force - mc.osc_force_zero_offset) * mc.osc_force_conversion
    ).clamp_min(0.0)
    ordered_force = force_newtons.index_select(0, mc.osc_force_indices)
    mc.grf.copy_((ordered_force / mc.osc_training_weight).clamp(0.0, 1.0))

    with mc.last_command_lock:
        speed = abs(float(mc.last_command[0]))

    stopped = speed <= 0.5
    omega = 1.0 if stopped else min(4.0, 2.0 + speed - 0.5)
    coupling = 4.0 if stopped else 1.0
    cfg = CustomDeployConfig
    mc.osc_omega.copy_((omega + cfg.omega_var * torch.randn(1)).clamp_min(0.1))
    mc.osc_coupling.copy_((coupling + cfg.coupling_var * torch.randn(1)).clamp_min(0.0))
    mc.osc_offset.fill_(1.0 if stopped else 0.0)

    substep_dt = (1.0 / mc.cfg.ctrl_freq) / 5
    for _ in range(5):
        feedback = mc.osc_coupling * mc.grf * (mc.oscillators.cos() + mc.osc_offset)
        velocity = 2 * math.pi * (mc.osc_omega - feedback)
        velocity += 2 * math.pi * cfg.process_noise_std * torch.randn_like(velocity)
        mc.oscillators_vel.copy_(velocity)
        mc.oscillators.add_(velocity * substep_dt).remainder_(2 * math.pi)

    mc.oscillator_obs.copy_(torch.cat((mc.oscillators.cos(), mc.oscillators.sin())))


def get_custom_obs_pieces():
    return {"oscillator_obs": _get_obs_oscillator_obs}


def _get_obs_oscillator_obs(main_controller, lowstate_msg):
    # Raw values: cos(FR, FL, RR, RL), then sin(FR, FL, RR, RL).
    # The host applies DeployScaling when it assembles all 53 observations.
    return main_controller.oscillator_obs
