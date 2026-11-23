import math
import torch
from gym.envs.go2.go2osc_config import Go2OscCfg, Go2OscRunnerCfg
from gym.envs.go2.go2_oscillators import phase_rate, step, desired_force

class CustomDeployConfig:

    kp = 20.0
    kd = 0.5

    exp_moving_avg = False

    command_limits = {"lin_vel_x": 0.7,"lin_vel_y": 0.0,"yaw_vel": 0.0,}

    force_calibration = None

    task_name = "go2osc"

    ctrl_freq = Go2OscCfg.control.ctrl_frequency

    default_dof_pos = torch.tensor([0.0, 0.66, -1.36] * 4)
    obs_vector = list(Go2OscRunnerCfg.actor.obs)

    custom_obs_sizes = {
        "phase_obs": 8,
        "phase_frequency": 4,
        "measured_grf": 4,
        "target_grf": 4,
    }

    hardware_approved = False

    joint_position_limits = None

    max_sensor_age_s = 0.1

    class DeployScaling(Go2OscCfg.scaling):
        pass

# Initialize any custom buffers your implementation uses (for 1 environment)
def get_custom_buffers_dict():
    phase = 2 * math.pi * torch.rand(4)

    return {
        "phase": phase,
        "phase_obs": torch.cat(
            (torch.sin(phase), torch.cos(phase))
        ),
        "phase_frequency": torch.full((4,), Go2OscCfg.osc.omega),
        "measured_grf": torch.zeros(4),
        "target_grf": desired_force(phase, Go2OscCfg.osc.amplitude),
        "last_osc_update_t": None,
    }

# Define how your custom buffers should be updated at the control frequency
def advance_oscillators(controller, normalized_grf, dt):
    osc = Go2OscCfg.osc

    controller.measured_grf = normalized_grf.clamp_min(0.0)
    grf = controller.measured_grf.clamp(max=1.0)

    controller.phase = step(
        controller.phase,
        grf,
        osc.omega,
        osc.sigma,
        osc.xi,
        dt,
    )

    controller.target_grf = desired_force(
        controller.phase, osc.amplitude
    )

    controller.phase_obs = torch.cat(
        (torch.sin(controller.phase),
         torch.cos(controller.phase))
    )

    controller.phase_frequency = phase_rate(
        controller.phase,
        grf,
        osc.omega,
        osc.sigma,
        osc.xi,
    ) / (2 * math.pi)

def update_custom_buffers(controller, t):
    cal = CustomDeployConfig.force_calibration

    if cal is None:
        raise RuntimeError("Foot force calibration required")

    msg = controller.last_lowstate_msg

    if msg is None:
        raise RuntimeError("No robot sensor data")

    grf = normalize_foot_forces(
        raw=getattr(msg, cal["field"]),
        offsets=cal["offsets"],
        gains=cal["gains"],
        order=cal["order"],
        mass_kg=cal["mass_kg"],
    )

    last_t = controller.last_osc_update_t
    dt = 0.0 if last_t is None else t - last_t
    if not math.isfinite(dt) or dt < 0 or dt > 0.1:
        raise RuntimeError("Invalid oscillator update interval")

    controller.last_osc_update_t = t

    # Reproduce the simulation's 200 Hz integration.
    sim_dt = 1.0 / Go2OscCfg.control.desired_sim_frequency
    substeps = max(1, math.ceil(dt / sim_dt))

    for _ in range(substeps):
        advance_oscillators(controller, grf, dt / substeps)

# Return a dict mapping observation names to functions that generate the observation pieces
def get_custom_obs_pieces():
    return {
        "phase_obs": _get_obs_phase_obs,
        "phase_frequency": _get_obs_phase_frequency,
        "measured_grf": _get_obs_measured_grf,
        "target_grf": _get_obs_target_grf,
    }


def _get_obs_phase_obs(controller, lowstate_msg):
    return controller.phase_obs


def _get_obs_phase_frequency(controller, lowstate_msg):
    return controller.phase_frequency


def _get_obs_measured_grf(controller, lowstate_msg):
    return controller.measured_grf


def _get_obs_target_grf(controller, lowstate_msg):
    return controller.target_grf

def normalize_foot_forces(raw, offsets, gains, order, mass_kg):
    raw = torch.as_tensor(raw, dtype=torch.float32)
    offsets = torch.as_tensor(offsets, dtype=torch.float32)
    gains = torch.as_tensor(gains, dtype=torch.float32)

    if any(x.shape != (4,) for x in (raw, offsets, gains)):
        raise ValueError("Expected four foot force measurements")

    if not torch.isfinite(torch.cat((raw, offsets, gains))).all():
        raise ValueError("Invalid foot force data")

    if sorted(order) != [0, 1, 2, 3]:
        raise ValueError("Invalid foot ordering")

    if not math.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("Invalid robot mass")

    if (gains == 0).any():
        raise ValueError("Invalid calibration gains")

    forces = (raw - offsets) * gains
    forces = forces[list(order)]

    return forces.clamp_min(0.0) / (mass_kg * 9.81)


def validate_hardware(cfg):
    custom = CustomDeployConfig
    cal = custom.force_calibration

    if not custom.hardware_approved:
        raise RuntimeError("Hardware configuration not approved")

    if cal is None:
        raise RuntimeError("Foot force calibration required")

    if cal["field"] not in ("foot_force", "foot_force_est"):
        raise ValueError("Invalid force sensor field")

    normalize_foot_forces(
        cal["offsets"], cal["offsets"],
        cal["gains"], cal["order"], cal["mass_kg"]
    )

    if custom.joint_position_limits is None:
        raise RuntimeError("Joint position limits required")

    lower, upper = [
        torch.as_tensor(x) for x in custom.joint_position_limits
    ]

    if lower.shape != (12,) or upper.shape != (12,):
        raise ValueError("Expected 12 joint limits")

    if not torch.isfinite(torch.cat((lower, upper))).all():
        raise ValueError("Invalid joint limits")

    if not (lower < upper).all():
        raise ValueError("Invalid joint limit ranges")


def validate_action(controller, action):
    lower, upper = [
        torch.as_tensor(x)
        for x in CustomDeployConfig.joint_position_limits
    ]

    target = controller.cfg.default_dof_pos + action

    if ((target < lower) | (target > upper)).any():
        raise RuntimeError("Joint target outside approved limits")

    vx = controller.last_command[0].item()

    if not 0.3 - 1e-6 <= vx <= 0.7 + 1e-6:
        raise RuntimeError("Command outside training range")
    