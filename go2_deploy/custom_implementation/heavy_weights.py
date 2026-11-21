"""Custom deploy implementation for the go2payload task (mjh449, gls246, rry9).

Self-contained on purpose: every value is written out instead of imported from
a training config, because the course deploy tree's Go2TrotCfg uses different
action scaling (0.2/0.3/0.3) than this policy was trained with (0.2/0.5/0.5).
tests/unit_tests/test_custom_implementation.py checks these against
Go2PayloadCfg in the training repo.
"""

import math

import torch


class CustomDeployConfig:
    # 48 values; matches Go2PayloadRunnerCfg.actor.obs. The actor observes no
    # body-height target and no payload information.
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

    # Length of observation vector piece
    custom_obs_sizes = {
        "phase_obs": 2,
        "phase_frequency": 1,
    }

    # Observations are divided by these; actions are multiplied by
    # dof_pos_target. Canonical FL, FR, RL, RR order; hip, thigh, calf per leg.
    class DeployScaling:
        base_ang_vel = 0.3
        projected_gravity = 1.0
        commands = [3, 1, 3]
        dof_pos_obs = 4 * [1.0472, 2.53075, 0.94247]
        dof_vel = 4 * [2.0, 2.0, 4.0]
        dof_pos_target = 4 * [0.2, 0.5, 0.5]
        phase_obs = 1.0
        phase_frequency = 1.0

    # Training settings the deploy config must match. DeployConfig does not read
    # these; they are listed here so they travel with the observation layout.
    trained_ctrl_freq = 50  # Hz, policy rate
    trained_kp = 20.0  # every joint
    trained_kd = 0.5  # every joint
    trained_exp_moving_avg = False  # no action smoothing in training
    # Nominal posture (rad); policy actions are offsets from it.
    trained_default_joint_angles = {"hip": 0.0, "thigh": 0.96, "calf": -1.36}
    # Joystick limits inside the trained command ranges (vx -1.0 to 2.5 m/s,
    # vy +/-0.5 m/s, yaw +/-1.5 rad/s). The stick is symmetric, so vx is capped
    # by the trained reverse speed.
    command_limits = {"lin_vel_x": 1.0, "lin_vel_y": 0.5, "yaw_vel": 1.5}


# Initialize any custom buffers your implementation uses (for 1 environment)
def get_custom_buffers_dict():
    buffer_dict = {
        # Gait frequency in Hz; training sampled 1-3 Hz.
        "phase_frequency": 2.0,
        "phase": 0.0,
    }
    return buffer_dict


# Define how your custom buffers should be updated at the control frequency
def update_custom_buffers(main_controller, t):
    main_controller.phase = (
        2 * math.pi * main_controller.phase_frequency * t % (2 * math.pi)
    )


# Return a dict mapping observation names to functions that generate the
# observation pieces
def get_custom_obs_pieces():
    obs_pieces = {
        "phase_obs": _get_obs_phase_obs,
        "phase_frequency": _get_obs_phase_frequency,
    }
    return obs_pieces


# Return the phase observation piece of the observation vector
def _get_obs_phase_obs(main_controller, lowstate_msg):
    phase = torch.tensor([main_controller.phase])
    return torch.tensor([torch.sin(phase), torch.cos(phase)])


# Return the phase frequency observation piece
def _get_obs_phase_frequency(main_controller, lowstate_msg):
    return torch.tensor([main_controller.phase_frequency])
