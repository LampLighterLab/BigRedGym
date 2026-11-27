"""Post-step measurements shared by the controller pilots and evaluator."""

import math

import torch

from evaluation_contacts import FootGroundContacts
from gait_controller import moving_command


MIN_LOAD_N = 5.0
SLIP_SPEED_M_S = 0.1
SUSTAINED_SUPPORT_S = 0.10

COUNT_FIELDS = (
    "loaded_foot_samples",
    "slipping_foot_samples",
    "moving_samples",
    "matching_diagonal_samples",
    "non_foot_support_samples",
    "sustained_non_foot_support_samples",
    "joint_target_samples",
    "clipped_joint_target_samples",
    "falls",
)
RATE_FIELDS = (
    "foot_slip_rate",
    "diagonal_contact_agreement",
    "sustained_non_foot_support_fraction",
    "joint_target_clip_fraction",
)


def contact_force_source(env):
    if hasattr(env._backend, "_datas"):
        return "MuJoCo world-ground contacts, last physics substep"
    return "net upward body contact forces, may include self-contact"


class ControllerTelemetry:
    def __init__(self, env, contacts=None):
        self.env = env
        if contacts is None and hasattr(env._backend, "_datas"):
            contacts = FootGroundContacts(env)
        self.contacts = contacts
        if self.contacts is None:
            self.non_foot_mask = torch.ones(
                env.num_bodies, dtype=torch.bool, device=env.device
            )
            self.non_foot_mask[env.feet_indices] = False
        self.support_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        self.sustained_steps = math.ceil(SUSTAINED_SUPPORT_S / env.dt)

    def sample(self, command, *, ground_forces=None, non_foot_load=None):
        env = self.env
        if ground_forces is not None:
            normal_force = ground_forces[:, :, 2]
        elif self.contacts is not None:
            normal_force, non_foot_load = self.contacts.read_vertical_loads()
        else:
            normal_force = env.contact_forces[:, env.feet_indices, 2]
            non_foot_load = (
                env.contact_forces[:, self.non_foot_mask, 2].clamp_min(0).sum(1)
            )
        normal_force = torch.as_tensor(normal_force, device=env.device)
        non_foot_load = torch.as_tensor(non_foot_load, device=env.device)
        stance = normal_force > MIN_LOAD_N
        speed_sq = env._rigid_body_lin_vel[:, env.feet_indices, :2].square().sum(dim=2)
        moving = moving_command(command)
        intended = env._expected_stance()
        supported = non_foot_load > MIN_LOAD_N
        self.support_steps.add_(1).masked_fill_(~supported, 0)
        sustained = self.support_steps >= self.sustained_steps
        duration = self.support_steps * env.dt
        self.support_steps.masked_fill_(env.terminated | env.timed_out, 0)
        return {
            "gait_phase_rad": env.phase[:, 0],
            "gait_frequency_hz": env.phase_frequency[:, 0],
            "gait_reference_blend": env.gait_blend[:, 0],
            "commanded_motion": moving,
            "intended_stance": intended,
            "loaded_foot_contact": stance,
            "foot_slip": stance & (speed_sq > SLIP_SPEED_M_S**2),
            "matches_diagonal_contact": moving & (stance == intended).all(dim=1),
            "non_foot_support_N": non_foot_load,
            "non_foot_support_duration_s": duration,
            "sustained_non_foot_support": sustained,
            "pd_joint_target_rad": env.pd_joint_target,
            "joint_position_rad": env.dof_pos[:, env.actuated_dof_indices],
            "joint_target_samples": torch.full_like(
                env.clipped_joint_targets, env.joint_target_samples
            ),
            "clipped_joint_target_samples": env.clipped_joint_targets,
        }


def controller_counts(sample, fell):
    return (
        sample["loaded_foot_contact"].sum(1),
        sample["foot_slip"].sum(1),
        sample["commanded_motion"],
        sample["matches_diagonal_contact"],
        sample["non_foot_support_N"] > MIN_LOAD_N,
        sample["sustained_non_foot_support"],
        sample["joint_target_samples"],
        sample["clipped_joint_target_samples"],
        fell,
    )


def controller_rates(totals, samples):
    loaded, slipping, moving, matching, _, sustained, targets, clipped, _ = totals
    return (
        slipping / loaded if loaded else None,
        matching / moving if moving else None,
        sustained / samples if samples else None,
        clipped / targets if targets else None,
    )
