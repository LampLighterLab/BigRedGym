"""Go2Trot controller options for the locomotion pilots."""

from dataclasses import asdict, dataclass
import math

import mujoco
import numpy as np
import torch

from gym.envs.go2.go2trot import Go2Trot
from gym.utils.sampling import masked_update
from gym.utils.task_registry import task_registry


CONTROLLER_TASK = "go2trot_friction_controller"
ACTOR_HZ = 50
WALK_SPEEDS = (0.0, 0.5, 1.0)
WALK_PROBABILITIES = (0.2, 0.4, 0.4)
WALK_GAIT_HZ = 2.0
RAMP_S = 0.25
MOTION_THRESHOLD = 0.05  # m/s for planar speed, rad/s for yaw
PAIRED_WEIGHT = 3.0
PAIRED_CONTACT_WEIGHTS = {
    "trot_support": 0.0,
    "swing_contact": 0.0,
    "paired_contact": PAIRED_WEIGHT,
}


@dataclass(frozen=True)
class ControllerOptions:
    reference_mode: str = "reward-only"
    physics_hz: int = ACTOR_HZ
    coordination_reward: str = "legacy"
    base_height_target: float | None = None
    command_curriculum: str = "legacy"
    clip_joint_targets: bool = False

    def __post_init__(self):
        if self.reference_mode not in ("reward-only", "additive"):
            raise ValueError("Invalid gait reference mode")
        if self.physics_hz not in (50, 100):
            raise ValueError("Controller physics frequency must be 50 or 100 Hz")
        if self.coordination_reward not in ("legacy", "paired-contact"):
            raise ValueError("Invalid gait coordination reward")
        if self.command_curriculum not in ("legacy", "stand-walk"):
            raise ValueError("Invalid command curriculum")
        height = self.base_height_target
        if height is not None and (not math.isfinite(height) or height <= 0):
            raise ValueError("Base height target must be finite and positive")


def add_controller_arguments(parser):
    # None distinguishes an explicit request for the profiled controller from
    # a historical run that should keep its original Go2Trot task.
    parser.add_argument("--reference-mode", choices=("reward-only", "additive"))
    parser.add_argument("--physics-hz", type=int, choices=(50, 100))
    parser.add_argument("--coordination-reward", choices=("legacy", "paired-contact"))
    parser.add_argument(
        "--base-height-target", type=float, help="Height reward target in m"
    )
    parser.add_argument("--command-curriculum", choices=("legacy", "stand-walk"))
    parser.add_argument("--clip-joint-targets", action="store_true", default=None)


def controller_options(args):
    supplied = {
        name: getattr(args, name)
        for name in asdict(ControllerOptions())
        if getattr(args, name) is not None
    }
    return ControllerOptions(**supplied) if supplied else None


def configure_controller(env_cfg, train_cfg, options):
    env_cfg.control.ctrl_frequency = train_cfg.actor.frequency = ACTOR_HZ
    env_cfg.control.desired_sim_frequency = options.physics_hz
    if options.base_height_target is not None:
        env_cfg.reward_settings.base_height_target = options.base_height_target
    if options.command_curriculum == "stand-walk":
        env_cfg.commands.ranges.lin_vel_x = list(WALK_SPEEDS)
        env_cfg.commands.ranges.lin_vel_y = env_cfg.commands.ranges.yaw_vel = 0.0
        env_cfg.commands.var = 0.0
        env_cfg.control.gait_freq = [WALK_GAIT_HZ, WALK_GAIT_HZ]
        env_cfg.push_robots.toggle = False
    if options.coordination_reward == "paired-contact":
        enable_paired_contact_reward(train_cfg)


def enable_paired_contact_reward(train_cfg):
    for name, weight in PAIRED_CONTACT_WEIGHTS.items():
        setattr(train_cfg.critic.reward.weights, name, weight)


def register_controller(env_cfg, train_cfg, options, *, scripted=False):
    env_cfg._controller_options = options
    env_cfg._scripted_commands = scripted
    task_registry.register(CONTROLLER_TASK, ControllerGo2Trot, env_cfg, train_cfg)
    return CONTROLLER_TASK


def moving_command(command):
    return (command[:, :2].square().sum(1) > MOTION_THRESHOLD**2) | (
        command[:, 2].abs() > MOTION_THRESHOLD
    )


def paired_contact_score(leg_phase_sine, normal_load_n, body_weight_n, moving):
    load = (normal_load_n / (0.1 * body_weight_n)).clamp(0, 1)
    stance = leg_phase_sine > 0
    weaker_stance = load.masked_fill(~stance, 1).amin(dim=1)
    stronger_swing = load.masked_fill(stance, 0).amax(dim=1)
    return (
        moving * leg_phase_sine.abs().amax(dim=1) * weaker_stance * (1 - stronger_swing)
    )


class ControllerGo2Trot(Go2Trot):
    def __init__(self, cfg, device, headless, backend):
        self.controller_options = cfg._controller_options
        self.scripted_commands = cfg._scripted_commands
        super().__init__(cfg, device, headless, backend)

    def _initialize_sim(self):
        super()._initialize_sim()
        if self.controller_options.physics_hz == ACTOR_HZ or not hasattr(
            self._backend, "_model_for_env"
        ):
            return
        # MuJoCo normally clamps each positive time constant to 2*sim_dt.
        # Match the effective 50 Hz constraints in the CPU rate comparison.
        models = self._backend._models or (self._backend._model_for_env(0),)
        for model in models:
            if model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_REFSAFE:
                continue
            for constants in (model.geom_solref[:, 0], model.jnt_solref[:, 0]):
                np.maximum(constants, 2 / ACTOR_HZ, out=constants, where=constants > 0)

    def _init_buffers(self):
        super()._init_buffers()
        self._nominal_target = self.default_dof_pos.index_select(
            1, self.actuated_dof_indices
        )
        self.gait_ramp = torch.zeros_like(self.phase)
        self.gait_blend = torch.zeros_like(self.phase)
        self.pd_joint_target = torch.zeros_like(self.dof_pos_target)
        self.clipped_joint_targets = torch.zeros_like(
            self.phase[:, 0], dtype=torch.long
        )
        self.joint_target_samples = self.num_actuators * self.cfg.control.decimation
        if self.controller_options.command_curriculum == "stand-walk":
            self._command_rng = torch.Generator(device=self.device).manual_seed(
                self.cfg.seed
            )
            self._walk_speeds = self.commands.new_tensor(WALK_SPEEDS)
            self._walk_cutoffs = self.commands.new_tensor(WALK_PROBABILITIES).cumsum(0)[
                :-1
            ]

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        self.gait_ramp.masked_fill_(reset_mask[:, None], 0.0)
        self.gait_blend.masked_fill_(reset_mask[:, None], 0.0)

    def _resample_commands(self, command_mask):
        if self.scripted_commands:
            return
        if self.controller_options.command_curriculum == "legacy":
            super()._resample_commands(command_mask)
            return
        draws = torch.rand(
            self.num_envs, device=self.device, generator=self._command_rng
        )
        vx = self._walk_speeds[torch.bucketize(draws, self._walk_cutoffs, right=True)]
        masked_update(self.commands[:, 0], vx, command_mask)
        self.commands[:, 1:].masked_fill_(command_mask[:, None], 0.0)

    def _pre_decimation_step(self):
        super()._pre_decimation_step()
        self.clipped_joint_targets.zero_()

    def _compute_torques(self):
        requested = self._nominal_target + self.dof_pos_target
        if self.controller_options.reference_mode == "additive":
            direction = (
                moving_command(self.commands)[:, None]
                .to(self.gait_ramp)
                .mul_(2)
                .sub_(1)
            )
            self.gait_ramp.add_(direction, alpha=self.cfg.sim_dt / RAMP_S).clamp_(0, 1)
            u = self.gait_ramp
            self.gait_blend.copy_(u.square() * (3 - 2 * u))
            self._update_gait_reference()
            requested = requested + self.gait_blend * self.gait_reference

        if self.controller_options.clip_joint_targets:
            low, high = self._actuated_dof_pos_limits.unbind(1)
            target = requested.clamp(min=low, max=high)
            self.clipped_joint_targets.add_((target != requested).sum(dim=1))
        else:
            target = requested
        self.pd_joint_target.copy_(target)

        pos = self.dof_pos.index_select(1, self.actuated_dof_indices)
        vel = self.dof_vel.index_select(1, self.actuated_dof_indices)
        torque = (
            self.p_gains * (target - pos)
            + self.d_gains * (self.dof_vel_target - vel)
            + self.tau_ff
        )
        return torque.clamp(-self.actuated_torque_limits, self.actuated_torque_limits)

    def _reward_paired_contact(self):
        return paired_contact_score(
            torch.sin(self._leg_phases()),
            self.contact_forces[:, self.feet_indices, 2],
            9.81 * self._backend.link_mass.sum(dim=1, keepdim=True),
            moving_command(self.commands),
        )
