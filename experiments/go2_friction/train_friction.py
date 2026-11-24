"""Train one Go2trot policy at a friction range, actor noise scale, and seed.

Place in BigRedGym/experiments/go2_friction/ and launch with:
    uv run --frozen experiments/go2_friction/train_friction.py --task go2trot \
        --min-friction 0.6 --max-friction 1.25 --seed 42 \
        --max_iterations 200 --num_envs 256 --headless --disable_wandb

Reuses BigRedGym's *actual* scripts/train.py setup and policy runner.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from gym.utils.helpers import class_to_dict
from sweep_friction import (
    REFERENCE_FRICTION,
    experiment_name,
    noise_experiment_name,
    value_name,
)

# Reuse BigRedGym core training code without copying or modifying it.
CORE_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(CORE_SCRIPTS))
from train import make_train_parser, setup, train  # noqa: E402


def configure_friction(
    env_cfg, minimum: float, maximum: float, *, friction_only: bool
) -> None:
    """Change only the requested experimental condition.

    Other DR axes can be disabled completely to isolate friction. No mutation
    of BigRedGym's version-controlled configuration source is required.
    """
    if not math.isfinite(maximum) or not 0.0 <= minimum <= maximum:
        raise ValueError(f"Invalid friction range [{minimum}, {maximum}]")
    dr = env_cfg.domain_randomization
    dr.startup.contact_friction_range = [float(minimum), float(maximum)]
    if friction_only:
        dr.startup.link_mass_scale_range = None
        dr.episode.scale_ranges = {}


def configure_observation_noise(train_cfg, scale: float) -> dict:
    defaults = train_cfg.actor.noise
    default_std = {
        "base_ang_vel": defaults.base_ang_vel,
        "projected_gravity": defaults.gravity_vec,
        "dof_pos_obs": defaults.dof_pos_obs,
        "dof_vel": defaults.dof_vel,
    }
    # The on-policy rollout reads this map directly. An empty map produces
    # zero perturbation while preserving the normal rollout's RNG draws.
    train_cfg.actor.noise = (
        {
            name: (torch.as_tensor(std, dtype=torch.float64) * scale).tolist()
            for name, std in default_std.items()
        }
        if scale
        else {}
    )
    train_cfg.actor.add_noise = bool(train_cfg.actor.noise)
    return default_std


def reward_stance_foot_slip(self):
    """Penalize horizontal foot sliding during stance."""
    stance = torch.relu(torch.sin(self._leg_phases()))

    # Only penalize feet actually touching the ground
    normal_force = self.contact_forces[:, self.feet_indices, 2]
    in_contact = (normal_force > 5.0).float()

    # World-frame horizontal foot velocity
    foot_vel_xy = self._rigid_body_lin_vel[:, self.feet_indices, :2]
    speed_squared = foot_vel_xy.square().sum(dim=-1)

    # Normalize by 0.5 m/s and cap large penalties
    slip = (speed_squared / 0.25).clamp(max=4.0)

    return -(stance * in_contact * slip).mean(dim=1)


def reward_swing_foot_clearance(self):
    """Penalize feet that fail to lift during swing."""
    swing = torch.relu(-torch.sin(self._leg_phases()))

    # Initial target: foot-link origin 8 cm above flat ground
    target_z = 0.08
    foot_z = self._rigid_body_pos[:, self.feet_indices, 2]

    deficit = ((target_z - foot_z) / target_z).clamp(
        min=0.0, max=1.0
    )

    # Don't require stepping when commanded to stand still
    moving = (
        (self.commands[:, :2].norm(dim=1) > 0.2)
        | (self.commands[:, 2].abs() > 0.2)
    ).float()

    return -(swing * deficit.square()).mean(dim=1) * moving


def main(argv=None) -> None:
    parser = make_train_parser()
    parser.add_argument("--min-friction", type=float, required=True)
    parser.add_argument("--max-friction", type=float, default=REFERENCE_FRICTION)
    parser.add_argument(
        "--noise-scale",
        type=float,
        help="Multiplier for the actor's IMU and joint noise (default: 0, disabled)",
    )
    parser.add_argument(
        "--rollout-size", type=int, help="Override transitions per PPO rollout"
    )

    parser.add_argument(
        "--gait-rewards",
        action="store_true",
        help="Enable foot-slip and swing-clearance penalties",
    )

    parser.add_argument(
    "--easy-gait",
    action="store_true",
    help="Use a simplified forward-walking training curriculum",
    )

    parser.add_argument(
        "--keep-other-randomization", action="store_true",
        help="Keep mass and PD-gain randomization instead of friction-only training.",
    )
    args = parser.parse_args(argv)
    if args.task != "go2trot":
        parser.error("This experiment is designed for --task go2trot")
    if args.seed is None:
        parser.error("Supply --seed so results are reproducible")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be an integer in [0, 2**32 - 1]")
    if not (
        math.isfinite(args.max_friction) and 0 <= args.min_friction <= args.max_friction
    ):
        parser.error("Friction bounds must be finite and satisfy 0 <= min <= max")
    noise_scale = 0.0 if args.noise_scale is None else args.noise_scale
    if not math.isfinite(noise_scale) or noise_scale < 0:
        parser.error("--noise-scale must be finite and nonnegative")
    if args.rollout_size is not None and args.rollout_size < 1:
        parser.error("--rollout-size must be positive")
    if args.original_cfg:
        parser.error("Do not use --original_cfg with this experiment.")
    if args.resume and args.load_run is None:
        parser.error("Supply --load_run when resuming training.")
    if args.experiment_name is None:
        fixed_reference = args.min_friction == args.max_friction == REFERENCE_FRICTION
        if args.noise_scale is not None and fixed_reference:
            args.experiment_name = noise_experiment_name(noise_scale, args.seed)
        else:
            args.experiment_name = experiment_name(
                args.min_friction, args.seed, upper=args.max_friction
            )
            if noise_scale:
                args.experiment_name += f"_noise_scale_{value_name(noise_scale)}"

    # Task registration must occur before asking for the configuration.
    import gym.envs  # noqa: F401
    from gym.utils.task_registry import task_registry

    env_cfg, train_cfg = deepcopy(task_registry.get_cfgs(args.task))
    default_noise = configure_observation_noise(train_cfg, noise_scale)
    if args.rollout_size is not None:
        train_cfg.algorithm.rollout_size = args.rollout_size

    if args.gait_rewards:
        from gym.envs.go2.go2trot import Go2Trot

        Go2Trot._reward_stance_foot_slip = reward_stance_foot_slip
        Go2Trot._reward_swing_foot_clearance = reward_swing_foot_clearance

        train_cfg.critic.reward.weights.stance_foot_slip = 0.5
        train_cfg.critic.reward.weights.swing_foot_clearance = 1.0
    configure_friction(
        env_cfg, args.min_friction, args.max_friction,
        friction_only=not args.keep_other_randomization,
    )

    # Penalize head and calf contact with the ground
    env_cfg.asset.penalize_contacts_on = ["Head_", "calf"]
    # Positive weight because _reward_collision returns negative values
    train_cfg.critic.reward.weights.collision = 0.5
    train_cfg.runner.save_interval = 25

    if args.easy_gait:
        if not args.gait_rewards:
            parser.error("--easy-gait requires --gait-rewards")

        # Learn forward walking before turning or moving sideways.
        env_cfg.commands.ranges.lin_vel_x = [0.5, 1.0]
        env_cfg.commands.ranges.lin_vel_y = 0.0
        env_cfg.commands.ranges.yaw_vel = 0.0
        env_cfg.commands.var = 0.0

        # Start with a consistent, relatively slow trot.
        env_cfg.control.gait_freq = [1.5, 1.5]

        # No external disturbances during initial gait learning.
        env_cfg.push_robots.toggle = False

        # Give foot lifting and avoiding sliding more importance.
        train_cfg.critic.reward.weights.stance_foot_slip = 1.0
        train_cfg.critic.reward.weights.swing_foot_clearance = 3.0

        # Encourage a level, supported body.
        train_cfg.critic.reward.weights.orientation = 1.5
        train_cfg.critic.reward.weights.min_base_height = 1.0

        # Stronger incentives for alternating diagonal leg support
        train_cfg.critic.reward.weights.trot_support = 6.0
        train_cfg.critic.reward.weights.swing_contact = 5.0
    print(
        f"Training {args.experiment_name}: "
        f"friction=[{args.min_friction}, {args.max_friction}], "
        f"noise scale={noise_scale}, seed={args.seed}"
    )
    train_cfg, runner = setup(args, env_cfg=env_cfg, train_cfg=train_cfg)
    log_dir = Path(train_cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    applied_friction = runner.env.domain_randomizer.contact_friction
    manifest = {
        "experiment_name": args.experiment_name,
        "task": args.task,
        "seed": args.seed,
        "friction_range": [args.min_friction, args.max_friction],
        "applied_friction_range": [
            applied_friction.min().item(), applied_friction.max().item()
        ],
        "friction_only": not args.keep_other_randomization and noise_scale == 0.0,
        "noise_scale": noise_scale,
        "observation_noise": {
            "distribution": "gaussian",
            "default_std": default_noise,
            "effective_std": runner.actor_cfg["noise"],
        },
        "gait_rewards_enabled": args.gait_rewards,
        "max_iterations": train_cfg.runner.max_iterations,
        "rollout_size": train_cfg.algorithm.rollout_size,
        "num_envs": env_cfg.env.num_envs,
        "device": args.device,
        "backend": args.backend,
        "train_pushes_enabled": bool(env_cfg.push_robots.toggle),
        "friction_resampling": "once per simulated environment at startup",
        "env_config": class_to_dict(env_cfg),
        "train_config": class_to_dict(train_cfg),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    path = log_dir / "friction_experiment.json"

    def save_manifest():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    save_manifest()
    try:
        train(train_cfg, runner)
    except BaseException as exc:
        manifest["status"] = "interrupted_or_failed"
        manifest["error"] = repr(exc)
        save_manifest()
        raise
    else:
        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        save_manifest()
        print(f"Finished! Log/checkpoints: {log_dir}")


if __name__ == "__main__":
    main()
