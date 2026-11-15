"""Train one Go2trot policy at one friction range and seed.

Place in BigRedGym/experiments/go2_friction/ and launch with:
    uv run --frozen experiments/go2_friction/train_friction.py --task go2trot \
        --min-friction 0.6 --max-friction 1.25 --seed 42 \
        --max_iterations 200 --num_envs 256 --headless --disable_wandb

Reuses BigRedGym's *actual* scripts/train.py setup and policy runner.
"""
from __future__ import annotations
import torch
import json
from datetime import datetime, timezone
from pathlib import Path

# Reuse BigRedGym core training code without copying or modifying it.
import sys
CORE_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(CORE_SCRIPTS))
from train import make_train_parser, setup, train


def friction_name(lower: float) -> str:
    return f"{lower:.2f}".replace(".", "p")


def configure_friction(env_cfg, minimum: float, maximum: float, *, friction_only: bool) -> None:
    """Change only the requested experimental condition.

    Other DR axes can be disabled completely to isolate friction. No mutation
    of BigRedGym's version-controlled configuration source is required.
    """
    if not 0.0 <= minimum <= maximum:
        raise ValueError(f"Invalid friction range [{minimum}, {maximum}]")
    dr = env_cfg.domain_randomization
    dr.startup.contact_friction_range = [float(minimum), float(maximum)]
    if friction_only:
        dr.startup.link_mass_scale_range = None
        dr.episode.scale_ranges = {}


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
    parser.add_argument("--max-friction", type=float, default=1.25)

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
    if not 0 <= args.min_friction <= args.max_friction:
        parser.error("Required: 0 <= min-friction <= max-friction")
    if args.original_cfg:
        parser.error("Do not use --original_cfg with this experiment.")
    if args.resume and args.load_run is None:
        parser.error("Supply --load_run when resuming training.")
    if args.experiment_name is None:
        args.experiment_name = f"go2friction_lb_{friction_name(args.min_friction)}_seed_{args.seed}"

    # Task registration must occur before asking for the configuration.
    import gym.envs  # noqa: F401
    from gym.utils.task_registry import task_registry

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)

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


    print(
        f"Training {args.experiment_name}: "
        f"mu ~ U({args.min_friction}, {args.max_friction}), seed={args.seed}"
    )
    train_cfg, runner = setup(args, env_cfg=env_cfg, train_cfg=train_cfg)
    log_dir = Path(train_cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_name": args.experiment_name,
        "task": args.task,
        "seed": args.seed,
        "friction_range": [args.min_friction, args.max_friction],
        "friction_only": not args.keep_other_randomization,
        "gait_rewards_enabled": args.gait_rewards,
        "max_iterations": train_cfg.runner.max_iterations,
        "num_envs": env_cfg.env.num_envs,
        "device": args.device,
        "backend": args.backend,
        "train_pushes_enabled": bool(env_cfg.push_robots.toggle),
        "friction_resampling": "once per simulated environment at startup",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    path = log_dir / "friction_experiment.json"

    def save_manifest():
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

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
