"""Train one Go2trot policy at one friction range and seed.

Place in BigRedGym/experiments/go2_friction/ and launch with:
    uv run --frozen experiments/go2_friction/train_friction.py --task go2trot \
        --min-friction 0.6 --max-friction 1.25 --seed 42 \
        --max_iterations 200 --num_envs 256 --headless --disable_wandb

Reuses BigRedGym's *actual* scripts/train.py setup and policy runner.
"""
from __future__ import annotations

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


def main(argv=None) -> None:
    parser = make_train_parser()
    parser.add_argument("--min-friction", type=float, required=True)
    parser.add_argument("--max-friction", type=float, default=1.25)
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
    if args.resume or args.original_cfg:
        parser.error("This script creates fresh policies: don't use --resume/--original_cfg")
    if args.experiment_name is None:
        args.experiment_name = f"go2friction_lb_{friction_name(args.min_friction)}_seed_{args.seed}"

    # Task registration must occur before asking for the configuration.
    import gym.envs  # noqa: F401
    from gym.utils.task_registry import task_registry

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    configure_friction(
        env_cfg, args.min_friction, args.max_friction,
        friction_only=not args.keep_other_randomization,
    )
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
