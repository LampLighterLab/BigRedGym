"""Run the 7 friction ranges x 3 paired seeds in separate processes.

Dry-run first:
    uv run --frozen experiments/go2_friction/sweep_friction.py --dry-run

Actual training:
    uv run --frozen experiments/go2_friction/sweep_friction.py --max-iterations 200 \
        --num-envs 256 --device cpu

Each subprocess has an independent RNG and uniquely named checkpoint folder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import final_protocol as final

LOWER_BOUNDS = (1.25, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2)
REFERENCE_FRICTION = 1.25
SEEDS = (42, 123, 456)


def parse_values(text: str, cast):
    values = tuple(cast(s.strip()) for s in text.split(",") if s.strip())
    if not values:
        raise ValueError("Provide at least one comma-separated value")
    return values


def value_name(value: float) -> str:
    rounded = f"{value:.2f}"
    return (rounded if float(rounded) == value else str(value)).replace(".", "p")


def experiment_name(lower: float, seed: int, *, upper=REFERENCE_FRICTION) -> str:
    name = f"go2friction_lb_{value_name(lower)}"
    if upper != REFERENCE_FRICTION:
        name += f"_ub_{value_name(upper)}"
    return f"{name}_seed_{seed}"


def noise_experiment_name(scale: float, seed: int) -> str:
    return f"go2noise_scale_{value_name(scale)}_seed_{seed}"


def command_for(
    repo: Path, lower: float, seed: int, args, *, noise_scale=0.0, name=None
) -> list[str]:
    cmd = [
        sys.executable,
        str(repo / "experiments" / "go2_friction" / "train_friction.py"),
        "--task",
        "go2trot",
        "--min-friction",
        str(lower),
        "--max-friction",
        str(args.upper),
        "--noise-scale",
        str(noise_scale),
        "--seed",
        str(seed),
        "--max_iterations",
        str(args.max_iterations),
        "--num_envs",
        str(args.num_envs),
        "--backend",
        args.backend,
        "--device",
        args.device,
        "--experiment_name",
        name or experiment_name(lower, seed, upper=args.upper),
        "--headless",
    ]
    if not args.wandb:
        cmd.append("--disable_wandb")
    if args.keep_other_randomization:
        cmd.append("--keep-other-randomization")
    if getattr(args, "rollout_size", None) is not None:
        cmd.extend(("--rollout-size", str(args.rollout_size)))
    if getattr(args, "finalized_template", None) is not None:
        cmd.extend(
            (
                "--batch_size",
                str(final.BATCH_SIZE),
                "--save_interval",
                str(final.SAVE_INTERVAL),
                "--finalized-template",
                str(args.finalized_template),
                "--run_name",
                args.run_name,
            )
        )
    return cmd


def add_sweep_arguments(parser):
    parser.add_argument(
        "--seeds",
        help="Comma-separated training seeds (Step 1 default: 42,123,456)",
    )
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument(
        "--rollout-size", type=int, help="Override transitions per rollout"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=["mujoco", "vsim"], default="mujoco")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--keep-other-randomization", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--finalized-template",
        type=Path,
        help="Use the finalized matrix and the fresh Control checkpoint profile",
    )
    parser.add_argument(
        "--run-name", help="Shared output label for both finalized sweeps"
    )


def parse_seeds(parser, args):
    if args.run_name is not None:
        parser.error("--run-name requires --finalized-template")
    try:
        seeds = SEEDS if args.seeds is None else parse_values(args.seeds, int)
    except ValueError as exc:
        parser.error(f"Invalid --seeds: {exc}")
    if any(not 0 <= seed < 2**32 for seed in seeds) or len(set(seeds)) != len(seeds):
        parser.error("--seeds must contain unique integers in [0, 2**32 - 1]")
    if args.max_iterations is None:
        args.max_iterations = 200
    if args.num_envs is None:
        args.num_envs = 256
    if args.max_iterations < 1 or args.num_envs < 1:
        parser.error("--max-iterations and --num-envs must be positive")
    if args.rollout_size is not None and args.rollout_size < 1:
        parser.error("--rollout-size must be positive")
    return seeds


def remaining_staged_runs(logs_dir, run_name, template, runs):
    with template.open("rb") as stream:
        template_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    for run in runs:
        directory = logs_dir / run["experiment_name"] / run_name
        if not directory.exists():
            yield run
            continue
        # Import only when checking a trained policy. Planning alone needs no torch.
        from controller_checkpoint import read_controller_profile

        data = final.read_completed_run(directory, run)
        profile = read_controller_profile(
            directory / f"model_{final.ITERATIONS}.pt",
            expected_iteration=final.ITERATIONS,
        )
        if profile != data["controller_profile"] or (
            data["final_research"]["template_checkpoint_sha256"] != template_hash
        ):
            raise ValueError(f"Checkpoint or staged template differs: {directory}")
        print(f"Reusing completed staged run {directory}")


def run_sweep(args, runs, manifest_name):
    repo = Path(__file__).resolve().parents[2]
    summary_path = Path(__file__).resolve().parent / "results" / manifest_name
    staged = getattr(args, "finalized_template", None) is not None
    if staged:
        runs = tuple(
            remaining_staged_runs(
                repo / "logs", args.run_name, args.finalized_template, runs
            )
        )
        if not runs:
            print("All staged conditions in this sweep are complete.")
            return
    if not args.dry_run:
        summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for run in runs:
        name = run["experiment_name"]
        cmd = command_for(
            repo,
            run["lower_bound"],
            run["seed"],
            args,
            noise_scale=run["noise_scale"],
            name=name,
        )
        print("\n$ " + shlex.join(cmd), flush=True)
        if args.dry_run:
            continue

        result = subprocess.run(cmd, cwd=repo, check=False)
        rows.append(
            {
                **run,
                "upper_bound": args.upper,
                "iterations": args.max_iterations,
                "num_envs": args.num_envs,
                "rollout_size": args.rollout_size,
                "status": "completed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        with summary_path.open(
            "a" if staged else "w", newline="", encoding="utf-8"
        ) as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            if not staged or fh.tell() == 0:
                writer.writeheader()
            writer.writerows(rows[-1:] if staged else rows)
        if result.returncode != 0 and not args.continue_on_error:
            raise SystemExit(f"Training failed for {name}; stopping sweep")
    message = "Dry run only." if args.dry_run else f"Manifest: {summary_path}"
    print(f"\nSweep finished. {message}")


def run_final_sweep(parser, args, method, *, grid_overridden):
    try:
        final.validate_run_name(args.run_name)
        if (
            grid_overridden
            or args.seeds is not None
            or args.max_iterations is not None
            or args.num_envs is not None
            or args.rollout_size is not None
            or args.backend != "mujoco"
            or args.device != "cpu"
            or args.wandb
            or args.keep_other_randomization
        ):
            raise ValueError(
                "The finalized preset fixes the grid, seeds, and CPU budget"
            )
        args.finalized_template = args.finalized_template.resolve(strict=True)
        if not args.finalized_template.is_file():
            raise ValueError("--finalized-template must be a checkpoint file")
        args.upper = final.FRICTION
        args.max_iterations = final.ITERATIONS
        args.num_envs = final.NUM_ENVS
        args.rollout_size = final.ROLLOUT_SIZE
        runs = tuple(final.final_training_runs(method))
        if method == "noise":
            print("sweep_friction.py schedules the shared zero-noise baseline.")
        print(f"{len(runs)} finalized {method} runs")
        run_sweep(args, runs, f"final_staged_{method}_{args.run_name}_runs.csv")
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Go2 friction-lower-bound training sweep"
    )
    parser.add_argument("--lower-bounds")
    parser.add_argument("--upper", type=float)
    add_sweep_arguments(parser)
    args = parser.parse_args(argv)
    if args.finalized_template is not None:
        run_final_sweep(
            parser,
            args,
            "friction",
            grid_overridden=args.lower_bounds is not None or args.upper is not None,
        )
        return
    seeds = parse_seeds(parser, args)
    args.upper = REFERENCE_FRICTION if args.upper is None else args.upper
    try:
        lowers = (
            LOWER_BOUNDS
            if args.lower_bounds is None
            else parse_values(args.lower_bounds, float)
        )
    except ValueError as exc:
        parser.error(f"Invalid --lower-bounds: {exc}")
    if not math.isfinite(args.upper) or args.upper < 0:
        parser.error("--upper must be finite and nonnegative")
    if any(not 0 <= x <= args.upper for x in lowers):
        parser.error("Every lower bound must be between 0 and --upper")
    if len(set(lowers)) != len(lowers):
        parser.error("--lower-bounds must be unique")
    print(
        f"{len(lowers)} friction conditions x {len(seeds)} seeds "
        f"= {len(lowers) * len(seeds)} runs"
    )
    runs = (
        {
            "experiment_name": experiment_name(lower, seed, upper=args.upper),
            "lower_bound": lower,
            "noise_scale": 0.0,
            "seed": seed,
        }
        for lower in lowers
        for seed in seeds
    )
    run_sweep(args, runs, "friction_sweep_runs.csv")


if __name__ == "__main__":
    main()
