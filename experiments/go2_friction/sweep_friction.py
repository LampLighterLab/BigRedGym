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
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LOWER_BOUNDS = (1.25, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2)
SEEDS = (42, 123, 456)


def parse_values(text: str, cast):
    values = tuple(cast(s.strip()) for s in text.split(",") if s.strip())
    if not values:
        raise ValueError("Provide at least one comma-separated value")
    return values


def experiment_name(lower: float, seed: int) -> str:
    return f"go2friction_lb_{lower:.2f}_seed_{seed}".replace(f"{lower:.2f}", f"{lower:.2f}".replace(".", "p"))


def command_for(repo: Path, lower: float, seed: int, args) -> list[str]:
    cmd = [
        sys.executable, str(repo / "experiments" / "go2_friction" / "train_friction.py"),
        "--task", "go2trot",
        "--min-friction", f"{lower:.6g}",
        "--max-friction", f"{args.upper:.6g}",
        "--seed", str(seed),
        "--max_iterations", str(args.max_iterations),
        "--num_envs", str(args.num_envs),
        "--backend", args.backend,
        "--device", args.device,
        "--experiment_name", experiment_name(lower, seed),
        "--headless",
    ]
    if not args.wandb:
        cmd.append("--disable_wandb")
    if args.keep_other_randomization:
        cmd.append("--keep-other-randomization")
    return cmd


def main(argv=None):
    parser = argparse.ArgumentParser(description="Go2 friction-lower-bound training sweep")
    parser.add_argument("--lower-bounds", default=",".join(map(str, LOWER_BOUNDS)))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--upper", type=float, default=1.25)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=["mujoco", "vsim"], default="mujoco")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--keep-other-randomization", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    lowers = parse_values(args.lower_bounds, float)
    seeds = parse_values(args.seeds, int)
    if args.max_iterations < 1 or args.num_envs < 1:
        parser.error("--max-iterations and --num-envs must be positive")
    if any(not 0 <= x <= args.upper for x in lowers):
        parser.error("Every lower bound must be between 0 and --upper")
    if len(set(lowers)) != len(lowers) or len(set(seeds)) != len(seeds):
        parser.error("Lower bounds and seeds should not contain duplicates")
    repo = Path(__file__).resolve().parents[2]
    if not (repo / "experiments" / "go2_friction" / "train_friction.py").exists():
        parser.error("Keep the project scripts in BigRedGym/experiments/go2_friction/")

    rows = []
    print(f"{len(lowers)} friction conditions x {len(seeds)} seeds = {len(lowers)*len(seeds)} runs")
    summary_path = Path(__file__).resolve().parent / "results" / "friction_sweep_runs.csv"
    if not args.dry_run:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
    for lower in lowers:
        for seed in seeds:
            cmd = command_for(repo, lower, seed, args)
            print("\n$ " + " ".join(cmd), flush=True)
            status = "dry_run" if args.dry_run else "running"
            exit_code = None
            if not args.dry_run:
                result = subprocess.run(cmd, cwd=repo, check=False)
                exit_code = result.returncode
                status = "completed" if exit_code == 0 else "failed"
            rows.append({
                "experiment_name": experiment_name(lower, seed),
                "lower_bound": lower, "upper_bound": args.upper,
                "seed": seed, "iterations": args.max_iterations,
                "num_envs": args.num_envs,
                "status": status, "returncode": exit_code,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })
            if not args.dry_run:
                with summary_path.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                if exit_code != 0 and not args.continue_on_error:
                    raise SystemExit(f"Training failed for {experiment_name(lower, seed)}; stopping sweep")
    print(f"\nSweep finished. {'Dry run only.' if args.dry_run else f'Manifest: {summary_path}'}")


if __name__ == "__main__":
    main()
