"""Evaluate every saved checkpoint from the 7x3 sweep, sequentially.

Example:
    uv run --frozen experiments/go2_friction/evaluate_all_friction.py --dry-run
    uv run --frozen experiments/go2_friction/evaluate_all_friction.py --trials 10 --device cpu

Each policy is evaluated independently at identical fixed-friction conditions.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from sweep_friction import LOWER_BOUNDS, SEEDS, experiment_name, parse_values


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate saved friction sweep policies")
    parser.add_argument("--lower-bounds", default=",".join(map(str, LOWER_BOUNDS)))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--frictions", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.25")
    parser.add_argument("--maneuvers", default="steady_0p5,steady_1p0,steady_1p5,steady_2p0,stand_2_stop,forward_backward,forward_sideways")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    for mu in parse_values(args.lower_bounds, float):
        for seed in parse_values(args.seeds, int):
            exp = experiment_name(mu, seed)
            cmd = [sys.executable, str(repo / "experiments/go2_friction/evaluate_friction.py"),
                   "--experiment-name", exp,
                   "--trials", str(args.trials),
                   "--frictions", args.frictions,
                   "--maneuvers", args.maneuvers,
                   "--device", args.device]
            print("$ " + " ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=repo, check=True)


if __name__ == "__main__":
    main()
