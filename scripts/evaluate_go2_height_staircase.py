"""Evaluate repeated stepwise Go2Trot height commands with a saved policy.

Example:
    uv run --frozen python scripts/evaluate_go2_height_staircase.py \
        --load_run Oct15_18-26-00_best_ --checkpoint 1000
"""

import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bigredgym-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from gym.utils.helpers import get_load_path
from play import setup


FIELDS = [
    "time_s",
    "cycle",
    "direction",
    "target_m",
    "hold_time_s",
    "height_m",
    "abs_error_m",
    "terminated",
    "reset",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load_run", default="Oct15_18-26-00_best_")
    parser.add_argument("--checkpoint", type=int, default=1000)
    parser.add_argument("--high", type=float, default=0.5)
    parser.add_argument("--low", type=float, default=0.15)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--hold_seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if not 0 < args.low < args.high or args.step <= 0:
        parser.error("heights must satisfy 0 < --low < --high and --step > 0")
    intervals = (args.high - args.low) / args.step
    if abs(intervals - round(intervals)) > 1e-6:
        parser.error("--step must divide the difference between --high and --low")
    if args.cycles < 2 or args.hold_seconds <= 0:
        parser.error("--cycles must be at least 2 and --hold_seconds positive")
    if args.cycles * (2 * round(intervals) + 1) * args.hold_seconds >= 50:
        parser.error("total duration must be under 50 seconds")
    return args


def targets(args):
    intervals = round((args.high - args.low) / args.step)
    down = [round(args.high - i * args.step, 6) for i in range(intervals + 1)]
    return [("down", h) for h in down] + [("up", h) for h in down[-2::-1]]


def collect(env, runner, args, staircase):
    env._reset_idx(torch.ones(1, dtype=torch.bool, device=env.device))
    env.commands[:, :3] = 0.0
    rows = []
    resets = 0
    step_count = 0
    hold_steps = max(1, round(args.hold_seconds / env.dt))

    def record(cycle, direction, target, hold_step, terminated=0, reset=0):
        height = float(env.base_height[0, 0])
        rows.append(
            {
                "time_s": round(step_count * env.dt, 6),
                "cycle": cycle,
                "direction": direction,
                "target_m": target,
                "hold_time_s": round(hold_step * env.dt, 6),
                "height_m": height,
                "abs_error_m": abs(height - target),
                "terminated": terminated,
                "reset": reset,
            }
        )

    total_holds = args.cycles * len(staircase)
    for hold_index in range(total_holds):
        cycle = hold_index // len(staircase) + 1
        direction, target = staircase[hold_index % len(staircase)]
        env.commands[:, 3] = target
        record(cycle, direction, target, 0)
        for hold_step in range(1, hold_steps + 1):
            runner.set_actions(
                runner.actor_cfg["actions"],
                runner.get_inference_actions(),
                runner.actor_cfg["disable_actions"],
            )
            env.step()
            step_count += 1
            terminated = bool(env.terminated[0] or env.timed_out[0])
            record(cycle, direction, target, hold_step, int(terminated))
            if terminated and not (
                hold_index == total_holds - 1 and hold_step == hold_steps
            ):
                resets += 1
                env._reset_idx(torch.ones(1, dtype=torch.bool, device=env.device))
                env.commands[:, :3] = 0.0
                env.commands[:, 3] = target
                record(cycle, direction, target, hold_step, reset=1)
    return rows, resets


def summarize(rows, staircase, args):
    summary = []
    for direction, target in staircase:
        samples = [
            row
            for row in rows
            if row["direction"] == direction
            and row["target_m"] == target
            and row["hold_time_s"] >= min(0.3, args.hold_seconds / 2)
            and not row["reset"]
        ]
        summary.append(
            {
                "direction": direction,
                "target_m": target,
                "mean_height_m": float(np.mean([r["height_m"] for r in samples])),
                "mean_abs_error_m": float(np.mean([r["abs_error_m"] for r in samples])),
                "termination_count": sum(
                    r["terminated"]
                    for r in rows
                    if r["direction"] == direction and r["target_m"] == target
                ),
            }
        )
    return summary


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_timeseries(rows, checkpoint, output):
    time = [r["time_s"] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True, height_ratios=[3, 1])
    axes[0].step(
        time,
        [r["target_m"] for r in rows],
        where="post",
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="Target height",
    )
    axes[0].plot(
        time,
        [r["height_m"] for r in rows],
        color="tab:blue",
        linewidth=1.0,
        label="Measured base height",
    )
    axes[1].plot(time, [r["abs_error_m"] for r in rows], color="tab:orange")
    for row in rows:
        if row["reset"]:
            axes[0].axvline(row["time_s"], color="tab:red", alpha=0.6)
    axes[0].set_ylabel("Height (m)")
    axes[1].set_ylabel("Absolute error (m)")
    axes[1].set_xlabel("Time (s)")
    axes[0].set_title(f"Go2Trot repeated height staircase — {checkpoint}")
    axes[0].legend(loc="best")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_performance(summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    targets = sorted({r["target_m"] for r in summary})
    axes[0].plot(targets, targets, "k--", linewidth=1, label="Target")
    for direction in ("down", "up"):
        samples = sorted(
            (r for r in summary if r["direction"] == direction),
            key=lambda r: r["target_m"],
        )
        axes[0].plot(
            [r["target_m"] for r in samples],
            [r["mean_height_m"] for r in samples],
            marker="o",
            label=direction.capitalize(),
        )
        axes[1].plot(
            [r["target_m"] for r in samples],
            [r["mean_abs_error_m"] for r in samples],
            marker="o",
            label=direction.capitalize(),
        )
    axes[0].set_ylabel("Mean measured height (m)")
    axes[1].set_ylabel("Mean absolute error (m)")
    for ax in axes:
        ax.set_xlabel("Target height (m)")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    staircase = targets(args)
    checkpoint = Path(get_load_path("go2trot", args.load_run, args.checkpoint))
    output = args.output_dir or (
        checkpoint.parent / "evaluation" / checkpoint.stem / "height_staircase"
    )
    output.mkdir(parents=True, exist_ok=True)
    play_args = SimpleNamespace(
        task="go2trot",
        device="cpu",
        num_envs=1,
        seed=args.seed,
        experiment_name="go2trot",
        load_run=args.load_run,
        checkpoint=args.checkpoint,
        original_cfg=True,
        headless=True,
        backend="mujoco",
        viewer_ui=False,
        keyboard=False,
    )
    env, runner = setup(play_args)
    try:
        training_range = list(env.cfg.commands.ranges.base_height)
        with torch.inference_mode():
            rows, resets = collect(env, runner, args, staircase)
    finally:
        env._backend.close()
    summary = summarize(rows, staircase, args)
    write_csv(output / "samples.csv", rows, FIELDS)
    write_csv(output / "summary.csv", summary, list(summary[0]))
    plot_timeseries(rows, checkpoint.stem, output / "height_timeseries.png")
    plot_performance(summary, output / "height_performance.png")
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "backend": "mujoco_cpu",
                "seed": args.seed,
                "cycles": args.cycles,
                "targets_per_cycle_m": [target for _, target in staircase],
                "hold_seconds": args.hold_seconds,
                "saved_training_height_range_m": training_range,
                "height_definition": "base origin world z",
                "termination_resets": resets,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved staircase evaluation to {output}")


if __name__ == "__main__":
    main()
