"""Plot repeated high/low height tracking for a Go2Trot checkpoint.

Example:
    uv run --frozen python scripts/evaluate_go2_height_cycles.py \
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
    "phase",
    "phase_time_s",
    "target_m",
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
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--hold_seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if not 0 < args.low < args.high:
        parser.error("heights must satisfy 0 < --low < --high")
    if args.cycles < 2 or args.hold_seconds <= 0:
        parser.error("--cycles must be at least 2 and --hold_seconds positive")
    if args.cycles * 2 * args.hold_seconds >= 50:
        parser.error("total duration must be under 50 seconds")
    return args


def collect(env, runner, args):
    env._reset_idx(torch.ones(1, dtype=torch.bool, device=env.device))
    env.commands[:, :3] = 0.0
    rows = []
    resets = 0
    step_count = 0
    hold_steps = max(1, round(args.hold_seconds / env.dt))
    for cycle in range(1, args.cycles + 1):
        for phase, target in (("high", args.high), ("low", args.low)):
            env.commands[:, 3] = target
            rows.append(
                {
                    "time_s": round(step_count * env.dt, 6),
                    "cycle": cycle,
                    "phase": phase,
                    "phase_time_s": 0.0,
                    "target_m": target,
                    "height_m": float(env.base_height[0, 0]),
                    "abs_error_m": abs(float(env.base_height[0, 0]) - target),
                    "terminated": 0,
                    "reset": 0,
                }
            )
            for phase_step in range(1, hold_steps + 1):
                runner.set_actions(
                    runner.actor_cfg["actions"],
                    runner.get_inference_actions(),
                    runner.actor_cfg["disable_actions"],
                )
                env.step()
                step_count += 1
                height = float(env.base_height[0, 0])
                terminated = bool(env.terminated[0] or env.timed_out[0])
                rows.append(
                    {
                        "time_s": round(step_count * env.dt, 6),
                        "cycle": cycle,
                        "phase": phase,
                        "phase_time_s": round(phase_step * env.dt, 6),
                        "target_m": target,
                        "height_m": height,
                        "abs_error_m": abs(height - target),
                        "terminated": int(terminated),
                        "reset": 0,
                    }
                )
                if terminated and not (
                    cycle == args.cycles and phase == "low" and phase_step == hold_steps
                ):
                    resets += 1
                    env._reset_idx(torch.ones(1, dtype=torch.bool, device=env.device))
                    env.commands[:, :3] = 0.0
                    env.commands[:, 3] = target
                    rows.append(
                        {
                            "time_s": round(step_count * env.dt, 6),
                            "cycle": cycle,
                            "phase": phase,
                            "phase_time_s": round(phase_step * env.dt, 6),
                            "target_m": target,
                            "height_m": float(env.base_height[0, 0]),
                            "abs_error_m": abs(float(env.base_height[0, 0]) - target),
                            "terminated": 0,
                            "reset": 1,
                        }
                    )
    return rows, resets


def summarize(rows, args):
    summary = []
    for cycle in range(1, args.cycles + 1):
        for phase in ("high", "low"):
            samples = [
                row
                for row in rows
                if row["cycle"] == cycle
                and row["phase"] == phase
                and row["phase_time_s"] >= min(1.0, args.hold_seconds / 2)
                and not row["reset"]
            ]
            summary.append(
                {
                    "cycle": cycle,
                    "phase": phase,
                    "target_m": args.high if phase == "high" else args.low,
                    "mean_height_m": float(np.mean([r["height_m"] for r in samples])),
                    "mean_abs_error_m": float(
                        np.mean([r["abs_error_m"] for r in samples])
                    ),
                    "termination_count": sum(
                        r["terminated"]
                        for r in rows
                        if r["cycle"] == cycle and r["phase"] == phase
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
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, height_ratios=[3, 1])
    axes[0].step(
        time,
        [r["target_m"] for r in rows],
        where="post",
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Target height",
    )
    axes[0].plot(
        time,
        [r["height_m"] for r in rows],
        color="tab:blue",
        linewidth=1.1,
        label="Measured base height",
    )
    axes[1].plot(time, [r["abs_error_m"] for r in rows], color="tab:orange")
    for row in rows:
        if row["reset"]:
            axes[0].axvline(row["time_s"], color="tab:red", alpha=0.6)
    axes[0].set_ylabel("Height (m)")
    axes[1].set_ylabel("Absolute error (m)")
    axes[1].set_xlabel("Time (s)")
    axes[0].legend(loc="best")
    axes[0].set_title(f"Go2Trot repeated height commands — {checkpoint}")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_cycles(rows, args, output):
    fig, ax = plt.subplots(figsize=(10, 5))
    hold = args.hold_seconds
    ax.step(
        [0, hold, 2 * hold],
        [args.high, args.low, args.low],
        where="post",
        color="black",
        linestyle="--",
        label="Target height",
    )
    for cycle in range(1, args.cycles + 1):
        samples = [r for r in rows if r["cycle"] == cycle]
        t0 = (cycle - 1) * 2 * hold
        ax.plot(
            [r["time_s"] - t0 for r in samples],
            [r["height_m"] for r in samples],
            linewidth=1.1,
            label=f"Cycle {cycle}",
        )
    ax.set(
        xlabel="Time within cycle (s)",
        ylabel="Measured base height (m)",
        title="Repeated high-to-low response",
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    checkpoint = Path(get_load_path("go2trot", args.load_run, args.checkpoint))
    output = args.output_dir or (
        checkpoint.parent / "evaluation" / checkpoint.stem / "height_cycles"
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
            rows, resets = collect(env, runner, args)
    finally:
        env._backend.close()
    summary = summarize(rows, args)
    write_csv(output / "samples.csv", rows, FIELDS)
    write_csv(output / "summary.csv", summary, list(summary[0]))
    plot_timeseries(rows, checkpoint.stem, output / "height_timeseries.png")
    plot_cycles(rows, args, output / "cycles_overlay.png")
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "backend": "mujoco_cpu",
                "seed": args.seed,
                "cycles": args.cycles,
                "high_target_m": args.high,
                "low_target_m": args.low,
                "hold_seconds": args.hold_seconds,
                "saved_training_height_range_m": training_range,
                "height_definition": "base origin world z",
                "termination_resets": resets,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved repeated height evaluation to {output}")


if __name__ == "__main__":
    main()
