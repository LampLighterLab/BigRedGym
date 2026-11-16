"""Sweep Go2Osc contact-phase tracking and survival over fixed frequencies.

Example:
    uv run scripts/evaluate_go2osc.py --replicas 32 --seconds 16

Outputs two PNG plots. Relative leg offsets
stay as trained; a common initial phase and robot states are randomized using
the training reset ranges. Each frequency uses the same seed for paired trials.
"""

import argparse
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch

from gym.envs.base.domain_randomization import apply_domain_randomization_override
from gym.utils.helpers import get_load_path, set_seed
from gym.utils.task_registry import task_registry


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_name", default="go2osc")
    parser.add_argument(
        "--load_run", help="Run directory (default: latest with models)"
    )
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--original_cfg", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("mujoco", "vsim"), default="mujoco")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--replicas", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=16.0)
    parser.add_argument("--warmup", type=float, default=0.0)
    parser.add_argument(
        "--frequencies", type=float, nargs="+", default=np.linspace(0, 5, 11).tolist()
    )
    parser.add_argument("--output", type=Path, default=Path("logs/go2osc/evaluation"))
    args = parser.parse_args()

    if args.replicas < 2:
        parser.error("--replicas must be at least 2")
    if not (math.isfinite(args.seconds) and 0 <= args.warmup < args.seconds):
        parser.error("require finite seconds and 0 <= warmup < seconds")
    if any(not math.isfinite(hz) or hz < 0 for hz in args.frequencies):
        parser.error("frequencies must be finite and nonnegative")
    args.frequencies = sorted(set(args.frequencies))
    return args


@torch.no_grad()
def evaluate_frequency(env, runner, frequency, seconds, warmup, seed):
    # Set the reset frequency too, because reset() advances physics once.
    env.cfg.osc.frequency = frequency
    set_seed(seed)
    env.reset()
    commands = env.commands.clone()
    alive = ~env.terminated.clone()
    steps = round(seconds / env.dt)
    warmup_steps = round(warmup / env.dt)
    measured_steps = steps - warmup_steps
    if measured_steps < 1:
        raise ValueError("Duration leaves no measured control steps")

    matches = torch.zeros_like(env.phase, dtype=torch.float64)

    for step in range(steps):
        env.commands[:] = commands
        runner.set_actions(
            runner.actor_cfg["actions"],
            runner.get_inference_actions(),
            runner.actor_cfg["disable_actions"],
        )
        env.step()
        finite = (
            torch.isfinite(env.root_states).all(dim=1)
            & torch.isfinite(env.dof_pos).all(dim=1)
            & torch.isfinite(env.measured_grf).all(dim=1)
        )
        failed = alive & (env.terminated | ~finite)
        alive &= ~failed

        if step < warmup_steps:
            continue
        # Use simultaneous post-step phase and force, exactly as the reward
        # does. At 0 Hz training asks for standing, irrespective of phase.
        stance = torch.sin(env.phase) < 0
        if frequency == 0:
            stance.fill_(True)
        contact = env.measured_grf > env.cfg.osc.contact_threshold
        valid = alive[:, None]
        matches += ((contact == stance) & valid).double()

    return {
        "frequency_hz": frequency,
        "measured_seconds": measured_steps * env.dt,
        "stable_fraction": alive.double().mean().item(),
        "mean_full_horizon_overlap": (matches / measured_steps).mean().item(),
        "overlap_by_trial": (matches / measured_steps).mean(dim=1).cpu().tolist(),
    }


def save_plots(cases, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hz = [case["frequency_hz"] for case in cases]
    confidence_level = 0.95
    tail_probability = (1 - confidence_level) / 2
    z = NormalDist().inv_cdf(1 - tail_probability)

    trial_means = np.asarray([case["overlap_by_trial"] for case in cases])
    fig, ax = plt.subplots(figsize=(9, 5))
    mean = trial_means.mean(axis=1)
    # Replicas, rather than correlated time samples or legs, are the units
    # of uncertainty. These are pointwise approximate 95% mean intervals.
    error = z * trial_means.std(axis=1, ddof=1) / np.sqrt(trial_means.shape[1])
    ax.plot(hz, mean * 100, "ko-", label="All legs")
    ax.fill_between(
        hz,
        np.maximum(0, mean - error) * 100,
        np.minimum(1, mean + error) * 100,
        color="black",
        alpha=0.12,
        label=f"{confidence_level:.0%} CI",
    )
    ax.set(
        xlabel="Oscillator frequency (Hz)",
        ylabel="Contact-phase overlap (%)",
        ylim=(0, 102),
        title=(
            "Go2Osc contact-phase tracking\n"
            "Full-horizon overlap; failures count as unmatched"
        ),
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "phase_tracking.png", dpi=160)
    plt.close(fig)

    stable = np.asarray([case["stable_fraction"] for case in cases])
    n = trial_means.shape[1]
    center = (stable + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(stable * (1 - stable) / n + z * z / (4 * n * n))
    half /= 1 + z * z / n
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(hz, stable * 100, "o-", label="Survived entire rollout")
    ax.fill_between(
        hz,
        (center - half) * 100,
        (center + half) * 100,
        alpha=0.2,
        label=f"{confidence_level:.0%} CI",
    )
    ax.set(
        xlabel="Oscillator frequency (Hz)",
        ylabel="Stable trials (%)",
        ylim=(0, 102),
        title="Go2Osc stability",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "stability.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    import gym.envs  # noqa: F401

    if args.load_run is None:
        root = Path(__file__).resolve().parents[1] / "logs" / args.experiment_name
        runs = [
            path
            for path in root.iterdir()
            if path.is_dir() and any(path.glob("model_*.pt"))
        ]
        if not runs:
            raise FileNotFoundError(f"No checkpoint runs found under {root}")
        args.load_run = max(runs, key=lambda path: path.stat().st_ctime).name

    env_cfg, train_cfg = task_registry.get_cfgs(
        "go2osc",
        original_cfg=args.original_cfg,
        experiment_name=args.experiment_name,
        load_run=args.load_run,
    )
    checkpoint = Path(
        get_load_path(
            name=args.experiment_name,
            load_run=args.load_run,
            checkpoint=args.checkpoint,
        )
    )
    env_cfg.env.num_envs = args.replicas
    env_cfg.env.episode_length_s = args.seconds + 1
    env_cfg.init_state.reset_mode = "reset_to_range"
    env_cfg.commands.resampling_time = args.seconds + 1
    env_cfg.osc.randomize_frequency = False
    env_cfg.push_robots.toggle = False
    apply_domain_randomization_override(env_cfg, "off")
    env_cfg.seed = args.seed
    train_cfg.runner.device = args.device
    train_cfg.runner.resume = False
    train_cfg.logging.enable_local_saving = False
    train_cfg.log_dir = None
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    set_seed(args.seed)
    env = task_registry.make_env("go2osc", env_cfg, args.device, True, args.backend)
    cases = []
    args.output.mkdir(parents=True, exist_ok=True)

    try:
        runner = task_registry.make_alg_runner(env, train_cfg)
        runner.load(checkpoint, load_optimizer=False)
        runner.switch_to_eval()
        for frequency in args.frequencies:
            case = evaluate_frequency(
                env, runner, frequency, args.seconds, args.warmup, args.seed
            )
            cases.append(case)
            print(
                f"{frequency:g} Hz: overlap "
                f"{case['mean_full_horizon_overlap']:.1%}, "
                f"stable {case['stable_fraction']:.1%}",
                flush=True,
            )
    finally:
        env._backend.close()

    save_plots(cases, args.output)
    print(f"Results saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
