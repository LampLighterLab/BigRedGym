"""Play back a policy trained with a supported Q2 physics backend.

Usage:
    uv run scripts/play.py --task pendulum
    uv run scripts/play.py --task mini_cheetah --num_envs 8
    uv run scripts/play.py --task mini_cheetah --load_run May08_12-34-56_ \
                           --checkpoint 1500

Supports MuJoCo CPU/Warp and optional VSim checkpoints.
"""

import argparse
from datetime import datetime
from pathlib import Path
import random

import torch

from gym.utils.helpers import set_seed
from gym.utils.task_registry import task_registry


def get_play_args(argv=None):
    parser = argparse.ArgumentParser(description="Play a policy trained with Q2")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Checkpoint root under logs/ (default: task config value)",
    )
    parser.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="Run directory name under logs/<experiment_name>/ (default: latest)",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=-1,
        help="Model iteration index (default: latest)",
    )
    parser.add_argument(
        "--original_cfg",
        action="store_true",
        help="Load environment and runner configs saved with the selected run.",
    )
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument(
        "--backend",
        type=str,
        default="mujoco",
        choices=["mujoco", "vsim"],
        help="Physics backend (vsim is CUDA-only; start via "
        "`uv run --env-file .env.vsim ...`)",
    )
    parser.add_argument(
        "--viewer_ui",
        action="store_true",
        default=False,
        help="show the MuJoCo viewer's side panels (hidden by default: their "
        "shortcuts bind most letters to visualisation toggles, which fire "
        "alongside keyboard teleop)",
    )
    parser.add_argument(
        "--keyboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keyboard teleop in the MuJoCo viewer (see "
        "gym/utils/interfaces/MujocoKeyboardInterface.py for key bindings). "
        "Use --no-keyboard to disable.",
    )
    return parser.parse_args(argv)


def setup(args):
    # Register tasks (imports the task classes).
    import gym.envs  # noqa: F401

    env_cfg, train_cfg = task_registry.get_cfgs(
        args.task,
        original_cfg=args.original_cfg,
        experiment_name=args.experiment_name,
        load_run=args.load_run,
    )

    # Play-time overrides — small batch, long episodes, no pushes.
    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.episode_length_s = 50
    if hasattr(env_cfg, "commands"):
        env_cfg.commands.resampling_time = 9999
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False
    env_cfg.init_state.reset_mode = "reset_to_basic"
    if args.task == "go2osc":
        env_cfg.osc.randomize_frequency = False
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.show_ui = args.viewer_ui

    if args.seed is not None:
        env_cfg.seed = args.seed
        train_cfg.seed = args.seed
    else:
        env_cfg.seed = random.randint(0, 10000)
        train_cfg.seed = env_cfg.seed

    train_cfg.runner.device = args.device
    train_cfg.runner.resume = True
    if args.experiment_name is not None:
        train_cfg.runner.experiment_name = args.experiment_name
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    train_cfg.logging.enable_local_saving = False

    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    task_registry.set_log_dir_name(train_cfg)
    set_seed(env_cfg.seed)

    env = task_registry.make_env(
        args.task,
        env_cfg,
        device=args.device,
        headless=args.headless,
        backend=args.backend,
    )

    runner = task_registry.make_alg_runner(env, train_cfg)
    runner.switch_to_eval()
    return env, runner


def save_grf_plot(samples, dt, foot_names, path):
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.arange(1, len(samples) + 1) * dt
    forces_by_foot = np.asarray([forces for forces, _ in samples]).T
    frequencies = np.asarray([frequency for _, frequency in samples])
    fig, axes = plt.subplots(
        len(foot_names) + 1,
        1,
        figsize=(12, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [2] * len(foot_names) + [1]},
    )
    for index, (foot, forces) in enumerate(zip(foot_names, forces_by_foot)):
        ax = axes[index]
        ax.scatter(times, forces, s=3, alpha=0.5, color=f"C{index}")
        ax.set(ylabel="Force (N)", title=foot)
        ax.grid(alpha=0.3)
    ax_frequency = axes[-1]
    ax_frequency.scatter(times, frequencies, s=3, color="black")
    ax_frequency.set(xlabel="Playback time (s)", ylabel="Frequency (Hz)")
    ax_frequency.grid(alpha=0.3)
    fig.suptitle("Go2 vertical foot contact forces and oscillator frequency")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Foot force and frequency plot saved to {path.resolve()}")


def play(env, runner, grf_plot_path=None):
    n_steps = 10 * int(env.max_episode_length)
    grf_samples = [] if grf_plot_path is not None else None
    try:
        for _ in range(n_steps):
            runner.set_actions(
                runner.actor_cfg["actions"],
                runner.get_inference_actions(),
                runner.actor_cfg["disable_actions"],
            )
            env.step()
            if grf_samples is not None:
                grf_samples.append((
                    env.contact_forces[0, env.feet_indices, 2]
                    .clamp_min(0.0)
                    .cpu()
                    .tolist(),
                    env.phase_frequency[0, 0].item(),
                ))
                viewer = getattr(env._backend, "_viewer", None)
                if viewer is not None and not viewer.is_running():
                    break
            env.check_exit()
    except KeyboardInterrupt:
        print("Playback interrupted; saving foot force and frequency plot.")
    finally:
        if grf_samples:
            save_grf_plot(
                grf_samples,
                env.dt,
                env.robot_layout.body_groups["feet"],
                grf_plot_path,
            )


if __name__ == "__main__":
    args = get_play_args()
    with torch.no_grad():
        env, runner = setup(args)
        # The MuJoCo warp (GPU) backend is headless-only: render() is a no-op and
        # it has no passive viewer, so nothing draws.  Point the user at cpu.
        if (
            args.backend == "mujoco"
            and not args.headless
            and not hasattr(env._backend, "_viewer_overlay_fn")
        ):
            print(
                "WARNING: the MuJoCo warp (GPU) backend has no interactive viewer "
                "— nothing will render. Use --device cpu for playback."
            )
        if args.keyboard and hasattr(env, "commands") and not args.headless:
            # Each viewer has its own input model: MuJoCo dispatches key
            # events via a callback, vlearn polls key state per frame.
            if args.backend == "vsim":
                from gym.utils.interfaces.VsimKeyboardInterface import (
                    VsimKeyboardInterface,
                )

                VsimKeyboardInterface(env)
            else:
                from gym.utils.interfaces.MujocoKeyboardInterface import (
                    MujocoKeyboardInterface,
                )

                MujocoKeyboardInterface(env)
        if hasattr(env, "commands") and not args.headless:
            # Same indicators, different drawing primitives per viewer.
            if args.backend == "vsim":
                from gym.utils.interfaces.VsimCommandVisualizer import (
                    VsimCommandVisualizer,
                )

                VsimCommandVisualizer(env)
            else:
                from gym.utils.interfaces.CommandVisualizer import CommandVisualizer

                CommandVisualizer(env)
        grf_plot_path = None
        if args.task == "go2osc":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            grf_plot_path = (
                Path(__file__).resolve().parents[1]
                / "logs"
                / "go2osc"
                / f"grf_playback_{timestamp}.png"
            )
        play(env, runner, grf_plot_path)
