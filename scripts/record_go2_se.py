"""Record a Go2 state-estimator policy on ground of fixed friction.

Renders offscreen (no viewer / mjpython needed) and writes, per friction:
    <out>/go2_se_mu<mu>.gif
plus one plot of estimated vs true forward velocity for all frictions:
    <out>/velocity_estimate.png

Usage:
    python scripts/record_go2_se.py --load_run <run_dir> --mu 0.8 0.4 0.2 --vx 1.0
"""

import argparse
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from gym.utils.helpers import set_seed  # noqa: E402
from gym.utils.task_registry import task_registry  # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="go2_se")
    p.add_argument("--experiment_name", default=None)
    p.add_argument("--load_run", default=None, help="default: latest run")
    p.add_argument("--checkpoint", type=int, default=-1)
    p.add_argument("--mu", type=float, nargs="+", default=[0.8, 0.4, 0.2])
    p.add_argument("--vx", type=float, default=1.0, help="forward command [m/s]")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--out", default=None, help="default: <run dir>/videos")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def make(args, mu):
    import gym.envs  # noqa: F401

    env_cfg, train_cfg = task_registry.get_cfgs(
        args.task, experiment_name=args.experiment_name, load_run=args.load_run
    )
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = args.seconds + 1.0
    env_cfg.commands.resampling_time = 9999
    # * these exist only on the state-estimator configs
    if hasattr(env_cfg.commands, "zero_command_prob"):
        env_cfg.commands.zero_command_prob = 0.0
    if hasattr(env_cfg.commands, "curriculum"):
        env_cfg.commands.curriculum.toggle = False
    env_cfg.init_state.reset_mode = "reset_to_basic"
    if hasattr(env_cfg.init_state, "continue_prob"):
        env_cfg.init_state.continue_prob = 0.0
    env_cfg.push_robots.toggle = False
    # * fixed friction, nominal everything else
    env_cfg.domain_randomization.startup.contact_friction_range = [mu, mu]
    env_cfg.domain_randomization.startup.link_mass_scale_range = None
    env_cfg.domain_randomization.episode.scale_ranges = {
        "p_gains": [1.0, 1.0],
        "d_gains": [1.0, 1.0],
    }
    env_cfg.seed = train_cfg.seed = args.seed

    train_cfg.runner.device = "cpu"
    train_cfg.runner.resume = True
    if args.experiment_name is not None:
        train_cfg.runner.experiment_name = args.experiment_name
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    train_cfg.logging.enable_local_saving = False

    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    task_registry.set_log_dir_name(train_cfg)
    set_seed(args.seed)
    random.seed(args.seed)
    env = task_registry.make_env(
        args.task, env_cfg, device="cpu", headless=True, backend="mujoco"
    )
    runner = task_registry.make_alg_runner(env, train_cfg)
    runner.switch_to_eval()
    return env, runner


def set_command(env, vx):
    env.commands[:] = 0.0
    env.commands[:, 0] = vx


@torch.no_grad()
def rollout(args, mu):
    env, runner = make(args, mu)
    env._reset_idx(torch.ones(1, dtype=torch.bool))
    set_command(env, args.vx)

    backend = env._backend
    model, data = backend._mjm, backend._datas[0]
    renderer = mujoco.Renderer(model, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1
    cam.distance, cam.azimuth, cam.elevation = 1.6, 120.0, -15.0

    stride = max(1, round(1.0 / (args.fps * env.dt)))
    n_steps = int(args.seconds / env.dt)
    vel_scale = env.scales["base_lin_vel"]
    vel_scale = vel_scale[0] if torch.is_tensor(vel_scale) and vel_scale.ndim else vel_scale
    t, v_true, v_est, frames, fell_at = [], [], [], [], None
    for i in range(n_steps):
        runner.set_actions(
            runner.actor_cfg["actions"],
            runner.get_inference_actions(),
            runner.actor_cfg["disable_actions"],
        )
        env.step()
        if env.terminated[0] and fell_at is None:
            fell_at = i * env.dt
        runner.reset_envs()
        set_command(env, args.vx)

        t.append(i * env.dt)
        v_true.append(env.base_lin_vel[0, 0].item())
        if hasattr(runner, "get_estimate"):
            v_est.append((runner.get_estimate()[0, 0] * vel_scale).item())
        else:  # * plain policy, no estimator
            v_est.append(float("nan"))
        if i % stride == 0:
            renderer.update_scene(data, cam)
            frames.append(Image.fromarray(renderer.render()))
    renderer.close()
    return t, v_true, v_est, frames, fell_at, runner.log_dir, stride * env.dt


def main():
    args = get_args()
    results = {}
    for mu in args.mu:
        t, v_true, v_est, frames, fell_at, log_dir, frame_dt = rollout(args, mu)
        out = args.out or os.path.join(log_dir, "videos")
        os.makedirs(out, exist_ok=True)
        gif = os.path.join(out, f"go2_se_mu{mu}.gif")
        frames[0].save(
            gif,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 * frame_dt),
            loop=0,
        )
        err = (torch.tensor(v_est) - torch.tensor(v_true)).nan_to_num(0.0)
        steady = torch.tensor(t) > 1.0
        print(
            f"mu={mu}: fell={'no' if fell_at is None else f'at {fell_at:.2f}s'}  "
            f"mean vx={torch.tensor(v_true)[steady].mean():.2f} (cmd {args.vx})  "
            f"est err mean={err.abs().mean():.3f} peak={err.abs().max():.3f} m/s  "
            f"-> {gif}"
        )
        results[mu] = (t, v_true, v_est)

    fig, axes = plt.subplots(len(results), 1, figsize=(8, 2.5 * len(results)), sharex=True)
    axes = [axes] if len(results) == 1 else axes
    for ax, (mu, (t, v_true, v_est)) in zip(axes, results.items()):
        ax.plot(t, v_true, label="true", color="black")
        ax.plot(t, v_est, label="estimated", color="tab:orange")
        ax.axhline(args.vx, ls="--", color="gray", lw=0.8, label="command")
        ax.set_ylabel(f"vx [m/s], mu={mu}")
        ax.legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    png = os.path.join(out, "velocity_estimate.png")
    fig.savefig(png, dpi=120)
    print(f"plot -> {png}")


if __name__ == "__main__":
    main()
