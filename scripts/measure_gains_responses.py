"""Evaluate the effect of changing P and D gains (stiffness and damping) on a policy
"""

import argparse
import random
import sys

import torch
import matplotlib.pyplot as plt

from gym.utils.helpers import set_seed
from gym.utils.task_registry import task_registry


def get_measure_args(argv=None):
    parser = argparse.ArgumentParser(description="Measure a Go2 policy's response to changing stiffness and damping")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min_kp", type=float, default=0.0)
    parser.add_argument("--max_kp", type=float, default=50.0)
    parser.add_argument("--min_kd", type=float, default=0.0)
    parser.add_argument("--max_kd", type=float, default=5.0)
    parser.add_argument("--num_kp", type=int, default=25)
    parser.add_argument("--num_kd", type=int, default=50)
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
    env_cfg.env.episode_length_s = 1  # changed from 50
    if hasattr(env_cfg, "commands"):
        env_cfg.commands.resampling_time = 9999
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False
    env_cfg.init_state.reset_mode = "reset_to_basic"
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


def reset_gains(env):
    # Reset the stiffness and damping (p_gains and d_gains) manually
    for actuator_index, name in enumerate(env.actuated_dof_names):
        matching_gains = [
            gain_name
            for gain_name in env.cfg.control.stiffness
            if gain_name in name
        ]
        if len(matching_gains) != 1:
            raise ValueError(
                f"expected exactly one PD gain match for {name!r}, "
                f"got {matching_gains}"
            )
        gain_name = matching_gains[0]
        env.p_gains[:, actuator_index] = env.cfg.control.stiffness[gain_name]
        env.d_gains[:, actuator_index] = env.cfg.control.damping[gain_name]


def play(env, runner):
    n_steps = 10 * int(env.max_episode_length)
    # Capture the starting positions of the envs
    start_root_states = env.root_states[:,:2].clone()
    for step in range(n_steps):
        runner.set_actions(
            runner.actor_cfg["actions"],
            runner.get_inference_actions(),
            runner.actor_cfg["disable_actions"],
        )
        env.step()
        if env.exit:
            break
    # Record the distance traveled and whether the robot is upright
    # Need to offset for initial position (not always 0,0,0)
    displacement = torch.sum(env.commands[:,:2] * (env.root_states[:,:2] - start_root_states), axis=1)  # num envs x 2 --> root planar position 2-vectors
    upright = torch.sum(env.gravity_vec * env.projected_gravity, axis=1)  # 1 if upright, 0 if sideways, -1 if flipped
    return displacement, upright


if __name__ == "__main__":
    print("Setup...")
    script_args = get_measure_args()

    stiffness = torch.linspace(script_args.min_kp, script_args.max_kp, script_args.num_kp)
    damping = torch.linspace(script_args.min_kd, script_args.max_kd, script_args.num_kd)

    class args:
        task = script_args.task
        device = script_args.device
        num_envs = script_args.num_envs
        seed = script_args.seed
        experiment_name = script_args.experiment_name
        load_run = script_args.load_run
        checkpoint = script_args.checkpoint
        original_cfg = script_args.original_cfg
        headless = True
        backend = script_args.backend
        viewer_ui = False
        keyboard = False

    print("Beginning sweep...")
    # Namespace(task='go2trot', device='cpu', num_envs=1, seed=None, experiment_name=None, load_run=None, checkpoint=-1, original_cfg=False, headless=False, backend='mujoco', viewer_ui=False, keyboard=True)
    displacements = torch.zeros((stiffness.size()[0], damping.size()[0]))
    uprights = torch.zeros((stiffness.size()[0], damping.size()[0]))
    with torch.no_grad():
        env, runner = setup(args)
        # Save the original stiffness and damping values to plot
        # This assumes that stiffness and damping is uniform across all joints
        orig_stiffness = env.cfg.control.stiffness["hip"]
        orig_damping = env.cfg.control.damping["hip"]
        for p_idx in range(stiffness.size()[0]):
            for d_idx in range(damping.size()[0]):
                print(f"Percent: {100 * (p_idx * damping.size()[0] + d_idx) / (stiffness.size()[0] * damping.size()[0]) :.1f}% (Testing k={stiffness[p_idx]}, b={damping[d_idx]})")
                env.reset()
                # Start with forward command
                env.commands[:] = 0.0
                env.commands[:, 0] = 1.0
                # Reset the stiffness and damping
                env.cfg.control.stiffness = {"hip": stiffness[p_idx], "thigh": stiffness[p_idx], "calf": stiffness[p_idx]}
                env.cfg.control.damping = {"hip": damping[d_idx], "thigh": damping[d_idx], "calf": damping[d_idx]}
                reset_gains(env)
                displacement, upright = play(env, runner)
                displacements[p_idx, d_idx] = torch.mean(displacement)
                uprights[p_idx, d_idx] = torch.mean(upright)

    # Display the arrays as an image
    plt.imshow(torch.flip(displacements, [0]), cmap='viridis', interpolation='nearest', aspect='auto', extent =[torch.min(damping).item(), torch.max(damping).item(), torch.min(stiffness).item(), torch.max(stiffness).item()])
    cbar = plt.colorbar()
    cbar.set_label("Bigger positive = follows commands better", rotation=270)
    plt.plot(orig_damping, orig_stiffness, 'o', markerfacecolor='none', markeredgecolor='red', label="Trained")
    plt.xlabel("Damping (derivative gain)")
    plt.ylabel("Stiffness (proportional gain)")
    plt.title(f"Displacement products\nEnvs: {script_args.num_envs}; Sim / Control Freq Ratio: {env.cfg.control.desired_sim_frequency / env.cfg.control.ctrl_frequency}")
    plt.legend()
    # plt.show()
    plt.savefig(f"displacement.png")

    plt.imshow(torch.flip(uprights, [0]), cmap='magma', interpolation='nearest', aspect='auto', extent =[torch.min(damping).item(), torch.max(damping).item(), torch.min(stiffness).item(), torch.max(stiffness).item()])
    cbar = plt.colorbar()
    cbar.set_label("1 = upright, 0 = sideways, -1 = upside-down", rotation=270)
    plt.plot(orig_damping, orig_stiffness, 'o', markerfacecolor='none', markeredgecolor='red', label="Trained")
    plt.xlabel("Damping (derivative gain)")
    plt.ylabel("Stiffness (proportional gain)")
    plt.title(f"Upright products\nEnvs: {script_args.num_envs}; Sim / Control Freq Ratio: {env.cfg.control.desired_sim_frequency / env.cfg.control.ctrl_frequency}")
    plt.legend()
    # plt.show()
    plt.savefig(f"upright.png")

    # On cpu with stiffness 0:5:50, damping 0:0.1:1, desired frequency 150:
    # 5 second episodes, 1 env: 16.3 seconds per 1%
    # 5 second episodes, 2 envs: 23.5 seconds per 1%
    # 5 second episodes, 4 envs: 30 seconds per 1%
    # 5 second episodes, 8 envs: 54 seconds per 1%
    # --> Time is roughly linear with the number of envs, seconds per % = 5.254 envs + 11.24783
    # 1 second episodes, 4 env: 6.4 seconds per 1%
    # 1 second episodes, 8 env: 11 seconds per 1%

