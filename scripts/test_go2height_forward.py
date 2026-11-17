"""Quick headless check: walk forward while stepping the height command."""

import argparse
import copy
import json

import torch

from gym.envs.base.domain_randomization import apply_domain_randomization_override
from gym.utils.helpers import get_load_path, set_seed
from gym.utils.task_registry import task_registry


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load_run", default="Oct16_18-01-11_height_v1")
    parser.add_argument("--checkpoint", type=int, default=500)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--segment_seconds", type=float, default=2.0)
    args = parser.parse_args()

    import gym.envs  # noqa: F401

    env_cfg, train_cfg = copy.deepcopy(
        task_registry.get_cfgs(
            "go2height",
            original_cfg=True,
            load_run=args.load_run,
        )
    )
    total_seconds = 4 * args.segment_seconds
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = total_seconds + 1
    env_cfg.commands.resampling_time = total_seconds + 1
    env_cfg.push_robots.toggle = False
    env_cfg.init_state.reset_mode = "reset_to_basic"
    apply_domain_randomization_override(env_cfg, "off")
    env_cfg.seed = 123
    train_cfg.runner.device = "cpu"
    train_cfg.runner.resume = False
    train_cfg.logging.enable_local_saving = False
    train_cfg.log_dir = None
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    set_seed(123)

    path = get_load_path("go2height", args.load_run, args.checkpoint)
    env = task_registry.make_env("go2height", env_cfg, "cpu", True, "mujoco")
    try:
        runner = task_registry.make_alg_runner(env, train_cfg)
        runner.load(path, load_optimizer=False)
        runner.switch_to_eval()
        env.phase_frequency[:] = 2.0
        start_x = env.root_states[0, 0].item()
        heights = []
        failed = False
        targets = (0.38, 0.15, 0.38, 0.15)
        steps = round(args.segment_seconds / env.dt)
        for target in targets:
            samples = []
            for step in range(steps):
                env.commands[:] = torch.tensor([args.speed, 0.0, 0.0])
                env.base_height_command[:] = target
                actions = runner.get_inference_actions()
                runner.set_actions(runner.actor_cfg["actions"], actions)
                env.step()
                failed |= bool(env.terminated[0])
                if step >= steps // 2:
                    samples.append(env.base_height[0, 0].item())
            heights.append(sum(samples) / len(samples))
        distance = env.root_states[0, 0].item() - start_x
    finally:
        env._backend.close()

    high = (heights[0] + heights[2]) / 2
    low = (heights[1] + heights[3]) / 2
    mean_speed = distance / total_seconds
    height_change = high - low
    passed = not failed and distance >= 0.25 and height_change >= 0.15
    print(
        json.dumps(
            {
                "passed": passed,
                "survived": not failed,
                "forward_distance_m": round(distance, 3),
                "mean_forward_speed_m_s": round(mean_speed, 3),
                "high_height_m": round(high, 3),
                "low_height_m": round(low, 3),
                "height_change_m": round(height_change, 3),
            }
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
