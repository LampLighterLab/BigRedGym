"""Switch coordination rewards between PPO rollouts on the same runner."""

from contextlib import contextmanager
from dataclasses import replace

import torch

from final_protocol import REWARD_TRANSITION_ITERATION, reward_schedule


def configure_stage(train_cfg, options, stage):
    for name, weight in stage["weights"].items():
        setattr(train_cfg.critic.reward.weights, name, weight)
    return replace(options, coordination_reward=stage["coordination_reward"])


@contextmanager
def staged_coordination_rewards(runner, train_cfg, on_transition):
    legacy, paired = reward_schedule()
    env = runner.env
    weights = runner.critic_cfg["reward"]["weights"]
    functions = {name: runner.reward_functions[name] for name in legacy["weights"]}
    zeros = torch.zeros(env.num_envs, device=runner.device)

    def inactive_reward():
        return zeros

    def activate(stage):
        env.controller_options = env.cfg._controller_options = configure_stage(
            train_cfg, env.controller_options, stage
        )
        weights.update(stage["weights"])
        for name, weight in stage["weights"].items():
            runner.reward_functions[name] = (
                functions[name] if weight else inactive_reward
            )

    # The core preallocates reward buffers and registers their names once.
    # Retain both stages' keys and emit exact zeros for inactive rewards.
    activate(legacy)
    save_checkpoint = runner.save

    def save_and_transition():
        save_checkpoint()
        if (
            runner.it == REWARD_TRANSITION_ITERATION
            and env.controller_options.coordination_reward
            == legacy["coordination_reward"]
        ):
            activate(paired)
            on_transition()

    runner.save = save_and_transition
    try:
        yield
    finally:
        runner.save = save_checkpoint
        runner.reward_functions.update(functions)
        for name in functions:
            if not weights[name]:
                del weights[name]
