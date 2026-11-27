"""Record command-conditioned training metrics without changing PPO rewards."""

from contextlib import contextmanager
import csv
import math
from pathlib import Path

import torch

from controller_telemetry import (
    COUNT_FIELDS,
    RATE_FIELDS,
    ControllerTelemetry,
    controller_counts,
    controller_rates,
)


SPEED_EDGES = (-0.25, 0.25, 0.75, 1.25, 1.75)
SPEED_BINS = ("backward", "near_zero", "0p5", "1p0", "1p5", "2p0")
COMMAND_BINS = ("standing",) + tuple(
    f"{kind}_{speed}" for kind in ("straight", "mixed") for speed in SPEED_BINS
)
MIN_STANCE_FORCE_N = 5.0


def command_bin_indices(commands, edges):
    speed = torch.bucketize(commands[:, 0].contiguous(), edges, right=True)
    mixed = (commands[:, 1:] != 0).any(dim=1)
    bins = 1 + speed + mixed.to(torch.long) * len(SPEED_BINS)
    return bins.masked_fill((commands == 0).all(dim=1), 0)


class CommandMetrics:
    def __init__(self, runner, output):
        self.env = runner.env
        reward_cfg = runner.critic_cfg["reward"]
        self.rewards = sorted(
            reward_cfg["weights"].keys() | reward_cfg["termination_weight"].keys()
        )
        feet = self.env.robot_layout.body_groups["feet"]
        self.stance_fields = [f"stance_duty_{foot}" for foot in feet]
        self.state_fields = [
            "mean_command_vx_m_s",
            "tracking_rmse_m_s",
            "mean_forward_velocity_m_s",
            "mean_base_height_m",
            *self.stance_fields,
        ]
        self.reward_fields = [f"reward_{name}" for name in self.rewards]
        self.controller = (
            ControllerTelemetry(self.env)
            if hasattr(self.env, "controller_options")
            else None
        )
        self.controller_counts = COUNT_FIELDS if self.controller is not None else ()
        self.controller_rates = RATE_FIELDS if self.controller is not None else ()
        self.writer = csv.DictWriter(
            output,
            fieldnames=[
                "iteration",
                "policy_iteration",
                "rollout_complete",
                "command_bin",
                "samples",
                "reward_samples",
                *self.state_fields,
                *self.reward_fields,
                "reward_total",
                *self.controller_counts,
                *self.controller_rates,
            ],
        )
        self.writer.writeheader()
        self.output = output
        self.edges = self.env.commands.new_tensor(SPEED_EDGES)
        self.sums = torch.zeros(
            len(COMMAND_BINS),
            2
            + len(self.state_fields)
            + len(self.rewards)
            + len(self.controller_counts),
            dtype=torch.float64,
            device=self.env.commands.device,
        )
        self.expected_samples = runner.num_steps_per_env * self.env.num_envs
        self.iteration = None
        self.samples = 0

    def record(self, iteration, command, rewards):
        if self.iteration is not None and iteration != self.iteration:
            self.flush()
        self.iteration = iteration
        velocity = self.env.base_lin_vel[:, :2]
        error = (velocity - command[:, :2]).square().sum(dim=1)
        counts = ()
        if self.controller is None:
            stance = (
                self.env.contact_forces[:, self.env.feet_indices, 2]
                > MIN_STANCE_FORCE_N
            )
        else:
            sample = self.controller.sample(command)
            stance = sample["loaded_foot_contact"]
            counts = controller_counts(sample, self.env.terminated)
        same_command = (command == self.env.commands).all(dim=1)
        # A resampling boundary can pair the action's old command with the
        # reward's new command. Use those steps for kinematics only.
        terms = [
            torch.where(same_command, rewards[name] / self.env.dt, 0.0)
            for name in self.rewards
        ]
        values = torch.column_stack(
            (
                torch.ones_like(error),
                same_command,
                command[:, 0],
                error,
                velocity[:, 0],
                self.env.base_height.reshape(-1),
                stance,
                *terms,
                *counts,
            )
        )
        self.sums.index_add_(
            0, command_bin_indices(command, self.edges), values.to(self.sums)
        )
        self.samples += len(command)

    def flush(self):
        if not self.samples:
            return
        for name, values in zip(COMMAND_BINS, self.sums.tolist(), strict=True):
            n, n_reward, *sums = values
            row = {
                "iteration": self.iteration,
                "policy_iteration": self.iteration - 1,
                "rollout_complete": self.samples == self.expected_samples,
                "command_bin": name,
                "samples": int(n),
                "reward_samples": int(n_reward),
            }
            for field, total in zip(
                self.state_fields, sums[: len(self.state_fields)], strict=True
            ):
                mean = total / n if n else None
                if field == "tracking_rmse_m_s" and mean is not None:
                    mean = math.sqrt(mean)
                row[field] = mean
            rewards_end = len(self.state_fields) + len(self.reward_fields)
            reward_totals = sums[len(self.state_fields) : rewards_end]
            for field, total in zip(self.reward_fields, reward_totals, strict=True):
                row[field] = total / n_reward if n_reward else None
            row["reward_total"] = sum(reward_totals) / n_reward if n_reward else None
            if self.controller is not None:
                counts = tuple(map(int, sums[rewards_end:]))
                row.update(zip(self.controller_counts, counts, strict=True))
                row.update(
                    zip(self.controller_rates, controller_rates(counts, n), strict=True)
                )
            self.writer.writerow(row)
        self.output.flush()
        self.sums.zero_()
        self.samples = 0


@contextmanager
def log_command_metrics(runner, path):
    env = runner.env
    if int((1 / env.dt) / runner.actor_cfg["frequency"]) != 1:
        raise ValueError("Command metrics require one control step per actor step")
    step = env.step
    update_rewards = runner.update_rewards_dict
    command = None

    with Path(path).open("x", newline="", encoding="utf-8") as output:
        recorder = CommandMetrics(runner, output)

        def step_with_command():
            nonlocal command
            command = env.commands.clone()
            return step()

        def update_and_record(rewards_dict, step_index):
            result = update_rewards(rewards_dict, step_index)
            recorder.record(runner.it, command, rewards_dict[step_index])
            return result

        env.step = step_with_command
        runner.update_rewards_dict = update_and_record
        try:
            yield
        finally:
            env.step = step
            runner.update_rewards_dict = update_rewards
            recorder.flush()
