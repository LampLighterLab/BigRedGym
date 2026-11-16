import math
from unittest.mock import patch

import torch

from gym.envs.go2.go2 import Go2
from gym.envs.go2.go2osc import Go2Osc
from gym.envs.go2.go2_config import Go2RunnerCfg
from gym.envs.go2.go2osc_config import Go2OscCfg, Go2OscRunnerCfg
from gym.utils.helpers import class_to_dict
from gym.utils.task_registry import task_registry, select_backend
from learning.utils import remove_zero_weighted_rewards


def test_oscillator_integration():

    cfg = Go2OscCfg()
    cfg.env.num_envs = 2
    cfg.seed = 1

    task_registry.convert_frequencies_to_params(cfg, Go2RunnerCfg())

    backend = select_backend(cfg, "cpu")

    env = Go2Osc(cfg, "cpu", True, backend)

    try:
        assert env.max_episode_length_s == 16
        assert env.max_episode_length == 800
        penalized_names = [
            backend.body_names[index]
            for index in env.penalised_contact_indices.tolist()
        ]
        assert penalized_names
        assert any("thigh" in name for name in penalized_names)
        assert any("calf" in name for name in penalized_names)
        assert all("thigh" in name or "calf" in name for name in penalized_names)
        assert not any("foot" in name for name in penalized_names)

        initial_phases = env.phase.clone()
        initial_frequency = env.phase_frequency.clone()
        assert torch.all(initial_frequency >= 0.0)
        assert torch.all(initial_frequency <= 5.0)
        assert torch.all(initial_frequency == initial_frequency[:, :1])
        expected_offsets = torch.tensor(
            [0.0, math.pi / 2, math.pi, 3 * math.pi / 2],
            device=env.device,
        ).unsqueeze(0).expand_as(env.phase)
        relative_phases = torch.remainder(
            initial_phases - initial_phases[:, :1], 2 * math.pi
        )
        assert torch.allclose(relative_phases, expected_offsets, atol=1e-5)

        env.step()
        expected_next = torch.remainder(
            initial_phases + 2 * math.pi * initial_frequency * env.dt,
            2 * math.pi,
        )
        assert torch.allclose(env.phase, expected_next, atol=1e-5)

        with patch.object(cfg.osc, "zero_frequency_fraction", 1.0):
            env.reset()
            assert torch.all(env.phase_frequency == 0.0)
            stopped_phase = env.phase.clone()
            env.step()
            assert torch.allclose(env.phase, stopped_phase)

        for _ in range(49):
            env.step()

        assert env.phase.shape == (2, 4)
        assert torch.isfinite(env.phase).all()
        assert torch.isfinite(env.measured_grf).all()
        assert torch.all(env.phase_frequency >= 0.0)
        assert torch.all(env.phase_frequency <= 5.0)

    finally:
        backend.close()


def test_gait_reward_depends_on_contact_timing_not_force_magnitude():
    env = object.__new__(Go2Osc)
    env.cfg = Go2OscCfg()
    env.phase = torch.tensor([[3 * math.pi / 2, math.pi / 2]])
    env.phase_frequency = torch.tensor([[2.0, 2.0]])

    env.measured_grf = torch.tensor([[0.1, 0.0]])
    matched = env._reward_gait_contact()
    assert torch.allclose(matched, torch.tensor([0.5]))
    env.measured_grf = torch.tensor([[0.8, 0.0]])
    assert torch.allclose(env._reward_gait_contact(), matched)

    env.measured_grf = torch.tensor([[0.0, 0.8]])
    assert torch.all(env._reward_gait_contact() < matched)

    env.measured_grf = torch.zeros_like(env.measured_grf)
    assert torch.all(env._reward_gait_contact() < 0)


def test_zero_frequency_rewards_four_foot_support():
    env = object.__new__(Go2Osc)
    env.cfg = Go2OscCfg()
    env.phase = torch.tensor([[0.0, math.pi / 2, math.pi, 3 * math.pi / 2]])
    env.phase_frequency = torch.zeros(1, 4)
    env.measured_grf = torch.full((1, 4), 0.1)
    assert torch.allclose(env._reward_gait_contact(), torch.tensor([0.5]))

    env.measured_grf[0, 0] = 0.0
    assert torch.allclose(env._reward_gait_contact(), torch.tensor([0.25]))


def test_frequency_changes_every_two_seconds_for_selected_training_envs():
    env = object.__new__(Go2Osc)
    env.cfg = Go2OscCfg()
    env.num_envs = 2
    env.device = "cpu"
    env.dt = 0.02
    env.max_episode_length = 800
    env.episode_length_buf = torch.tensor([99, 98])
    env.phase_frequency = torch.full((2, 4), 3.0)

    with patch.object(env.cfg.osc, "zero_frequency_fraction", 1.0):
        with patch.object(
            Go2,
            "_post_decimation_step",
            side_effect=lambda: env.episode_length_buf.add_(1),
        ):
            env._post_decimation_step()
    assert torch.all(env.phase_frequency[0] == 0.0)
    assert torch.all(env.phase_frequency[1] == 3.0)

    env.episode_length_buf[:] = 199
    with patch.object(env.cfg.osc, "randomize_frequency", False):
        with patch.object(
            Go2,
            "_post_decimation_step",
            side_effect=lambda: env.episode_length_buf.add_(1),
        ):
            env._post_decimation_step()
    assert torch.all(env.phase_frequency[0] == 0.0)
    assert torch.all(env.phase_frequency[1] == 3.0)


def test_upright_reward_rejects_inverted_pose():
    env = object.__new__(Go2Osc)
    env.projected_gravity = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(env._reward_orientation(), torch.tensor([2.0, -2.0]))


def test_low_body_height_terminates_only_affected_envs():
    env = object.__new__(Go2Osc)
    env.cfg = Go2OscCfg()
    env.root_states = torch.zeros(3, 13)
    env.root_states[:, 2] = torch.tensor([0.19, 0.20, 0.21])
    env.contact_forces = torch.zeros(3, 1, 3)
    env.termination_contact_indices = torch.tensor([0])
    env.episode_length_buf = torch.zeros(3, dtype=torch.long)
    env.max_episode_length = 400
    env.terminated = torch.zeros(3, dtype=torch.bool)
    env.timed_out = torch.zeros(3, dtype=torch.bool)

    env._check_terminations_and_timeouts()
    assert torch.equal(env.terminated, torch.tensor([True, False, False]))

    env.contact_forces[2, 0, 2] = 2.0
    env._check_terminations_and_timeouts()
    assert torch.equal(env.terminated, torch.tensor([True, False, True]))


def test_enabled_rewards_include_leg_collision_penalty():
    reward_cfg = class_to_dict(Go2OscRunnerCfg().critic.reward)
    weights = reward_cfg["weights"]
    remove_zero_weighted_rewards(weights)
    assert set(weights) == {
        "gait_contact",
        "torques",
        "orientation",
        "action_rate",
        "action_rate2",
        "ang_vel_xy",
        "collision",
    }
    assert reward_cfg["termination_weight"] == {"termination": 0.01}
