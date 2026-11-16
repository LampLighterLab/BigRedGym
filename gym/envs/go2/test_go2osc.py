import math

import torch

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
        initial_phases = env.phase.clone()
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
            initial_phases + 2 * math.pi * cfg.osc.frequency * env.dt,
            2 * math.pi,
        )
        assert torch.allclose(env.phase, expected_next, atol=1e-5)

        for _ in range(49):
            env.step()

        assert env.phase.shape == (2, 4)
        assert torch.isfinite(env.phase).all()
        assert torch.isfinite(env.measured_grf).all()
        assert torch.all(env.phase_frequency == cfg.osc.frequency)

    finally:
        backend.close()


def test_gait_reward_depends_on_contact_timing_not_force_magnitude():
    env = object.__new__(Go2Osc)
    env.cfg = Go2OscCfg()
    env.phase = torch.tensor([[3 * math.pi / 2, math.pi / 2]])

    env.measured_grf = torch.tensor([[0.1, 0.0]])
    matched = env._reward_gait_contact()
    assert torch.allclose(matched, torch.tensor([0.5]))
    env.measured_grf = torch.tensor([[0.8, 0.0]])
    assert torch.allclose(env._reward_gait_contact(), matched)

    env.measured_grf = torch.tensor([[0.0, 0.8]])
    assert torch.all(env._reward_gait_contact() < matched)

    env.measured_grf = torch.zeros_like(env.measured_grf)
    assert torch.all(env._reward_gait_contact() < 0)


def test_upright_reward_rejects_inverted_pose():
    env = object.__new__(Go2Osc)
    env.projected_gravity = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(env._reward_orientation(), torch.tensor([2.0, -2.0]))


def test_only_requested_rewards_are_enabled():
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
    }
    assert reward_cfg["termination_weight"] == {"termination": 0.01}
