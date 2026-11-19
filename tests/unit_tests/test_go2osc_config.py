"""Exercise the production oscillator configuration through MuJoCo and PPO."""

import math

import pytest
import torch
from tensordict import TensorDict

from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg
from gym.envs.go2.go2osc import Go2Osc
from gym.envs.go2.go2osc_config import Go2OscCfg, Go2OscRunnerCfg
from gym.utils.helpers import class_to_dict
from gym.utils.task_registry import select_backend, task_registry
from learning.runners.on_policy_runner import OnPolicyRunner


@pytest.fixture
def configured_task():
    cfg = Go2OscCfg()
    cfg.env.num_envs = 4
    cfg.seed = 17  # The training entry point normally resolves this before setup.
    train_cfg = Go2OscRunnerCfg()
    task_registry.convert_frequencies_to_params(cfg, train_cfg)
    backend = select_backend(cfg, "cpu", "mujoco")
    try:
        yield Go2Osc(cfg, "cpu", True, backend), train_cfg
    finally:
        backend.close()


def _make_runner(env, train_cfg, *, small_rollout=False):
    train_dict = class_to_dict(train_cfg)
    train_dict["log_dir"] = None
    if small_rollout:
        train_dict["algorithm"].update(
            rollout_size=16, batch_size=16, max_gradient_steps=1
        )
    return OnPolicyRunner(env, train_dict, device="cpu")


def test_oscillator_config_does_not_change_ordinary_go2_defaults():
    original_env = class_to_dict(Go2Cfg())
    original_train = class_to_dict(Go2RunnerCfg())
    cfg = Go2OscCfg()
    train_cfg = Go2OscRunnerCfg()
    cfg.env.num_envs = 4

    task_registry.convert_frequencies_to_params(cfg, train_cfg)

    assert class_to_dict(Go2Cfg()) == original_env
    assert class_to_dict(Go2RunnerCfg()) == original_train
    assert cfg.control.decimation == 5
    assert cfg.sim_dt == pytest.approx(0.002)
    assert cfg.control.ctrl_dt == pytest.approx(0.01)


def test_configured_observations_separate_privileged_state_and_run_networks(
    configured_task,
):
    env, train_cfg = configured_task
    runner = _make_runner(env, train_cfg)
    actor_obs = runner.get_obs(runner.actor_cfg["obs"])
    critic_obs = runner.get_obs(runner.critic_cfg["obs"])

    assert actor_obs.shape == (env.num_envs, 53)
    assert critic_obs.shape == (env.num_envs, 61)
    privileged = {"base_height", "base_lin_vel", "oscillators_vel"}
    assert privileged.isdisjoint(runner.actor_cfg["obs"])
    assert privileged.issubset(runner.critic_cfg["obs"])
    assert "oscillator_obs" in runner.actor_cfg["obs"]
    assert "oscillator_obs" in runner.critic_cfg["obs"]
    with torch.no_grad():
        actions = runner.alg.actor.act_inference(actor_obs)
        values = runner.alg.critic.evaluate(critic_obs)
    assert actions.shape == (env.num_envs, env.num_actuators)
    assert values.shape == (env.num_envs,)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(values).all()


def test_configured_rewards_resolve_without_a_prescribed_gait(configured_task):
    env, train_cfg = configured_task
    runner = _make_runner(env, train_cfg)
    weights = runner.critic_cfg["reward"]["weights"]
    assert weights["swing_grf"] == weights["stance_grf"] > 0
    assert weights["cursorial"] > 0
    assert {"trot", "pace", "bound", "pronk", "enc_pace"}.isdisjoint(weights)

    env.step()
    rewards = runner.get_rewards(weights)
    rewards.update(
        runner.get_rewards(runner.critic_cfg["reward"]["termination_weight"])
    )
    for reward in rewards.values():
        assert reward.shape == (env.num_envs,)
        assert torch.isfinite(reward).all()


def test_production_schedule_handles_stopping_forward_and_reverse(configured_task):
    env, _ = configured_task
    env.cfg.osc.omega_var = 0.0
    env.cfg.osc.coupling_var = 0.0
    env.cfg.osc.process_noise_std = 0.0
    env.commands[:, 0] = torch.tensor([0.0, 0.5, 1.0, -3.0])

    env._update_oscillator_parameters()

    torch.testing.assert_close(env.osc_omega[:, 0], torch.tensor([1.0, 1.0, 2.5, 4.0]))
    torch.testing.assert_close(
        env.osc_coupling[:, 0], torch.tensor([4.0, 4.0, 1.0, 1.0])
    )
    torch.testing.assert_close(env.osc_offset[:, 0], torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_configured_resets_preserve_unselected_observations(configured_task):
    env, train_cfg = configured_task
    observations = train_cfg.critic.obs
    before = env.get_states(observations).clone()
    before_phases = env.oscillators.clone()
    mask = torch.tensor([True, False, True, False])

    env._reset_idx(mask)

    after = env.get_states(observations).clone()
    torch.testing.assert_close(after[~mask], before[~mask], rtol=0, atol=0)
    assert not torch.equal(env.oscillators[mask], before_phases[mask])
    assert torch.count_nonzero(env.grf[mask]) == 0
    env._reset_idx(torch.zeros_like(mask))
    torch.testing.assert_close(env.get_states(observations), after, rtol=0, atol=0)


def test_configured_runner_can_collect_rollout_and_update_ppo(configured_task):
    env, train_cfg = configured_task
    runner = _make_runner(env, train_cfg, small_rollout=True)
    actor_before = [
        parameter.detach().clone() for parameter in runner.alg.actor.parameters()
    ]
    critic_before = [
        parameter.detach().clone() for parameter in runner.alg.critic.parameters()
    ]
    transitions = []
    with torch.no_grad():
        actor_obs = runner.get_obs(runner.actor_cfg["obs"])
        critic_obs = runner.get_obs(runner.critic_cfg["obs"])
        for _ in range(runner.num_steps_per_env):
            actions = runner.alg.act(actor_obs)
            runner.set_actions(runner.actor_cfg["actions"], actions)
            env.step()
            rewards = runner.initialize_rewards_dict(1)
            runner.update_rewards_dict(rewards, 0)
            total_rewards = torch.stack(tuple(rewards[0].values())).sum(dim=0)
            timed_out = env.timed_out.clone()
            terminated = env.terminated.clone()
            runner.reset_envs()
            next_actor_obs = runner.get_noisy_obs(
                runner.actor_cfg["obs"], runner.actor_cfg["noise"]
            )
            next_critic_obs = runner.get_obs(runner.critic_cfg["obs"])
            transitions.append(
                TensorDict(
                    {
                        "actor_obs": actor_obs,
                        "critic_obs": critic_obs,
                        "actions": actions,
                        "next_actor_obs": next_actor_obs,
                        "next_critic_obs": next_critic_obs,
                        "rewards": total_rewards,
                        "timed_out": timed_out,
                        "terminated": terminated,
                        "dones": timed_out | terminated,
                    },
                    batch_size=env.num_envs,
                )
            )
            actor_obs, critic_obs = next_actor_obs, next_critic_obs

    runner.alg.update(torch.stack(transitions))

    assert math.isfinite(runner.alg.mean_value_loss)
    assert math.isfinite(runner.alg.mean_surrogate_loss)
    for module, previous in (
        (runner.alg.actor, actor_before),
        (runner.alg.critic, critic_before),
    ):
        assert all(torch.isfinite(parameter).all() for parameter in module.parameters())
        assert any(
            not torch.equal(parameter.detach(), before)
            for parameter, before in zip(module.parameters(), previous)
        )
