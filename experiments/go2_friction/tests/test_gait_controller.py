"""CPU controller and checkpoint checks using zero or scripted actions."""

from contextlib import contextmanager
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate_friction as evaluation
from controller_checkpoint import (
    controller_checkpoint_profile,
    read_controller_profile,
    save_controller_checkpoints,
)
from controller_telemetry import ControllerTelemetry
from gait_controller import (
    ControllerOptions,
    configure_controller,
    CONTROLLER_TASK,
    paired_contact_score,
    register_controller,
)
from gym.utils.helpers import class_to_dict
from gym.utils.task_registry import task_registry
import train_friction


@contextmanager
def controller_env(options, *, scripted=True):
    env_cfg, train_cfg = deepcopy(task_registry.get_cfgs("go2trot"))
    env_cfg.seed = train_cfg.seed = 42
    env_cfg.env.num_envs = 2
    train_friction.configure_friction(env_cfg, 1.25, 1.25, friction_only=True)
    train_friction.configure_observation_noise(train_cfg, 0.0)
    configure_controller(env_cfg, train_cfg, options)
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    task = register_controller(env_cfg, train_cfg, options, scripted=scripted)
    env = task_registry.make_env(task, env_cfg, "cpu", True, "mujoco")
    try:
        yield env
    finally:
        env._backend.close()


@pytest.mark.parametrize("mode", ["reward-only", "additive"])
def test_zero_action_and_scripted_reference(mode):
    options = ControllerOptions(
        reference_mode=mode, command_curriculum="stand-walk", clip_joint_targets=True
    )
    with controller_env(options) as env, torch.inference_mode():
        nominal = env._nominal_target.expand_as(env.dof_pos_target)
        telemetry = ControllerTelemetry(env)
        assert tuple(env.robot_layout.body_groups["feet"]) == (
            "FL_foot",
            "FR_foot",
            "RL_foot",
            "RR_foot",
        )
        assert torch.all(env.phase_frequency == 2.0)
        assert torch.all(env._backend.contact_friction == 1.25)
        # Both modes give the same static target during a standing command.
        for _ in range(85):
            env.step()
            torch.testing.assert_close(env.pd_joint_target, nominal, rtol=0, atol=0)
            assert not env.gait_blend.any()
            sample = telemetry.sample(env.commands)
        assert sample["sustained_non_foot_support"].all()
        assert (sample["non_foot_support_N"] > 5).all()
        assert not env.terminated.any()  # The base-only fall flag misses this support.
        _, forces, non_foot = telemetry.contacts.read(include_non_foot=True)
        vertical, vertical_non_foot = telemetry.contacts.read_vertical_loads()
        np.testing.assert_allclose(forces[:, :, 2], vertical, atol=1e-9, rtol=0)
        np.testing.assert_allclose(non_foot, vertical_non_foot, atol=1e-9, rtol=0)

        evaluation.reset_batch(env, 1.25, (10000, 10001))
        env.phase.fill_(0.37)
        env.commands[:, 0] = torch.tensor([0.5, 1.0])
        offsets = torch.tensor([0, math.pi, math.pi, 0]).repeat_interleave(3)
        amplitude = torch.tensor([0.0, -0.15, 0.30]).repeat(4)
        for _ in range(38):
            phase = env.phase.clone()
            env.step()
            wave = amplitude * torch.sin(phase + offsets)
            expected = nominal + env.gait_blend * wave
            torch.testing.assert_close(env.pd_joint_target, expected, rtol=0, atol=5e-7)
            assert torch.equal(env.commands[:, 0], torch.tensor([0.5, 1.0]))
            assert not env.dof_pos_target.any()
            stance = env._expected_stance()
            assert torch.equal(stance[:, 0], stance[:, 3])
            assert torch.equal(stance[:, 1], stance[:, 2])
            assert (~(stance[:, 0] & stance[:, 1])).all()
        if mode == "additive":
            assert env.gait_blend.eq(1).all()
        else:
            assert not env.gait_blend.any()

        # A standing command removes the drive smoothly, even while phase runs.
        env.commands.zero_()
        env.step()
        if mode == "additive":
            assert ((0 < env.gait_blend) & (env.gait_blend < 1)).all()
        for _ in range(13):
            env.step()
        torch.testing.assert_close(env.pd_joint_target, nominal, atol=0, rtol=0)

        for column, value in enumerate((-10.0, 10.0)):
            residual = torch.full_like(env.dof_pos_target, value)
            env.dof_pos_target.copy_(residual)
            env.step()
            limit = env._actuated_dof_pos_limits[:, column]
            torch.testing.assert_close(env.pd_joint_target, limit.expand_as(residual))
            torch.testing.assert_close(env.dof_pos_target, residual, rtol=0, atol=0)
            assert env.clipped_joint_targets.tolist() == [12, 12]


def test_stand_walk_distribution_and_seeded_sampling():
    options = ControllerOptions(command_curriculum="stand-walk")
    with controller_env(options, scripted=False) as env:
        rng = torch.random.get_rng_state()
        mask = torch.ones(2, dtype=torch.bool)
        samples = []
        for _ in range(1_000):
            env._resample_commands(mask)
            samples.append(env.commands.clone())
        commands = torch.cat(samples)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, atol=0, rtol=0)
        assert not commands[:, 1:].any()
        for value, p in ((0.0, 0.2), (0.5, 0.4), (1.0, 0.4)):
            assert (commands[:, 0] == value).float().mean() == pytest.approx(
                p, abs=0.03
            )
        env.commands[0] = torch.tensor([-0.5, 0.25, 0.5])
        env._resample_commands(torch.tensor([False, True]))
        torch.testing.assert_close(env.commands[0], torch.tensor([-0.5, 0.25, 0.5]))


def test_independent_frequency_height_and_reward_options(monkeypatch):
    env_cfg, train_cfg = deepcopy(task_registry.get_cfgs("go2trot"))
    configure_controller(env_cfg, train_cfg, ControllerOptions(base_height_target=0.34))
    assert env_cfg.reward_settings.base_height_target == 0.34
    assert env_cfg.init_state.pos[2] == 0.40
    assert env_cfg.control.desired_sim_frequency == 50
    assert train_cfg.critic.reward.weights.swing_contact == 3.0

    configure_controller(
        env_cfg, train_cfg, ControllerOptions(coordination_reward="paired-contact")
    )
    assert train_cfg.critic.reward.weights.paired_contact == 3.0
    assert train_cfg.critic.reward.weights.swing_contact == 0.0
    assert train_cfg.critic.reward.weights.trot_support == 0.0
    assert train_cfg.critic.reward.weights.reference_traj == 3.0
    phase = torch.tensor([[1.0, -1.0, -1.0, 1.0]]).repeat(5, 1)
    loads = torch.tensor(
        [
            [80.0, 0.0, 0.0, 80.0],
            [40.0, 40.0, 40.0, 40.0],
            [0.0, 80.0, 80.0, 0.0],
            [80.0, 0.0, 0.0, 80.0],
            [80.0, 8.0, 0.0, 8.0],
        ]
    )
    scores = paired_contact_score(phase, loads, 160.0, torch.tensor([1, 1, 1, 0, 1]))
    torch.testing.assert_close(scores, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.25]))

    options = ControllerOptions(physics_hz=100, reference_mode="additive")
    with controller_env(options) as env, torch.inference_mode():
        assert env.dt == 0.02 and env.cfg.control.decimation == 2
        model = env._backend._model_for_env(0)
        assert model.opt.timestep == 0.01
        assert np.all(model.geom_solref[:, 0] == 0.04)
        assert np.all(model.jnt_solref[model.jnt_limited.astype(bool), 0] == 0.04)
        assert env.cfg.reward_settings.base_height_target == 0.40
        env.commands[:, 0] = 1.0
        env.phase.fill_(0.4)
        env.phase_frequency.fill_(2.0)
        env.gait_ramp.fill_(1.0)
        env.dof_pos_target.fill_(0.123)
        compute = env._compute_torques
        substeps = []

        def record():
            torques = compute()
            substeps.append(
                (
                    env.phase.clone(),
                    env.dof_pos_target.clone(),
                    env.pd_joint_target.clone(),
                )
            )
            return torques

        monkeypatch.setattr(env, "_compute_torques", record)
        env.step()
        assert len(substeps) == 2
        torch.testing.assert_close(substeps[0][1], substeps[1][1])
        assert not torch.equal(substeps[0][2], substeps[1][2])
        torch.testing.assert_close(
            substeps[1][0] - substeps[0][0], torch.full((2, 1), math.tau * 2 * 0.01)
        )
        torch.testing.assert_close(
            env.phase, torch.full((2, 1), 0.4 + math.tau * 2 * 0.02)
        )


def test_fresh_pilot_cli_and_checkpoint_reconstruction(tmp_path, monkeypatch):
    # Run actual setup and initialization. Replace the training entry point with
    # a save and a few zero-action samples so this test performs no PPO update.
    captures = {}

    def test_log_dir(cfg):
        cfg.log_dir = str(tmp_path / cfg.runner.experiment_name)

    def verify_initialization(cfg, runner):
        name = cfg.runner.experiment_name
        captures[name] = class_to_dict(cfg)
        assert runner.it == 0 and runner.num_learning_iterations == 100
        assert runner.actor_cfg["frequency"] == 50 and runner.save_interval == 25
        assert runner.env.cfg.control.desired_sim_frequency == 50
        assert not runner.actor_cfg["noise"]
        runner.save()
        runner.save()
        runner.it = 1  # Label a few diagnostic samples as a partial rollout.
        rewards = runner.initialize_rewards_dict(1)
        with torch.inference_mode():
            for _ in range(8):
                runner.env.step()
                runner.update_rewards_dict(rewards, 0)

    monkeypatch.setattr(task_registry, "set_log_dir_name", test_log_dir)
    monkeypatch.setattr(train_friction, "train", verify_initialization)
    setup_module = sys.modules[train_friction.setup.__module__]
    monkeypatch.setattr(
        setup_module.local_code_save_helper,
        "save_local_files_to_logs",
        lambda *args, **kwargs: None,
    )
    profiles = {}
    states = {}
    for mode in ("reward-only", "additive"):
        train_friction.main(
            [
                "--task",
                "go2trot",
                "--device",
                "cpu",
                "--backend",
                "mujoco",
                "--seed",
                "42",
                "--min-friction",
                "1.25",
                "--max-friction",
                "1.25",
                "--noise-scale",
                "0",
                "--num_envs",
                "2",
                "--max_iterations",
                "100",
                "--save_interval",
                "25",
                "--headless",
                "--disable_wandb",
                "--rollout-size",
                "64",
                "--batch_size",
                "32",
                "--command-curriculum",
                "stand-walk",
                "--clip-joint-targets",
                "--reference-mode",
                mode,
                "--physics-hz",
                "50",
                "--coordination-reward",
                "legacy",
                "--experiment_name",
                mode,
            ]
        )
        directory = tmp_path / mode
        checkpoint = directory / "model_0.pt"
        profiles[mode] = read_controller_profile(checkpoint)
        states[mode] = torch.load(checkpoint, weights_only=True)
        manifest = json.loads((directory / "friction_experiment.json").read_text())
        assert manifest["controller_profile"] == profiles[mode]
        assert manifest["applied_friction_range"] == [1.25, 1.25]
        with (directory / "command_metrics.csv").open() as source:
            rows = [r for r in csv.DictReader(source) if int(r["samples"])]
        assert rows and all(
            r["tracking_rmse_m_s"] and r["mean_base_height_m"] for r in rows
        )
        assert sum(int(r["joint_target_samples"]) for r in rows) == 2 * 8 * 12
        assert all(r["joint_target_clip_fraction"] == "0.0" for r in rows)
    assert profiles["reward-only"]["env_config"] == profiles["additive"]["env_config"]
    assert captures["reward-only"]["algorithm"] == captures["additive"]["algorithm"]
    assert (
        captures["reward-only"]["critic"]["reward"]
        == captures["additive"]["critic"]["reward"]
    )
    for kind in ("actor_state_dict", "critic_state_dict"):
        for name, tensor in states["reward-only"][kind].items():
            torch.testing.assert_close(
                tensor, states["additive"][kind][name], atol=0, rtol=0
            )

    # Reconstruct from the checkpoint alone, with the source manifest absent.
    checkpoint = tmp_path / "portable.pt"
    shutil.copyfile(tmp_path / "additive" / "model_0.pt", checkpoint)
    args = SimpleNamespace(
        frictions=(1.25,),
        noise_scales=(0.0,),
        num_envs=2,
        device="cpu",
        gait_frequency=None,
        original_cfg=False,
    )
    seeds = (10000, 10001)
    maneuvers = {
        "stand_1_stop": (
            (0.16, 0.0, 0.0, 0.0),
            (0.6, 1.0, 0.0, 0.0),
            (0.3, 0.0, 0.0, 0.0),
        )
    }
    env, train_cfg, schedules = evaluation.make_environment(
        checkpoint, args, seeds, maneuvers
    )
    try:
        assert env.checkpoint_controller_profile == profiles["additive"]
        assert env.controller_options.reference_mode == "additive"
        assert env.cfg.control.gait_freq == [2.0, 2.0]
        runner = evaluation.load_policy(env, train_cfg, checkpoint)
        monkeypatch.setattr(
            evaluation, "inference_actions", lambda *_: torch.zeros(2, 12)
        )
        output = tmp_path / "evaluation"
        metadata = evaluation.record_evaluation(
            checkpoint, output, env, runner, train_cfg, schedules, args, seeds
        )
        assert metadata["controller_profile"] == profiles["additive"]
        assert metadata["trials_written"] == 2
        with (output / "trials.csv").open() as source:
            trial = next(csv.DictReader(source))
        with np.load(output / trial["trace"]) as trace:
            moving = trace["commanded_motion"]
            assert not trace["gait_reference_blend"][0]
            assert trace["gait_reference_blend"][moving].max() == 1.0
            assert not trace["gait_reference_blend"][-1]
            assert trace["pd_joint_target_rad"].shape == (len(moving), 12)
            assert trace["loaded_foot_contact"].shape == (len(moving), 4)
            assert trace["non_foot_support_N"].max() > 5
            assert trace["sustained_non_foot_support"].any()
            assert np.all(trace["gait_frequency_hz"] == 2.0)
    finally:
        env._backend.close()
    args.gait_frequency = 1.0
    with pytest.raises(ValueError, match="differs from the checkpoint"):
        evaluation.make_environment(checkpoint, args, seeds, maneuvers)


def test_checkpoint_restores_nondefault_controller_settings_and_protects_existing_file(
    tmp_path,
):
    options = ControllerOptions(
        reference_mode="additive",
        physics_hz=100,
        base_height_target=0.34,
        coordination_reward="paired-contact",
        clip_joint_targets=True,
    )
    with controller_env(options) as original:
        _, cfg = task_registry.get_cfgs(CONTROLLER_TASK)
        cfg.log_dir = str(tmp_path)
        runner = task_registry.make_alg_runner(original, cfg)
        profile = controller_checkpoint_profile(original, cfg)
        save_controller_checkpoints(runner, profile)
        checkpoint = tmp_path / "model_0.pt"
        runner.save()
        original_contents = checkpoint.read_bytes()
        save_controller_checkpoints(runner, profile)
        with pytest.raises(FileExistsError):
            runner.save()
        assert checkpoint.read_bytes() == original_contents
    seeds = (10000, 10001)
    args = SimpleNamespace(
        frictions=(1.25,),
        num_envs=2,
        device="cpu",
        gait_frequency=None,
        original_cfg=False,
    )
    env, cfg, _ = evaluation.make_environment(
        checkpoint, args, seeds, evaluation.select_maneuvers("steady_0p5")
    )
    try:
        assert env.controller_options == options
        assert env._backend._model_for_env(0).opt.timestep == 0.01
        assert env.cfg.reward_settings.base_height_target == 0.34
        assert cfg.critic.reward.weights.paired_contact == 3.0
        assert env.cfg.control.gait_freq == [1.0, 3.0]
        runner = evaluation.load_policy(env, cfg, checkpoint)
        assert runner.it == 0
        evaluation.reset_batch(env, 1.25, seeds)
        freq = env.phase_frequency.clone()
        assert torch.all((freq >= 1.0) & (freq <= 3.0))
        evaluation.reset_batch(env, 1.25, seeds)
        torch.testing.assert_close(env.phase_frequency, freq, atol=0, rtol=0)
    finally:
        env._backend.close()
