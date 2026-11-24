"""Evaluation protocol and native MuJoCo telemetry regressions."""

import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate_friction as evaluation
from evaluation_contacts import FootGroundContacts
from sweep_evaluation import discover_checkpoints, main as sweep_evaluation


def test_maneuver_overrides_compile_to_control_steps(tmp_path):
    config = tmp_path / "maneuvers.json"
    stages = [[0.08, 0, 0, 0], [0.12, 1.8, 0, 0], [0.10, 0, 0, 0]]
    config.write_text(json.dumps({"stand_2_stop": stages}))
    maneuvers = evaluation.select_maneuvers("stand_2_stop,steady_1p5", config)
    assert evaluation.compile_maneuver(maneuvers["stand_2_stop"], 0.02) == [
        (4, (0, 0, 0)),
        (6, (1.8, 0, 0)),
        (5, (0, 0, 0)),
    ]
    assert maneuvers["steady_1p5"] == ((4.0, 1.5, 0, 0),)
    assert len(evaluation.select_maneuvers("all", config)) == 7
    config.write_text(json.dumps({"stand_2_stop": [[0, 0, 0, 0]]}))
    with pytest.raises(ValueError, match="positive seconds"):
        evaluation.select_maneuvers("stand_2_stop", config)
    with pytest.raises(ValueError, match="at least one"):
        evaluation.compile_maneuver(((0.001, 0, 0, 0),), 0.02)


def test_foot_ground_force_rotates_contact_axes_and_handles_both_geom_orders(
    monkeypatch,
):
    model = SimpleNamespace(
        geom_bodyid=np.array([0, 4, 5, 2]),
        geom_type=np.array([mujoco.mjtGeom.mjGEOM_PLANE, 2, 2, 2]),
    )
    up = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    down = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    contacts = [
        SimpleNamespace(geom=(0, 1), efc_address=0, frame=up),
        SimpleNamespace(geom=(1, 0), efc_address=3, frame=down),
        SimpleNamespace(geom=(2, 3), efc_address=6, frame=up),  # self-contact
        SimpleNamespace(geom=(0, 2), efc_address=-1, frame=up),  # inactive
    ]
    backend = SimpleNamespace(
        _model_for_env=lambda _: model,
        _datas=[SimpleNamespace(contact=contacts)],
        _canonical_to_native_body_np=np.array([4, 5]),
    )
    calls = []

    def contact_force(_, __, index, wrench):
        calls.append(index)
        wrench[:] = [8, 2 if index == 0 else -2, 3, 0, 0, 0]

    monkeypatch.setattr(mujoco, "mj_contactForce", contact_force)
    sample = FootGroundContacts(
        SimpleNamespace(_backend=backend, feet_indices=torch.arange(2))
    )
    active, forces = sample.read()
    assert calls == [0, 1]
    np.testing.assert_array_equal(active, [[True, False]])
    np.testing.assert_array_equal(forces, [[[-6, 4, 16], [0, 0, 0]]])


def test_roll_pitch_yaw_uses_xyzw_order():
    angle = 0.7
    sine, cosine = np.sin(angle / 2), np.cos(angle / 2)
    quats = np.column_stack((np.eye(3) * sine, np.full(3, cosine)))
    np.testing.assert_allclose(evaluation.roll_pitch_yaw(quats), np.eye(3) * angle)


def test_sweep_discovers_latest_completed_step1_checkpoint_and_previews(
    tmp_path, capsys
):
    exp = tmp_path / "logs" / "go2noise_scale_1p00_seed_42"
    for run, status, done_at, iterations in (
        ("older", "completed", "2026-11-01", (20,)),
        ("newer", "completed", "2026-11-02", (2, 10)),
        ("incomplete", "running", "2026-11-03", (100,)),
    ):
        folder = exp / run
        folder.mkdir(parents=True)
        (folder / "friction_experiment.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "completed_at": done_at,
                    "noise_scale": 1,
                }
            )
        )
        for iteration in iterations:
            (folder / f"model_{iteration}.pt").touch()
    checkpoint = exp / "newer" / "model_10.pt"
    assert list(discover_checkpoints(exp.parent)) == [checkpoint]
    output = tmp_path / "out"
    sweep_evaluation(
        [
            "--logs-dir",
            str(exp.parent),
            "--output-dir",
            str(output),
            "--dry-run",
            "--trials",
            "1",
        ]
    )
    text = capsys.readouterr().out
    assert "1 checkpoints, 20 trials each" in text
    assert str(checkpoint) in text and "--noise-scales 0.0,1.0" in text
    assert not output.exists()


def test_cpu_dataset_records_friction_policy_inputs_native_telemetry_and_falls(
    tmp_path, monkeypatch
):
    args = SimpleNamespace(
        frictions=(0.4, 1.25),
        noise_scales=(0.0, 1.0),
        num_envs=2,
        device="cpu",
        gait_frequency=2.0,
        original_cfg=False,
    )
    seeds = (42, 73)
    maneuvers = {"stand_2_stop": ((0.16, 0, 0, 0), (0.16, 2, 0, 0), (0.16, 0, 0, 0))}
    checkpoint = tmp_path / "model_0.pt"
    env, cfg, schedules = evaluation.make_environment(
        checkpoint, args, seeds, maneuvers
    )
    try:
        source = evaluation.task_registry.make_alg_runner(env, cfg)
        with torch.no_grad():
            for param in source.alg.actor.NN.parameters():
                param.zero_()
        source.log_dir = str(tmp_path)
        source.save()
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        policy = evaluation.load_policy(env, cfg, checkpoint)
        assert policy.it == 0 and not policy.alg.actor.training
        assert all(not param.requires_grad for param in policy.alg.actor.parameters())

        backend = env._backend
        native_feet = backend._canonical_to_native_body_np[env.feet_indices.numpy()]
        obs_names = policy.actor_cfg["obs"]
        widths = [policy.get_obs_size([name]) for name in obs_names]
        received = []
        policy.alg.actor.NN.register_forward_pre_hook(
            lambda _, inputs: received.append(inputs[0].clone())
        )
        checked_profiles = set()
        original_inference = evaluation.inference_actions

        def check_inference(runner, std, generators):
            clean = runner.get_obs(obs_names)
            actions = original_inference(runner, std, generators)
            changed = {
                name
                for name, values in zip(
                    obs_names,
                    (received.pop() - clean).split(widths, dim=1),
                    strict=True,
                )
                if torch.count_nonzero(values).item()
            }
            assert changed == (
                {"base_ang_vel", "projected_gravity", "dof_pos_obs", "dof_vel"}
                if std is not None
                else set()
            )
            assert torch.equal(runner.get_obs(obs_names), clean)
            checked_profiles.add(std is not None)
            return actions

        original_reset = evaluation.reset_batch
        original_step = mujoco.mj_step
        expected_mu = None
        solved_contacts = set()

        def check_reset(task, friction, trial_seeds):
            nonlocal expected_mu
            expected_mu = torch.tensor(friction).item()
            return original_reset(task, friction, trial_seeds)

        def check_mj_step(model, data):
            plane = model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
            np.testing.assert_array_equal(model.geom_friction[plane, 0], expected_mu)
            original_step(model, data)
            for contact in data.contact:
                if (
                    contact.efc_address >= 0
                    and contact.dim >= 3
                    and plane[contact.geom].any()
                ):
                    np.testing.assert_allclose(
                        contact.friction[:2], expected_mu, atol=1e-12
                    )
                    solved_contacts.add(expected_mu)

        original_capture = evaluation.capture_step
        reference_normal_forces = []

        def check_frame(task, contacts, step, stage, invalid_action):
            trace = original_capture(task, contacts, step, stage, invalid_action)
            for world, data in enumerate(backend._datas):
                model = backend._model_for_env(world)
                # The short neutral-action run has ground contacts on the feet.
                # Compare against MuJoCo's independent world-frame body wrenches.
                np.testing.assert_allclose(
                    trace["foot_ground_force_world_N"][world],
                    data.cfrc_ext[native_feet, 3:6],
                    atol=1e-8,
                    rtol=1e-8,
                )
                reference_normal_forces.extend(data.cfrc_ext[native_feet, 5])
                for slot, native_id in enumerate(native_feet):
                    velocity = np.zeros(6)
                    mujoco.mj_objectVelocity(
                        model, data, mujoco.mjtObj.mjOBJ_XBODY, native_id, velocity, 0
                    )
                    np.testing.assert_allclose(
                        trace["foot_velocity_world_m_s"][world, slot],
                        velocity[3:],
                        atol=2e-6,
                        rtol=1e-5,
                    )
            return trace

        step_env = env.step

        def trigger_one_fall():
            step_env()
            if env.episode_length_buf[0].item() == 14:
                env.terminated[0] = True

        monkeypatch.setattr(evaluation, "inference_actions", check_inference)
        monkeypatch.setattr(evaluation, "reset_batch", check_reset)
        monkeypatch.setattr(evaluation, "capture_step", check_frame)
        monkeypatch.setattr(mujoco, "mj_step", check_mj_step)
        monkeypatch.setattr(env, "step", trigger_one_fall)

        out = tmp_path / "dataset"
        metadata = evaluation.record_evaluation(
            checkpoint, out, env, policy, cfg, schedules, args, seeds
        )
        assert metadata["status"] == "completed" and metadata["trials_written"] == 8
        assert metadata["checkpoint_sha256"] == checkpoint_hash
        assert metadata["foot_order"] == ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        assert checked_profiles == {False, True}
        assert solved_contacts == {
            torch.tensor(value).item() for value in args.frictions
        }
        assert max(reference_normal_forces) > 0
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash
        for name, weights in source.alg.actor.state_dict().items():
            torch.testing.assert_close(policy.alg.actor.state_dict()[name], weights)

        with (out / "trials.csv").open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 8
        for row in rows:
            fall = row["trial_index"] == "0"
            length = 14 if fall else 24
            assert row["status"] == ("fall" if fall else "completed")
            with np.load(out / row["trace"], allow_pickle=False) as trace:
                np.testing.assert_allclose(
                    trace["time_s"], env.dt * np.arange(1, length + 1)
                )
                np.testing.assert_array_equal(trace["command"][:8, 0], 0.0)
                np.testing.assert_array_equal(
                    trace["command"][8 : min(16, length), 0], 2.0
                )
                np.testing.assert_array_equal(trace["command"][16:, 0], 0.0)
                assert trace["foot_contact"].shape == (length, 4)
                assert trace["fell"][-1] == trace["terminated"][-1] == fall
                assert not trace["timed_out"].any()
                assert not trace["numerical_failure"].any()
    finally:
        env._backend.close()


def test_cpu_autoreset_and_timeout_end_only_the_affected_trials(tmp_path, monkeypatch):
    args = SimpleNamespace(
        frictions=(0.4,),
        noise_scales=(0.0,),
        num_envs=3,
        device="cpu",
        gait_frequency=2.0,
        original_cfg=False,
    )
    seeds = (42, 73, 85)
    checkpoint = tmp_path / "model_0.pt"
    env, cfg, schedules = evaluation.make_environment(
        checkpoint, args, seeds, {"stand_2_stop": ((0.2, 0.0, 0.0, 0.0),)}
    )
    try:
        runner = evaluation.task_registry.make_alg_runner(env, cfg)
        with torch.no_grad():
            for param in runner.alg.actor.NN.parameters():
                param.zero_()
        runner.log_dir = str(tmp_path)
        runner.save()
        runner = evaluation.load_policy(env, cfg, checkpoint)
        native_step = mujoco.mj_step
        task_step = env.step
        first_world = env._backend._datas[0]

        def unstable_step(model, data):
            if data is first_world and env.episode_length_buf[0].item() == 3:
                data.qvel[0] = np.nan
            native_step(model, data)

        def timed_step():
            task_step()
            if env.episode_length_buf[1].item() == 6:
                env.timed_out[1] = True

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(mujoco, "mj_step", unstable_step)
        monkeypatch.setattr(env, "step", timed_step)
        out = tmp_path / "dataset"
        report = evaluation.record_evaluation(
            checkpoint, out, env, runner, cfg, schedules, args, seeds
        )
        assert report["status"] == "completed" and report["trials_written"] == 3
        with (out / "trials.csv").open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert [(row["status"], int(row["recorded_steps"])) for row in rows] == [
            ("numerical_failure", 4),
            ("timeout", 6),
            ("completed", 10),
        ]
        with np.load(out / rows[0]["trace"], allow_pickle=False) as trace:
            assert np.isfinite(trace["base_position_world_m"][-1]).all()
            np.testing.assert_array_equal(
                trace["numerical_failure"], [False, False, False, True]
            )
        with np.load(out / rows[1]["trace"], allow_pickle=False) as trace:
            np.testing.assert_array_equal(trace["timed_out"], [False] * 5 + [True])
            assert not trace["numerical_failure"].any()
        with np.load(out / rows[2]["trace"], allow_pickle=False) as trace:
            assert not (trace["timed_out"] | trace["numerical_failure"]).any()
    finally:
        env._backend.close()
