"""Replay saved CPU smoke datasets and cross-check telemetry against MuJoCo."""

import argparse
from collections import Counter
import csv
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate_friction as evaluation


class NativeAudit:
    def __init__(self, env, runner, foot_order):
        self.env, self.runner = env, runner
        self.backend = env._backend
        model = self.backend._model_for_env(0)
        self.feet = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in foot_order
            ]
        )
        assert len(self.feet) == len(set(self.feet)) == 4 and (self.feet > 0).all()
        self.base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        assert self.base > 0
        self.ground = (model.geom_bodyid == 0) & (
            model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
        )
        self.mass = model.body_mass.copy()
        self.gains = (env.p_gains.clone(), env.d_gains.clone())
        self.observations = runner.actor_cfg["obs"]
        self.widths = [runner.get_obs_size([name]) for name in self.observations]
        self.scales = torch.cat(
            [
                torch.as_tensor(env.scales.get(name, 1.0)).expand(width)
                for name, width in zip(self.observations, self.widths, strict=True)
            ]
        )
        self.solved_friction = Counter()
        self.calls = Counter()
        self.max_errors = Counter()
        self.max_force_z_n = 0.0
        self.orig_reset = evaluation.reset_batch
        self.orig_step = mujoco.mj_step
        self.orig_capture = evaluation.capture_step
        self.orig_inference = evaluation.inference_actions

    def reset(self, env, friction, seeds):
        self.friction = friction
        self.applied = float(np.float32(friction))
        phase, generators = self.orig_reset(env, friction, seeds)
        np.testing.assert_array_equal(
            self.backend.contact_friction.numpy(), self.applied
        )
        for world in range(env.num_envs):
            model = self.backend._model_for_env(world)
            np.testing.assert_array_equal(model.geom_friction[:, 0], self.applied)
            np.testing.assert_array_equal(model.body_mass, self.mass)
        assert torch.equal(env.p_gains, self.gains[0])
        assert torch.equal(env.d_gains, self.gains[1])
        self.start_clock = [data.time for data in self.backend._datas]
        np.testing.assert_allclose(self.start_clock, env.dt, atol=1e-12, rtol=0)
        self.active = np.arange(env.num_envs) < len(seeds)
        padded = (*seeds, *((seeds[0],) * (env.num_envs - len(seeds))))
        self.noise_rng = [torch.Generator().manual_seed(seed) for seed in padded]
        phase_draw = math.tau * torch.stack(
            [torch.rand((), generator=rng) for rng in self.noise_rng]
        )
        torch.testing.assert_close(phase, phase_draw, rtol=0, atol=0)
        self.calls["resets"] += 1
        return phase, generators

    def solver_step(self, model, data):
        np.testing.assert_array_equal(model.geom_friction[:, 0], self.applied)
        self.orig_step(model, data)
        for contact in data.contact:
            if contact.efc_address >= 0 and self.ground[contact.geom].any():
                np.testing.assert_allclose(
                    contact.friction[:2],
                    self.applied,
                    atol=1e-12,
                    rtol=0,
                )
                self.solved_friction[self.friction] += 1

    def receive_actor_input(self, module, inputs):
        self.actor_input = inputs[0].clone()

    def inference(self, runner, std, generators):
        clean = runner.get_obs(self.observations)
        self.actor_input = None
        actions = self.orig_inference(runner, std, generators)
        assert self.actor_input is not None and not self.actor_input.requires_grad
        if self.noise_scale:
            draw = torch.stack(
                [torch.randn(clean.shape[1], generator=rng) for rng in self.noise_rng]
            )
            perturbed = clean + draw * self.physical_std / self.scales
            torch.testing.assert_close(self.actor_input, perturbed, rtol=0, atol=2e-6)
            observed = self.actor_input - clean
            changed = (observed != 0).any(dim=0)
            assert not changed[self.physical_std == 0].any()
            assert changed[self.physical_std > 0].any()
            key = f"actor_delta_max_noise_{self.noise_scale:g}"
            self.max_errors[key] = max(
                self.max_errors[key], observed.abs().max().item()
            )
        else:
            torch.testing.assert_close(self.actor_input, clean, rtol=0, atol=0)
        torch.testing.assert_close(
            runner.get_obs(self.observations), clean, rtol=0, atol=0
        )
        self.calls[f"actor_batches_noise_{self.noise_scale:g}"] += 1
        return actions

    def compare(self, name, recorded, reference, atol):
        np.testing.assert_allclose(recorded, reference, atol=atol, rtol=0, err_msg=name)
        error = float(np.max(np.abs(recorded - reference), initial=0))
        self.max_errors[name] = max(error, self.max_errors[name])

    def capture(self, task, contacts, step, stage, invalid_action):
        frame = self.orig_capture(task, contacts, step, stage, invalid_action)
        for world in np.flatnonzero(self.active & ~frame["numerical_failure"]):
            model = self.backend._model_for_env(world)
            data = self.backend._datas[world]
            measured = {key: values[world] for key, values in frame.items()}
            self.compare(
                "clock_s",
                measured["time_s"],
                data.time - self.start_clock[world],
                1e-10,
            )

            self.compare(
                "position_m", measured["base_position_world_m"], data.qpos[:3], 1e-6
            )
            self.compare(
                "quaternion_xyzw",
                measured["base_orientation_xyzw"],
                data.qpos[[4, 5, 6, 3]],
                1e-6,
            )
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
            r = rotation.reshape(3, 3)
            euler = np.array(
                [
                    np.arctan2(r[2, 1], r[2, 2]),
                    np.arctan2(-r[2, 0], np.hypot(r[0, 0], r[1, 0])),
                    np.arctan2(r[1, 0], r[0, 0]),
                ]
            )
            angle_error = (
                measured["base_roll_pitch_yaw_rad"] - euler + np.pi
            ) % math.tau - np.pi
            self.compare("roll_pitch_yaw_rad", angle_error, 0, 2e-5)
            self.compare(
                "base_velocity_world_m_s",
                measured["base_linear_velocity_world_m_s"],
                data.qvel[:3],
                2e-6,
            )
            self.compare(
                "base_velocity_body_m_s",
                measured["base_linear_velocity_body_m_s"],
                r.T @ data.qvel[:3],
                2e-6,
            )
            self.compare(
                "base_angular_body_rad_s",
                measured["base_angular_velocity_body_rad_s"],
                data.qvel[3:6],
                2e-5,
            )
            self.compare(
                "base_angular_world_rad_s",
                measured["base_angular_velocity_world_rad_s"],
                r @ data.qvel[3:6],
                2e-5,
            )
            fell = np.linalg.norm(data.cfrc_ext[self.base, 3:6]) > 1.0
            assert measured["fell"] == measured["terminated"] == fell
            assert measured["timed_out"] == (step >= task.max_episode_length)
            self.calls["native_falls"] += int(fell)

            active_ground, other_contacts = set(), set()
            for contact in data.contact:
                if contact.efc_address < 0:
                    continue
                first, second = contact.geom
                first_body, second_body = model.geom_bodyid[contact.geom]
                if self.ground[first]:
                    active_ground.add(second_body)
                elif self.ground[second]:
                    active_ground.add(first_body)
                else:
                    other_contacts.update((first_body, second_body))
            contact_mask = np.isin(self.feet, list(active_ground))
            np.testing.assert_array_equal(measured["foot_contact"], contact_mask)
            assert np.all(measured["foot_ground_force_world_N"][~contact_mask] == 0)
            # cfrc_ext also includes self-contacts. Compare physical ground-only
            # forces to its world-frame body wrenches when that foot has no extras.
            ground_only = ~np.isin(self.feet, list(other_contacts))
            self.compare(
                "force_world_N",
                measured["foot_ground_force_world_N"][ground_only],
                data.cfrc_ext[self.feet[ground_only], 3:6],
                2e-8,
            )
            force_z = measured["foot_ground_force_world_N"][:, 2]
            assert force_z.min() >= -1e-8
            self.max_force_z_n = max(self.max_force_z_n, float(force_z.max()))
            self.calls["contact_slots"] += int(contact_mask.sum())
            self.calls["force_slots"] += int(ground_only.sum())
            jac = np.empty((3, model.nv))
            for slot, foot in enumerate(self.feet):
                mujoco.mj_jacBody(model, data, jac, None, int(foot))
                self.compare(
                    "foot_velocity_world_m_s",
                    measured["foot_velocity_world_m_s"][slot],
                    jac @ data.qvel,
                    2e-6,
                )
            self.calls["frames"] += 1
        self.active &= ~(
            frame["terminated"] | frame["timed_out"] | frame["numerical_failure"]
        )
        return frame


def verify_weights(runner, checkpoint_state):
    assert runner.it == checkpoint_state["iter"]
    for name in ("actor", "critic"):
        module = getattr(runner.alg, name)
        assert not module.training
        assert all(not value.requires_grad for value in module.parameters())
        current = module.state_dict()
        saved = checkpoint_state[f"{name}_state_dict"]
        assert current.keys() == saved.keys()
        for key in current:
            torch.testing.assert_close(current[key], saved[key], rtol=0, atol=0)


def replay(dataset):
    meta = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    assert meta["protocol"] == "go2trot-friction-eval-v1"
    assert meta["backend"] == "mujoco_cpu" and meta["policy_device"] == "cpu"
    checkpoint = Path(meta["checkpoint"])
    with checkpoint.open("rb") as source:
        assert (
            hashlib.file_digest(source, "sha256").hexdigest()
            == meta["checkpoint_sha256"]
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["iter"] == meta["checkpoint_iteration"]
    args = SimpleNamespace(
        frictions=meta["frictions"],
        num_envs=meta["num_envs"],
        device="cpu",
        gait_frequency=meta["gait_frequency_hz"],
        original_cfg=False,
    )
    schedules = {
        name: tuple(
            (stage["steps"] * meta["dt_s"], *stage["command"]) for stage in stages
        )
        for name, stages in meta["maneuvers"].items()
    }
    with (dataset / "trials.csv").open(newline="", encoding="utf-8") as source:
        index = list(csv.DictReader(source))
    env, cfg, compiled = evaluation.make_environment(
        checkpoint, args, meta["trial_seeds"], schedules
    )
    try:
        runner = evaluation.load_policy(env, cfg, checkpoint)
        verify_weights(runner, state)
        assert not env.cfg.push_robots.toggle
        assert env.cfg.init_state.reset_mode == "reset_to_basic"
        assert not env._backend._randomize_link_mass
        assert env.cfg.domain_randomization.episode.scale_ranges == {}
        assert env.cfg.asset.terminate_after_contacts_on == ["base"]
        audit = NativeAudit(env, runner, meta["foot_order"])
        assert not runner.actor_cfg["normalize_obs"]
        hook = runner.alg.actor.NN.register_forward_pre_hook(audit.receive_actor_input)
        with (
            patch.object(evaluation, "reset_batch", audit.reset),
            patch.object(evaluation, "capture_step", audit.capture),
            patch.object(evaluation, "inference_actions", audit.inference),
            patch.object(mujoco, "mj_step", audit.solver_step),
            patch.object(
                env, "_push_robots", side_effect=AssertionError("unexpected push")
            ),
        ):
            for (name, friction, scale), group in groupby(
                index,
                key=lambda row: (
                    row["maneuver"],
                    float(row["friction"]),
                    float(row["noise_scale"]),
                ),
            ):
                audit.noise_scale = scale
                noise_cfg = evaluation.Go2TrotRunnerCfg()
                defaults = evaluation.configure_observation_noise(noise_cfg, scale)
                expected = (
                    {name: value * scale for name, value in defaults.items()}
                    if scale
                    else {}
                )
                assert expected == meta["noise_std_physical"][str(scale)]
                audit.physical_std = torch.cat(
                    [
                        torch.as_tensor(expected.get(obs, 0.0)).expand(width)
                        for obs, width in zip(
                            audit.observations, audit.widths, strict=True
                        )
                    ]
                )
                std = evaluation.observation_noise_std(runner, noise_cfg.actor.noise)
                rows = list(group)
                assert [int(row["eval_seed"]) for row in rows] == meta["trial_seeds"]
                for start in range(0, len(rows), env.num_envs):
                    batch = rows[start : start + env.num_envs]
                    seeds = [int(row["eval_seed"]) for row in batch]
                    results = evaluation.collect_batch(
                        env,
                        runner,
                        evaluation.FootGroundContacts(env),
                        friction,
                        compiled[name],
                        seeds,
                        std,
                    )
                    for row, (trace, phase) in zip(batch, results, strict=True):
                        assert float(row["applied_friction"]) == audit.applied
                        assert phase == float(row["initial_phase_rad"])
                        assert evaluation.trial_status(trace) == row["status"]
                        assert len(trace["time_s"]) == int(row["recorded_steps"])
                        with np.load(
                            dataset / row["trace"], allow_pickle=False
                        ) as saved:
                            assert trace.keys() == set(saved.files)
                            for key in saved.files:
                                np.testing.assert_array_equal(
                                    trace[key],
                                    saved[key],
                                    err_msg=f"{row['trial_id']} {key}",
                                )
        hook.remove()
        verify_weights(runner, state)
        assert set(audit.solved_friction) == set(meta["frictions"])
        assert audit.max_force_z_n > 0
        return {
            "checkpoint": meta["training"]["experiment_name"],
            "dataset": str(dataset),
            "trials": len(index),
            "checks": dict(audit.calls),
            "solved_ground_contacts": dict(audit.solved_friction),
            "peak_normal_force_N": audit.max_force_z_n,
            "max_observed_differences": dict(audit.max_errors),
        }
    finally:
        env._backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, help="Saved Step 2 CPU dataset or sweep directory"
    )
    args = parser.parse_args()
    datasets = sorted(path.parent for path in args.root.rglob("metadata.json"))
    if not datasets:
        parser.error("No evaluation datasets found")
    for dataset in datasets:
        print(json.dumps(replay(dataset), sort_keys=True))


if __name__ == "__main__":
    main()
