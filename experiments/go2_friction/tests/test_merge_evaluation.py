"""Offline regressions for extending the pilot grid and rescoring it."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from shutil import copyfile, copytree
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_results import read_csv, run_analysis
from controller_checkpoint import PROFILE_VERSION
import final_protocol as final
from gait_controller import MOTION_THRESHOLD, RAMP_S
from merge_evaluation import merge_evaluations, read_complete_evaluation
from metrics import Thresholds, run_metrics, write_csv
from plot_pilot_results import POLICIES, pilot_curves, plot_pilots
import pandas as pd
import synthetic_step4 as synthetic


@pytest.fixture
def pilot_protocol(monkeypatch):
    monkeypatch.setattr(synthetic, "FRICTIONS", final.TEST_FRICTIONS)
    monkeypatch.setattr(synthetic, "EVAL_SEEDS", final.EVAL_SEEDS)
    monkeypatch.setattr(
        synthetic,
        "SCHEDULES",
        {
            "stand_1_stop": ((4, (0, 0, 0)), (12, (1, 0, 0)), (8, (0, 0, 0))),
            "steady_0p5": ((12, (0.5, 0, 0)),),
            "steady_1p0": ((12, (1, 0, 0)),),
        },
    )


@pytest.fixture
def sources(tmp_path, pilot_protocol):
    original, extra = tmp_path / "original", tmp_path / "extra"
    policies = {policy[0] for policy in POLICIES.values()}
    for index, run in enumerate(
        r for r in final.final_training_runs() if r["experiment_name"] in policies
    ):
        name = run["experiment_name"]
        suffix = Path(name, "synthetic", "model_400")
        full = tmp_path / "fixture" / suffix
        method = "friction" if run["method"] == "baseline" else run["method"]
        value = run["lower_bound"] if method == "friction" else run["noise_scale"]
        synthetic.write_dataset(full, method, value, (0.6, 0.6), 0, 4 + index)
        meta = json.loads((full / "metadata.json").read_text())
        meta.update(
            checkpoint_iteration=400, checkpoint=f"synthetic/{name}/model_400.pt"
        )
        meta["training"].update(
            experiment_name=name,
            seed=run["seed"],
            final_research={
                "study": final.FINAL_RESEARCH_ID,
                "method": run["method"],
                "run_name": "synthetic",
                "template_checkpoint_sha256": "synthetic-template",
                "reward_schedule": final.reward_schedule(),
            },
        )
        trials = read_csv(full / "trials.csv")
        for root, frictions in ((original, (0.2, 0.4, 1.25)), (extra, (0.6, 0.8))):
            directory = root / suffix
            (directory / "traces").mkdir(parents=True)
            rows = [row.copy() for row in trials if float(row["friction"]) in frictions]
            for i, row in enumerate(rows):
                trial_id = f"trial_{i:06d}"
                trace = f"traces/{trial_id}.npz"
                copyfile(full / row["trace"], directory / trace)
                row.update(trial_id=trial_id, trace=trace)
            write_csv(directory / "trials.csv", list(rows[0]), rows)
            (directory / "metadata.json").write_text(
                json.dumps(
                    meta
                    | {
                        "frictions": frictions,
                        "trials_written": len(rows),
                        "env_config": {
                            "domain_randomization": {
                                "startup": {
                                    "contact_friction_range": [frictions[0]] * 2
                                }
                            }
                        },
                    }
                )
            )
    return original, extra


def test_merge_pilots_and_score_critical_friction_on_five_points(tmp_path, sources):
    original, extra = sources
    out = tmp_path / "evaluation"
    assert merge_evaluations(original, extra, out, dry_run=True) == (540, 360)
    assert not out.exists()
    assert merge_evaluations(original, extra, out) == (540, 360)
    for path in sorted(out.rglob("metadata.json")):
        meta, rows = read_complete_evaluation(path.parent)
        assert tuple(meta["frictions"]) == final.TEST_FRICTIONS
        assert len(rows) == len({row["trial_id"] for row in rows.values()}) == 300
        assert any(row["status"] == "fall" for row in rows.values())
        assert [source["trials_written"] for source in meta["merge_sources"]] == [
            180,
            120,
        ]
        for row in rows.values():
            source = Path(row["source_dataset"])
            raw = source / "traces" / f"{row['source_trial_id']}.npz"
            assert raw.read_bytes() == (path.parent / row["trace"]).read_bytes()
    with pytest.raises(FileExistsError):
        merge_evaluations(original, extra, out)

    metrics, analysis, figures = (
        tmp_path / name for name in ("metrics", "analysis", "figures")
    )
    assert (
        run_metrics(out, metrics, Thresholds(), {}, primary_maneuver="stand_1_stop")
        == 3
    )
    assert len(read_csv(metrics / "per_trial.csv")) == 900
    cutoffs = read_csv(metrics / "critical_friction.csv")
    assert len(cutoffs) == 6
    assert all(row["eligible_conditions"] == "5" for row in cutoffs)
    expected_cutoffs = {"0.0": "0.6", "1.0": "0.8"}
    assert all(
        row["critical_friction"] == expected_cutoffs[row["noise_scale"]]
        for row in cutoffs
    )
    report = run_analysis(metrics, analysis)
    assert report["data_kind"] == "synthetic"
    assert len(read_csv(analysis / "seed_conditions.csv")) == 120
    for scale in final.TEST_NOISE_SCALES:
        data = pd.read_csv(analysis / "seed_conditions.csv")
        curves = list(pilot_curves(data, "stand_1_stop", "success_rate", scale))
        assert len(curves) == 3
        assert all(
            tuple(values.index) == final.TEST_FRICTIONS for _, _, _, values in curves
        )
    plot_pilots(analysis, figures)
    assert {path.name for path in figures.iterdir()} == {
        f"pilot_comparison_noise_{scale}.{ext}"
        for scale in (0, 1)
        for ext in ("png", "pdf")
    }


def test_merge_rejects_overlaps_and_different_protocols(tmp_path, sources):
    original, extra = sources
    out = tmp_path / "evaluation"
    duplicate = tmp_path / "duplicate"
    copytree(original, duplicate)
    with pytest.raises(ValueError, match="Overlapping trial conditions"):
        merge_evaluations(original, duplicate, out)

    path = next(extra.rglob("metadata.json"))
    meta = json.loads(path.read_text())
    meta["mujoco_version"] = "different"
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="settings differ.*mujoco_version"):
        merge_evaluations(original, extra, out)
    assert not out.exists()


@pytest.fixture
def staged_sources(tmp_path, pilot_protocol):
    original, supplemental = tmp_path / "original", tmp_path / "supplemental"
    experiments = {
        "go2_final_staged_baseline_seed_42": original,
        "go2_final_staged_noise_scale_4p00_seed_43": supplemental,
    }
    for run in final.final_training_runs():
        name = run["experiment_name"]
        if name not in experiments:
            continue
        folder = experiments[name] / name / "staged" / "model_400"
        seed, scale = run["seed"], run["noise_scale"]
        synthetic.write_dataset(folder, "noise", scale, (0.6, 0.6), 0, 4)
        friction = [run["lower_bound"], final.FRICTION]
        noise = {"dof_pos_obs": 0.04} if scale else {}
        env = {
            "seed": seed,
            "domain_randomization": {"startup": {"contact_friction_range": friction}},
        }
        config = {
            "seed": seed,
            "log_dir": name,
            "runner": {"experiment_name": name, "run_name": "staged"},
            "actor": {"add_noise": bool(noise), "noise": noise},
        }
        stages = {
            stage["coordination_reward"]: {
                "status": "completed",
                "controller_profile": {
                    "version": PROFILE_VERSION,
                    "ramp_s": RAMP_S,
                    "motion_threshold": MOTION_THRESHOLD,
                    "options": {
                        "reference_mode": "additive",
                        "coordination_reward": stage["coordination_reward"],
                    },
                    "env_config": env,
                    "train_config": config
                    | {"critic": {"reward": {"weights": stage["weights"]}}},
                },
            }
            for stage in final.reward_schedule()
        }
        profile = stages["paired-contact"]["controller_profile"]
        training = {
            "experiment_name": name,
            "seed": seed,
            "max_iterations": 400,
            "status": "completed",
            "friction_range": friction,
            "noise_scale": scale,
            "observation_noise": {"effective_std": noise},
            "controller_profile": profile,
            "final_research": {
                "study": final.FINAL_RESEARCH_ID,
                "method": run["method"],
                "run_name": "staged",
                "template_checkpoint_sha256": "synthetic-template",
                "fresh_optimizers": True,
                "initial_iteration": 0,
                "initial_actor_sha256": f"synthetic-actor-{seed}",
                "initial_critic_sha256": f"synthetic-critic-{seed}",
                "reward_schedule": final.reward_schedule(),
                "stages": stages,
                "transition": {"iteration": 200, "mode": "same_runner"},
            },
        }
        checkpoint = tmp_path / "logs" / name / "staged" / "model_400.pt"
        checkpoint.parent.mkdir(parents=True)
        torch.save({"iter": 400, "controller_profile": profile}, checkpoint)
        checkpoint.with_name("friction_experiment.json").write_text(
            json.dumps(training)
        )

        path = folder / "metadata.json"
        meta = json.loads(path.read_text())
        eval_env = deepcopy(env)
        eval_env["seed"] = final.EVAL_SEEDS[0]
        eval_env["domain_randomization"]["startup"]["contact_friction_range"] = [
            0.2,
            0.2,
        ]
        meta.update(
            checkpoint=str(checkpoint),
            checkpoint_iteration=400,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            training=training,
            controller_profile=profile,
            training_stage="paired-contact",
            env_config=eval_env,
            runner_config=config | {"seed": final.EVAL_SEEDS[0], "log_dir": None},
            controller_telemetry={"coordination_diagnostics": {"distal": True}},
            force_frame="world",
        )
        if experiments[name] == original:
            meta["merge_sources"] = [{"dataset": "synthetic-original"}]
            rows = read_csv(folder / "trials.csv")
            for row in rows:
                row["source_trial_id"] = row["trial_id"]
            write_csv(folder / "trials.csv", list(rows[0]), rows)
        path.write_text(json.dumps(meta))
    return original, supplemental


def test_disjoint_policy_merge_preserves_metadata_and_traces(tmp_path, staged_sources):
    original, supplemental = staged_sources
    output = tmp_path / "merged"
    with pytest.raises(ValueError, match="Finalized training conditions are missing"):
        merge_evaluations(original, supplemental, output, require_final_matrix=True)
    assert merge_evaluations(original, supplemental, output, dry_run=True) == (300, 300)
    assert not output.exists()
    assert merge_evaluations(original, supplemental, output) == (300, 300)

    checkpoints = set()
    for source in staged_sources:
        for file in source.rglob("*"):
            if file.is_file():
                assert (
                    output / file.relative_to(source)
                ).read_bytes() == file.read_bytes()
        [path] = source.rglob("metadata.json")
        dataset = output / path.parent.relative_to(source)
        meta, trials = read_complete_evaluation(dataset)
        checkpoints.add(meta["checkpoint_sha256"])
        assert len(trials) == 300
    assert len(checkpoints) == 2


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("checkpoint_sha256", "0" * 64, "checkpoint hash"),
        ("controller_profile", {}, "Controller profile differs"),
        ("training", {}, "Training provenance differs"),
        ("force_frame", "body", "Evaluation settings differ.*force_frame"),
    ],
)
def test_disjoint_policy_merge_rejects_mismatched_metadata(
    tmp_path, staged_sources, field, value, message
):
    original, supplemental = staged_sources
    [path] = supplemental.rglob("metadata.json")
    meta = json.loads(path.read_text())
    meta[field] = value
    path.write_text(json.dumps(meta))
    output = tmp_path / "merged"
    with pytest.raises(ValueError, match=message):
        merge_evaluations(original, supplemental, output)
    assert not output.exists()


def test_disjoint_policy_merge_rejects_duplicate_trials(tmp_path, staged_sources):
    original, supplemental = staged_sources
    [index] = supplemental.rglob("trials.csv")
    rows = read_csv(index)
    write_csv(index, list(rows[0]), rows + rows[-1:])
    with pytest.raises(ValueError, match="Duplicate trial"):
        merge_evaluations(original, supplemental, tmp_path / "merged")
