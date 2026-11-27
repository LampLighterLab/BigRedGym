"""Analytical tests for Step 3 metrics without simulator or training calls."""

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import (
    InvalidTrace,
    Thresholds,
    analyze_trace,
    assess_readiness,
    critical_friction,
    main,
    summarize_condition,
)


SCHEDULE = ((1, (0, 0, 0)), (2, (2, 0, 0)), (3, (0, 0, 0)))
DT = 0.1


def synthetic_trace():
    commands = np.repeat([stage[1] for stage in SCHEDULE], [1, 2, 3], axis=0)
    body = np.zeros((6, 3))
    body[:, 0] = [0, 1, 2, 0.4, 0.1, 0]
    body[1, 1] = 0.3
    body[:, 2] = 9
    world = body[:, [1, 0, 2]]
    position = np.zeros((6, 3))
    position[:, :2] = [
        [0, 0],
        [0.1, 0],
        [0.3, 0],
        [0.33, 0.04],
        [0.33, 0.05],
        [0.33, 0.05],
    ]
    orientation = np.zeros((6, 3))
    orientation[:, 1] = [0, 0.1, -0.3, 0.2, 0, 0]
    force = np.zeros((6, 4, 3))
    force[:, 0, :] = [
        [1, 0, 10],
        [3, 4, 10],
        [7, 0, 10],
        [4, 0, 10],
        [2, 0, 10],
        [1, 0, 10],
    ]
    # Swing, unloaded, and exactly-at-threshold feet have large tangential forces.
    force[:, 1:4, :] = [[90, 0, 10], [90, 0, 4], [90, 0, 5]]
    contact = np.broadcast_to([True, False, True, True], (6, 4)).copy()
    foot_velocity = np.zeros_like(force)
    foot_velocity[:, 0, 0] = [0.1, 0.11, 0.1, 0, 0.2, 0]
    foot_velocity[:, 0, 2] = 9
    foot_velocity[:, 1:4, 0] = 10
    return {
        "time_s": DT * np.arange(1, 7),
        "stage_index": np.array([0, 1, 1, 2, 2, 2]),
        "command": commands,
        "base_linear_velocity_body_m_s": body,
        "base_angular_velocity_body_rad_s": np.zeros((6, 3)),
        "base_linear_velocity_world_m_s": world,
        "base_position_world_m": position,
        "base_roll_pitch_yaw_rad": orientation,
        "foot_ground_force_world_N": force,
        "foot_contact": contact,
        "foot_velocity_world_m_s": foot_velocity,
        **{
            name: np.zeros(6, dtype=bool)
            for name in ("fell", "terminated", "timed_out", "numerical_failure")
        },
    }


def score(trace, friction=0.5, outcome="completed", cfg=None):
    return analyze_trace(
        trace, SCHEDULE, DT, 4, friction, outcome, cfg or Thresholds(), primary=True
    )


def test_per_foot_ratios_phase_percentiles_cone_utilization_and_slip():
    result, arrays = score(synthetic_trace())
    np.testing.assert_allclose(
        arrays["grf_ratio"][:, 0], [0.1, 0.5, 0.7, 0.4, 0.2, 0.1]
    )
    assert np.isnan(arrays["grf_ratio"][:, 1:]).all()
    assert arrays["stance_mask"].sum() == result["stance_samples"] == 6
    np.testing.assert_array_equal(arrays["acceleration_mask"], [0, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(arrays["braking_mask"], [0, 0, 0, 1, 1, 1])
    assert result["acceleration_stance_samples"] == 2
    assert result["braking_stance_samples"] == 3
    assert result["acceleration_grf_ratio_p95"] == pytest.approx(0.69)
    assert result["acceleration_grf_ratio_peak"] == pytest.approx(0.7)
    assert result["braking_grf_ratio_p95"] == pytest.approx(0.38)
    assert result["braking_grf_ratio_peak"] == pytest.approx(0.4)
    assert result["acceleration_cone_utilization_p95"] == pytest.approx(1.38)
    assert result["acceleration_cone_utilization_peak"] == pytest.approx(1.4)
    assert result["slipping_samples"] == 2
    assert result["foot_slip_rate"] == pytest.approx(2 / 6)


def test_tracking_rise_time_stopping_path_length_and_pitch():
    result, _ = score(synthetic_trace())
    assert result["tracking_rmse_m_s"] == pytest.approx(np.sqrt(1.26 / 6))
    assert result["moving_tracking_rmse_m_s"] == pytest.approx(np.sqrt(1.09 / 2))
    assert result["yaw_tracking_rmse_rad_s"] == 0
    assert result["rise_time_s"] == pytest.approx(0.2)
    assert result["stopping_distance_m"] == pytest.approx(0.06)
    assert result["max_pitch_excursion_rad"] == pytest.approx(0.3)
    assert (
        result["survived"] and result["tracking_passed"] and result["stopping_passed"]
    )
    assert result["success"] is True


def test_stopping_window_uses_sustained_stop_and_rejects_rolling_finish():
    trace = synthetic_trace()
    trace["base_linear_velocity_world_m_s"][3:, :2] = [[0.05, 0], [0.25, 0], [0.1, 0]]
    trace["base_position_world_m"][-1, :2] = [0.36, 0.05]
    result, _ = score(trace)
    assert result["stopping_distance_m"] == pytest.approx(0.09)
    trace["base_linear_velocity_world_m_s"][-1, :2] = [0.2, 0]
    result, _ = score(trace)
    assert result["stopping_passed"] is False and result["success"] is False
    assert result["stopping_distance_m"] is None


def test_motionless_policy_fails_directional_tracking_and_optional_yaw_gate():
    trace = synthetic_trace()
    trace["base_linear_velocity_body_m_s"][:, :2] = 0
    result, _ = score(trace, cfg=Thresholds(max_moving_rmse_m_s=3))
    assert result["moving_tracking_rmse_m_s"] == 2
    assert result["tracking_passed"] is False and result["success"] is False
    trace = synthetic_trace()
    trace["base_angular_velocity_body_rad_s"][:, 2] = 0.3
    result, _ = score(trace, cfg=Thresholds(max_yaw_rmse_rad_s=0.2))
    assert result["yaw_tracking_rmse_rad_s"] == pytest.approx(0.3)
    assert result["success"] is False


@pytest.mark.parametrize("friction", [0.0, None])
def test_zero_or_missing_friction_leaves_force_ratio_and_omits_utilization(friction):
    result, arrays = score(synthetic_trace(), friction=friction)
    assert result["grf_ratio_peak"] == pytest.approx(0.7)
    assert result["cone_utilization_p95"] is None
    assert np.isnan(arrays["cone_utilization"]).all()


def test_missing_signals_remain_unavailable_without_erasing_other_metrics():
    trace = synthetic_trace()
    del trace["foot_velocity_world_m_s"]
    trace["foot_ground_force_world_N"][2, 0, 0] = np.nan
    result, arrays = score(trace)
    assert result["data_status"] == "partial"
    assert "foot_velocity_world_m_s missing" in result["issues"]
    assert np.isnan(arrays["grf_ratio"][2, 0])
    assert result["stance_samples"] == 5
    assert result["slip_eligible_samples"] == 0 and result["foot_slip_rate"] is None
    assert result["success"] is True
    del trace["base_linear_velocity_world_m_s"]
    result, _ = score(trace)
    assert result["survived"] is True and result["stopping_passed"] is None
    assert result["success"] is None


@pytest.mark.parametrize("status,flag", [("fall", "fell"), ("timeout", "timed_out")])
def test_fall_and_timeout_preserve_valid_prefix_but_cannot_pass(status, flag):
    trace = {key: values[:4] for key, values in synthetic_trace().items()}
    trace[flag][-1] = True
    result, arrays = score(trace, outcome=status)
    assert result["valid_steps"] == 4
    assert result["rise_time_s"] == pytest.approx(0.2)
    assert result["stopping_distance_m"] is None
    assert result["success"] is False and result["survived"] is False
    assert arrays["stance_mask"][-1, 0]


def test_solver_recovery_frame_is_excluded_even_with_finite_telemetry():
    trace = {key: values[:4] for key, values in synthetic_trace().items()}
    trace["numerical_failure"][-1] = True
    trace["base_roll_pitch_yaw_rad"][-1, 1] = 1.5
    trace["foot_ground_force_world_N"][-1, 0, 0] = 500
    result, arrays = score(trace, outcome="numerical_failure")
    assert result["success"] is False and result["valid_steps"] == 3
    assert result["max_pitch_excursion_rad"] == pytest.approx(0.3)
    assert result["braking_grf_ratio_peak"] is None
    assert not arrays["valid_step"][-1]


def test_truncated_completed_trace_and_missing_flags_remain_unscored():
    prefix = {key: values[:5] for key, values in synthetic_trace().items()}
    result, _ = score(prefix)
    assert result["survived"] is None and result["success"] is None
    assert result["data_status"] == "partial"
    trace = synthetic_trace()
    del trace["timed_out"]
    result, _ = score(trace)
    assert result["success"] is None and "timed_out missing" in result["issues"]


def test_misaligned_clock_is_rejected():
    trace = synthetic_trace()
    trace["time_s"][2] += 0.02
    with pytest.raises(InvalidTrace, match="clock"):
        score(trace)


@pytest.mark.parametrize("direction", [(-1, 0, 0), (0, 1, 0)])
def test_direction_change_is_both_phase_windows_without_a_stop_requirement(direction):
    trace = synthetic_trace()
    schedule = ((1, (0, 0, 0)), (2, (1, 0, 0)), (3, direction))
    command = np.repeat([stage[1] for stage in schedule], [1, 2, 3], axis=0)
    trace["command"] = command
    trace["base_linear_velocity_body_m_s"] = command.astype(float)
    result, arrays = analyze_trace(
        trace, schedule, DT, 4, 0.5, "completed", Thresholds()
    )
    np.testing.assert_array_equal(arrays["acceleration_mask"], [0, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(arrays["braking_mask"], [0, 0, 0, 1, 1, 1])
    assert result["tracking_rmse_m_s"] == 0
    assert result["stop_required"] is False and result["success"] is True


def test_critical_friction_uses_nine_of_ten_scored_primary_trials():
    def at(mu, success, maneuver="stand_2_stop"):
        return summarize_condition([{"success": item} for item in success]) | {
            "friction": mu,
            "maneuver": maneuver,
        }

    rows = [
        at(0.1, [True] * 9),
        at(0.2, [True] * 9 + [None]),
        at(0.3, [True] * 8 + [False] * 2),
        at(0.4, [True] * 9 + [False]),
        at(0.5, [True] * 10),
        at(0.05, [True] * 10, "steady_2p0"),
    ]
    assert critical_friction(rows) == {
        "critical_friction": 0.4,
        "eligible_conditions": 3,
        "status": "qualified",
    }
    assert critical_friction(rows[:2])["status"] == "insufficient_trials"
    assert critical_friction(rows[:3])["status"] == "no_condition_passed"
    assert rows[1]["success_rate"] is None and rows[1]["unscored_trials"] == 1


def test_readiness_requires_explicit_rubric_and_available_metrics():
    row = {"success_rate": 0.9, "mean_rise_time_s": 0.3}
    assert assess_readiness(row, 0.4, {})[0] == "not_assessed"
    criteria = {"minimum": {"success_rate": 0.9}, "maximum": {"critical_friction": 0.4}}
    assert assess_readiness(row, 0.4, criteria)[0] == "meets_thresholds"
    assert assess_readiness(row, 0.6, criteria)[0] == "outside_thresholds"
    assert assess_readiness(row, None, criteria)[0] == "not_assessed"


def test_cli_preserves_input_and_indexes_missing_trials_and_traces(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    traces = dataset / "traces"
    traces.mkdir()
    meta = {
        "protocol": "go2trot-friction-eval-v1",
        "checkpoint_sha256": "synthetic",
        "dt_s": DT,
        "foot_order": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        "maneuvers": {
            "stand_2_stop": [
                {"steps": steps, "command": command} for steps, command in SCHEDULE
            ]
        },
        "frictions": [0.5],
        "noise_scales": [0.0],
        "trial_seeds": [42, 43, 44],
    }
    metadata = dataset / "metadata.json"
    metadata.write_text(json.dumps(meta))
    archive = traces / "trial_000000.npz"
    np.savez(archive, **synthetic_trace())
    trial_index = dataset / "trials.csv"
    rows = [
        {
            "trial_id": f"trial_{idx:06d}",
            "eval_seed": seed,
            "maneuver": "stand_2_stop",
            "friction": 0.5,
            "applied_friction": 0.5,
            "noise_scale": 0.0,
            "status": "completed",
            "planned_steps": 6,
            "recorded_steps": 6,
            "trace": f"traces/trial_{idx:06d}.npz",
        }
        for idx, seed in enumerate((42, 43))
    ]
    with trial_index.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    raw = {path: path.read_bytes() for path in (metadata, archive, trial_index)}
    output = tmp_path / "metrics"
    main([str(dataset), "--output-dir", str(output), "--slip-speed-m-s", "0.15"])
    assert {path: path.read_bytes() for path in raw} == raw
    assert not (dataset / "per_trial.csv").exists()
    with (output / "per_trial.csv").open(newline="") as fh:
        trials = list(csv.DictReader(fh))
    assert [trial["success"] for trial in trials] == ["True", "", ""]
    assert trials[0]["foot_slip_rate"] == str(1 / 6)
    assert trials[1]["data_status"] == "unavailable" and trials[1]["issues"]
    assert trials[2]["trial_status"] == "missing_trial"
    with np.load(output / trials[0]["derived_trace"], allow_pickle=False) as derived:
        assert derived["grf_ratio"].shape == (6, 4)
    with (output / "summary.csv").open(newline="") as fh:
        condition = next(csv.DictReader(fh))
    assert condition["expected_trials"] == "3"
    assert condition["scored_trials"] == "1" and condition["success_rate"] == ""
    assert condition["hardware_readiness_status"] == "not_assessed"
    with (output / "critical_friction.csv").open(newline="") as fh:
        assert next(csv.DictReader(fh))["status"] == "insufficient_trials"
    report = json.loads((output / "metrics_metadata.json").read_text())
    assert report["thresholds"]["slip_speed_m_s"] == 0.15
