"""Check offline Step 4 statistics, matching, and figures with known inputs."""

import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_results import (
    METRICS,
    aggregate_conditions,
    aggregate_nominal,
    evaluation_profile,
    read_csv,
    run_analysis,
    select_runs,
)
from metrics import Thresholds, run_metrics
from plot_results import (
    configuration_style,
    grf_history,
    load_trace,
    phase_windows,
    pitch_history,
    plt,
    run_plots,
    tradeoff_figure,
    velocity_history,
)
from synthetic_step4 import DT, generate


def table(output, name, **filters):
    return [
        row
        for row in read_csv(output / f"{name}.csv")
        if all(row[key] == str(value) for key, value in filters.items())
    ]


def hashes(directory):
    return {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.fixture(scope="module")
def offline(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_step4")
    raw, metrics, output = (
        root / name for name in ("evaluation", "metrics", "analysis")
    )
    generate(raw)
    raw_hashes = hashes(raw)
    assert run_metrics(raw, metrics, Thresholds(), {}) == 12
    metric_hashes = hashes(metrics)
    report = run_analysis(metrics, output)
    assert hashes(raw) == raw_hashes
    assert hashes(metrics) == metric_hashes
    return output, report


def test_seed_means_sample_deviation_and_noise_maneuver_separation(offline):
    output, report = offline
    assert report["data_kind"] == "synthetic"
    filters = {
        "maneuver": "stand_2_stop",
        "method": "friction",
        "training_friction_min": 0.4,
    }
    first, second = sorted(
        table(
            output,
            "seed_conditions",
            **filters,
            test_noise_scale=0.0,
            test_friction=0.4,
        ),
        key=lambda row: int(row["training_seed"]),
    )
    assert first["training_seed"] == "10"
    assert second["training_seed"] == "20"
    assert first["eval_seeds"].split(";") == [str(seed) for seed in range(100, 110)]
    assert (float(first["success_rate"]), float(second["success_rate"])) == (0.9, 0.4)
    [aggregate] = table(
        output,
        "condition_summary",
        **filters,
        test_noise_scale=0.0,
        test_friction=0.4,
    )
    assert aggregate["n_training_seeds"] == aggregate["n_success_rate"] == "2"
    assert aggregate["scored_trials"] == "20"
    assert float(aggregate["mean_success_rate"]) == pytest.approx(0.65)
    assert float(aggregate["std_success_rate"]) == pytest.approx(np.sqrt(0.125))

    [incomplete] = table(
        output,
        "condition_summary",
        **filters,
        test_noise_scale=1.0,
        test_friction=0.2,
    )
    assert incomplete["unscored_trials"] == "1"
    assert incomplete["n_training_seeds"] == "2"
    assert incomplete["n_success_rate"] == "1"
    assert float(incomplete["mean_success_rate"]) == pytest.approx(0.4)
    assert incomplete["std_success_rate"] == ""
    [undefined] = table(
        output,
        "seed_conditions",
        **filters,
        test_noise_scale=1.0,
        test_friction=0.2,
        training_seed=20,
    )
    assert undefined["success_rate"] == ""
    assert float(undefined["grf_ratio_peak"]) > 0
    [unqualified] = table(
        output,
        "seed_nominal",
        maneuver="stand_2_stop",
        test_noise_scale=1.0,
        method="noise",
        training_noise_scale=2.0,
        training_seed=20,
    )
    assert unqualified["critical_friction_status"] == "no_condition_passed"
    assert unqualified["critical_friction"] == unqualified["nominal_cost"] == ""
    assert unqualified["nominal_status"] == "no_full_horizon_trials"
    [other_maneuver] = table(
        output,
        "seed_nominal",
        maneuver="steady_1p0",
        test_noise_scale=0.0,
        method="friction",
        training_friction_min=0.4,
        training_seed=10,
    )
    assert other_maneuver["critical_friction_status"] == "not_applicable"
    assert other_maneuver["stopping_distance_m"] == ""


def test_equivalent_noise_uses_tested_candidates_and_tolerance(offline):
    output, _ = offline
    noise = table(
        output,
        "nominal_summary",
        maneuver="stand_2_stop",
        test_noise_scale=0.0,
        method="noise",
    )
    cutoffs = {
        float(row["training_noise_scale"]): float(row["mean_critical_friction"])
        for row in noise
    }
    assert cutoffs == pytest.approx({0.0: 1.025, 0.5: 0.6, 2.0: 0.8})

    matches = table(
        output, "equivalent_noise", test_noise_scale=0.0, scope="configuration"
    )
    by_min = {float(row["friction_training_min"]): row for row in matches}
    assert float(by_min[0.4]["equivalent_noise_scale"]) == 0.5
    assert float(by_min[0.8]["equivalent_noise_scale"]) == 0
    assert float(by_min[0.4]["critical_friction_difference"]) == pytest.approx(0)
    for metric in (
        "nominal_cost",
        "acceleration_grf_ratio_p95",
        "braking_grf_ratio_peak",
    ):
        match = by_min[0.4]
        assert float(match[f"{metric}_difference"]) == pytest.approx(
            float(match[f"candidate_{metric}"]) - float(match[f"friction_{metric}"])
        )
        assert match[f"friction_n_{metric}"] == match[f"candidate_n_{metric}"] == "2"
    rejected = by_min[1.25]
    assert rejected["status"] == "outside_tolerance"
    assert rejected["equivalent_noise_scale"] == ""
    assert float(rejected["candidate_noise_scale"]) == 0.5
    assert float(rejected["critical_friction_absolute_difference"]) == pytest.approx(
        0.4
    )
    [seed_match] = table(
        output,
        "equivalent_noise",
        test_noise_scale=0.0,
        scope="seed",
        friction_training_min=0.4,
        friction_training_seed=20,
    )
    assert seed_match["candidate_training_seed"] == "20"
    assert float(seed_match["equivalent_noise_scale"]) == 0.5


def test_matching_tolerance_includes_the_boundary(offline, tmp_path):
    _, report = offline
    output = tmp_path / "higher_tolerance"
    run_analysis(report["source_metrics"], output, match_tolerance=0.4)
    [match] = table(
        output,
        "equivalent_noise",
        test_noise_scale=0.0,
        scope="configuration",
        friction_training_min=1.25,
    )
    assert match["status"] == "matched"
    assert float(match["equivalent_noise_scale"]) == 0.5


def test_training_seed_weighting_and_tradeoff_pairing():
    anchor = {
        "maneuver": "stand_2_stop",
        "test_noise_scale": 0,
        "evaluation_profile": "abc",
        "method": "friction",
        "training_friction_min": 0.4,
        "training_friction_max": 1.25,
        "training_noise_scale": 0,
        "checkpoint_iteration": 100,
        "test_friction": 1.25,
        "nominal_friction": 1.25,
        "unscored_lower_friction_conditions": 0,
        **dict.fromkeys(METRICS),
    }
    short = anchor | {
        "training_seed": 10,
        "success_rate": 1,
        "expected_trials": 2,
        "scored_trials": 2,
        "successes": 2,
        "failures": 0,
        "unscored_trials": 0,
    }
    long = anchor | {
        "training_seed": 20,
        "success_rate": 0.1,
        "expected_trials": 10,
        "scored_trials": 10,
        "successes": 1,
        "failures": 9,
        "unscored_trials": 0,
    }
    [condition] = aggregate_conditions([short, long])
    assert condition["mean_success_rate"] == pytest.approx(0.55)
    assert condition["successes"] == 3 and condition["scored_trials"] == 12

    [nominal] = aggregate_nominal(
        [
            short | {"nominal_cost": 0.25, "critical_friction": None},
            long | {"nominal_cost": 0.5, "critical_friction": 0.8},
        ]
    )
    assert nominal["mean_nominal_cost"] == 0.375
    assert nominal["mean_paired_nominal_cost"] == 0.5
    assert nominal["mean_paired_critical_friction"] == 0.8
    assert nominal["n_paired_nominal_cost"] == 1
    assert nominal["std_paired_nominal_cost"] is None


def test_evaluation_profiles_and_duplicate_selection():
    metadata = {
        "dt_s": 0.02,
        "frictions": [1.25, 0.4],
        "trial_seeds": [100],
        "maneuvers": {"stand_2_stop": [{"steps": 3, "command": [0, 0, 0]}]},
        "env_config": {
            "domain_randomization": {
                "startup": {"contact_friction_range": [1.25, 1.25]}
            }
        },
    }
    profiles = {}
    first = evaluation_profile(metadata, "dataset1", "stand_2_stop", 0, profiles)
    repeated = evaluation_profile(
        metadata
        | {
            "frictions": [0.4, 1.25],
            "trial_seeds": [101, 102],
            "env_config": {
                "domain_randomization": {
                    "startup": {"contact_friction_range": [0.4, 0.4]}
                }
            },
        },
        "dataset2",
        "stand_2_stop",
        0,
        profiles,
    )
    different = evaluation_profile(
        metadata | {"frictions": [0.4, 0.8, 1.25]},
        "dataset3",
        "stand_2_stop",
        0,
        profiles,
    )
    assert first == repeated and different != first
    assert metadata["env_config"]["domain_randomization"]["startup"] == {
        "contact_friction_range": [1.25, 1.25]
    }

    anchor = {
        "maneuver": "stand_2_stop",
        "test_noise_scale": 0,
        "evaluation_profile": first,
        "method": "friction",
        "training_friction_min": 0.4,
        "training_friction_max": 1.25,
        "training_noise_scale": 0,
        "training_seed": 10,
        "checkpoint_iteration": 100,
    }
    missing = anchor | {
        "dataset": "a",
        "conditions": [{"success_rate": None, "scored_trials": 9}],
    }
    finished = anchor | {
        "dataset": "b",
        "conditions": [{"success_rate": 0, "scored_trials": 10}],
    }
    select_runs([missing, finished])
    assert (
        not missing["selected"]
        and missing["selection_status"] == "duplicate_training_seed"
    )
    assert finished["selected"]


def test_separates_synthetic_and_recorded_inputs(tmp_path):
    sources = []
    for folder, synthetic in (("recorded", False), ("synthetic", True)):
        source = tmp_path / folder
        source.mkdir()
        (source / "metadata.json").write_text(json.dumps({"synthetic": synthetic}))
        sources.append({"dataset": str(source)})
    (tmp_path / "metrics_metadata.json").write_text(
        json.dumps(
            {
                "protocol": "go2trot-friction-metrics-v1",
                "sources": sources,
            }
        )
    )
    with pytest.raises(ValueError, match="separate directories"):
        run_analysis(tmp_path, tmp_path / "analysis")
    assert not (tmp_path / "analysis").exists()


def test_failed_trace_retains_prefix_and_uses_applied_friction(offline):
    output, report = offline
    row = table(
        output,
        "traces",
        maneuver="stand_2_stop",
        test_noise_scale=0.0,
        method="friction",
        trial_status="fall",
        test_friction=0.2,
    )[0]
    assert row["success"] == "False"
    trace = load_trace(row, report["evaluation_profiles"][row["evaluation_profile"]])
    np.testing.assert_allclose(trace["time_s"], DT * np.arange(1, 8))
    assert trace["grf_ratio"].shape == (7, 4)
    assert len(phase_windows(trace["edges"], trace["braking_mask"])) == 0
    fig = grf_history(trace, row | {"applied_friction": "0.21"}, "SYNTHETIC")
    limits = [
        line
        for line in fig.axes[0].lines
        if line.get_label().startswith("Applied friction")
    ]
    [limit] = limits
    np.testing.assert_allclose(limit.get_ydata(), [0.21, 0.21])
    assert fig.axes[1].texts[0].get_text() == "No recorded braking samples"
    plt.close(fig)


def test_velocity_and_pitch_use_logged_body_signals_and_control_clock(offline):
    output, report = offline
    row = table(
        output,
        "traces",
        maneuver="stand_2_stop",
        test_noise_scale=0.0,
        trial_status="completed",
        test_friction=1.25,
    )[0]
    trace = load_trace(row, report["evaluation_profiles"][row["evaluation_profile"]])
    fig = velocity_history(trace, "SYNTHETIC")
    [command] = fig.axes[0].patches
    np.testing.assert_allclose(command.get_data().edges, np.arange(23) * DT)
    np.testing.assert_array_equal(
        command.get_data().values, [0] * 4 + [2] * 10 + [0] * 8
    )
    [measured] = fig.axes[0].lines
    np.testing.assert_allclose(measured.get_xdata(), DT * np.arange(1, 23))
    np.testing.assert_array_equal(
        measured.get_ydata(), trace["base_linear_velocity_body_m_s"][:, 0]
    )
    plt.close(fig)

    fig = pitch_history(trace, "SYNTHETIC")
    pitch = fig.axes[0].lines[0].get_ydata()
    np.testing.assert_allclose(
        pitch, np.rad2deg(trace["base_roll_pitch_yaw_rad"][:, 1])
    )
    assert pitch[4] > 0 and pitch[14] < 0
    np.testing.assert_allclose(
        phase_windows(trace["edges"], trace["braking_mask"]), [(14 * DT, 22 * DT)]
    )
    plt.close(fig)


def test_all_figure_pairs_and_tradeoff_axes(offline, tmp_path):
    output, _ = offline
    filters = {"maneuver": "stand_2_stop", "test_noise_scale": 0.0}
    summary = table(output, "nominal_summary", **filters)
    seeds = table(output, "seed_nominal", **filters)
    chart = tradeoff_figure(
        summary, seeds, configuration_style(summary), "tracking_rmse_m_s", "SYNTHETIC"
    )
    [axis] = chart.axes
    assert "Nominal cost" in axis.get_xlabel()
    assert axis.get_ylabel() == "Critical friction"
    _, labels = axis.get_legend_handles_labels()
    assert any(label.startswith("Friction [") for label in labels)
    assert any(label.startswith("Noise ×") for label in labels)
    assert "SYNTHETIC" in chart.texts[0].get_text()
    plt.close(chart)

    output_figures = tmp_path / "synthetic_figures"
    manifest = run_plots(output, output_figures, **filters, dpi=90)
    assert manifest["data_kind"] == "synthetic"
    [group] = manifest["groups"]
    assert group["trace_status"] == "completed"
    assert group["trace_issue"] is None
    assert len(group["figures"]) == 13
    folder = output_figures / group["directory"]
    for name in group["figures"]:
        png = (folder / f"{name}.png").read_bytes()
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert all(size > 250 for size in struct.unpack(">II", png[16:24]))
        assert (folder / f"{name}.pdf").read_bytes().startswith(b"%PDF-")
