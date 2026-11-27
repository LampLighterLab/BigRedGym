"""Summarize saved Go2Trot Step 3 metrics across independent training seeds."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, stdev

from final_protocol import FINAL_RESEARCH_ID, FINAL_RESEARCH_IDS, reward_schedule
from metrics import METRICS, PRIMARY_MANEUVER, write_csv


PROTOCOL = "go2trot-friction-analysis-v1"
DEFAULT_COST = "tracking_rmse_m_s"
GRF_METRICS = tuple(metric for metric in METRICS if "grf_ratio" in metric)
EVALUATION = (
    "maneuver",
    "test_noise_scale",
    "evaluation_profile",
    "checkpoint_iteration",
)
CONFIGURATION = (
    "method",
    "training_friction_min",
    "training_friction_max",
    "training_noise_scale",
)
CONDITION_COUNTS = (
    "expected_trials",
    "scored_trials",
    "successes",
    "failures",
    "unscored_trials",
)
SEED = (
    *EVALUATION,
    *CONFIGURATION,
    "training_seed",
    "training_protocol",
    "experiment_name",
    "dataset",
    "checkpoint",
    "checkpoint_sha256",
    "selected",
    "selection_status",
)
CONDITION_METRICS = ("success_rate", *METRICS)
NOMINAL_METRICS = ("critical_friction", "nominal_cost", *METRICS)
MATCH_METRICS = (
    "nominal_cost",
    "tracking_rmse_m_s",
    "rise_time_s",
    "stopping_distance_m",
    "foot_slip_rate",
    *GRF_METRICS,
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def number(value):
    if value is None or value == "":
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def subset(row, keys):
    return {key: row[key] for key in keys}


def grouped(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups.values()


def statistics(rows, fields):
    result = {}
    for field in fields:
        values = [row[field] for row in rows if row[field] is not None]
        result[f"mean_{field}"] = fmean(values) if values else None
        result[f"std_{field}"] = stdev(values) if len(values) > 1 else None
        result[f"n_{field}"] = len(values)
    return result


def stats_fields(fields):
    return tuple(
        prefix + field for field in fields for prefix in ("mean_", "std_", "n_")
    )


def dataset_details(name, raw, source):
    training = raw.get("training") or {}
    final = training.get("final_research", {})
    label = training.get("experiment_name") or ""
    method = "unknown"
    if final.get("study") in FINAL_RESEARCH_IDS:
        method = final["method"]
    elif label.startswith("go2friction_lb_"):
        method = "friction"
    elif label.startswith("go2noise_scale_"):
        method = "noise"
    bounds = training.get("friction_range") or (
        source.get("training_friction_min"),
        None,
    )
    return {
        "dataset": name,
        "method": method,
        "training_friction_min": number(bounds[0]),
        "training_friction_max": number(bounds[1]),
        "training_noise_scale": number(
            training.get("noise_scale", source.get("training_noise_scale"))
        ),
        "training_seed": training.get("seed", source.get("training_seed")),
        "training_protocol": final.get("study"),
        "experiment_name": label,
        "checkpoint": raw.get("checkpoint"),
        "checkpoint_iteration": raw.get("checkpoint_iteration"),
        "checkpoint_sha256": source.get("checkpoint_sha256"),
    }


def evaluation_profile(raw, dataset, maneuver, scale, profiles):
    # Paths, seeds, and batching are excluded from the evaluation design.
    environment = raw.get("env_config", {})
    protocol_environment = {
        key: environment[key]
        for key in (
            "asset",
            "control",
            "domain_randomization",
            "init_state",
            "mujoco",
            "push_robots",
            "scaling",
            "sim_dt",
            "terrain",
        )
        if key in environment
    }
    randomization = protocol_environment.get("domain_randomization")
    if randomization is not None and "startup" in randomization:
        # The evaluator uses the first grid value to enable the backend setter.
        # Each measured condition sets its own friction before reset.
        protocol_environment["domain_randomization"] = randomization | {
            "startup": {
                key: value
                for key, value in randomization["startup"].items()
                if key != "contact_friction_range"
            }
        }
    settings = {
        "backend": raw.get("backend"),
        "mujoco_version": raw.get("mujoco_version"),
        "dt_s": raw.get("dt_s"),
        "gait_frequency_hz": raw.get("gait_frequency_hz"),
        "stages": raw.get("maneuvers", {}).get(maneuver),
        "test_friction_grid": sorted(map(float, raw.get("frictions", ()))),
        "test_noise_scale": scale,
        "noise_std_physical": next(
            (
                values
                for key, values in raw.get("noise_std_physical", {}).items()
                if float(key) == scale
            ),
            None,
        ),
        "foot_order": raw.get("foot_order"),
        "environment": protocol_environment,
    }
    final = (raw.get("training") or {}).get("final_research", {})
    if final.get("study") in FINAL_RESEARCH_IDS:
        settings["final_research"] = {
            key: final[key]
            for key in ("study", "run_name", "template_checkpoint_sha256")
        }
        if final["study"] == FINAL_RESEARCH_ID:
            if final.get("reward_schedule") != reward_schedule():
                raise ValueError(f"Invalid staged reward schedule in {dataset}")
            settings["final_research"]["reward_schedule"] = final["reward_schedule"]
    if settings["stages"] is None or not settings["test_friction_grid"]:
        settings["unverified_dataset"] = dataset
    encoded = json.dumps(settings, sort_keys=True).encode()
    profile = hashlib.sha256(encoded).hexdigest()[:12]
    profiles.setdefault(profile, settings)
    return profile


def trial_key(row):
    return (
        row["dataset"],
        row["maneuver"],
        float(row["noise_scale"]),
        float(row["friction"]),
    )


def comparable(run):
    return run["method"] in ("friction", "noise") and all(
        run[key] is not None for key in (*CONFIGURATION[1:], "training_seed")
    )


def select_runs(runs):
    """Choose one evaluation per training seed/configuration in each stratum."""
    for candidates in grouped(
        filter(comparable, runs),
        (*EVALUATION, *CONFIGURATION, "training_seed"),
    ):

        def rank(run):
            conditions = run["conditions"]
            complete = sum(row["success_rate"] is not None for row in conditions)
            scored = sum(row["scored_trials"] for row in conditions)
            return -complete, -scored, run["dataset"]

        winner = min(candidates, key=rank)
        for run in candidates:
            run["selected"] = run is winner
            run["selection_status"] = (
                "selected" if run is winner else "duplicate_training_seed"
            )


def full_horizon(trial):
    return trial["survived"] == "True" and int(trial["valid_steps"] or 0) == int(
        trial["planned_steps"]
    )


def nominal_row(run, trials, cutoff, nominal_friction, cost_metric, primary_maneuver):
    planned = [
        trial for trial in trials if float(trial["friction"]) == nominal_friction
    ]
    completed = [trial for trial in planned if full_horizon(trial)]
    row = subset(run, SEED) | {
        "nominal_friction": nominal_friction,
        "nominal_expected_trials": len(planned),
        "nominal_full_horizon_trials": len(completed),
        "nominal_eval_seeds": ";".join(trial["eval_seed"] for trial in planned),
        "critical_friction": number(cutoff.get("critical_friction")),
        "critical_friction_status": cutoff.get(
            "status",
            "missing_cutoff"
            if run["maneuver"] == primary_maneuver
            else "not_applicable",
        ),
        "critical_eligible_conditions": number(cutoff.get("eligible_conditions")),
    }
    row["unscored_lower_friction_conditions"] = (
        sum(
            condition["expected_trials"] != 10 or condition["scored_trials"] != 10
            for condition in run["conditions"]
            if condition["test_friction"] < row["critical_friction"]
        )
        if row["critical_friction"] is not None
        else None
    )
    for metric in METRICS:
        values = [
            value
            for trial in completed
            if (value := number(trial[metric])) is not None
            and (
                metric != DEFAULT_COST
                or int(trial["tracking_samples"] or 0) == int(trial["planned_steps"])
            )
        ]
        row[metric] = fmean(values) if values else None
        row[f"n_trials_{metric}"] = len(values)
    row["nominal_cost"] = row[cost_metric]
    row["nominal_status"] = (
        "not_scheduled"
        if not planned
        else "no_full_horizon_trials"
        if not completed
        else "undefined_cost"
        if row["nominal_cost"] is None
        else "available"
    )
    return row


def aggregate_conditions(rows):
    result = []
    for group in grouped(rows, (*EVALUATION, *CONFIGURATION, "test_friction")):
        result.append(
            subset(group[0], (*EVALUATION, *CONFIGURATION, "test_friction"))
            | {"n_training_seeds": len(group)}
            | {key: sum(row[key] for row in group) for key in CONDITION_COUNTS}
            | statistics(group, CONDITION_METRICS)
        )
    return result


def aggregate_nominal(rows):
    result = []
    for group in grouped(rows, (*EVALUATION, *CONFIGURATION)):
        paired = [
            {
                "paired_nominal_cost": row["nominal_cost"],
                "paired_critical_friction": row["critical_friction"],
            }
            for row in group
            if row["critical_friction"] is not None and row["nominal_cost"] is not None
        ]
        result.append(
            subset(group[0], (*EVALUATION, *CONFIGURATION, "nominal_friction"))
            | {
                "n_training_seeds": len(group),
                "n_seeds_with_unscored_lower_friction": sum(
                    bool(row["unscored_lower_friction_conditions"]) for row in group
                ),
            }
            | statistics(group, NOMINAL_METRICS)
            | statistics(paired, ("paired_nominal_cost", "paired_critical_friction"))
        )
    return result


def equivalent_noise(rows, scope, tolerance):
    matches = []
    for group in grouped(rows, EVALUATION):
        friction = [row for row in group if row["method"] == "friction"]
        noise = [
            row
            for row in group
            if row["method"] == "noise" and row["critical_friction"] is not None
        ]
        for target in friction:
            cutoff = target["critical_friction"]
            candidate = None
            difference = None
            if cutoff is None:
                status = "undefined_critical_friction"
            elif not noise:
                status = "no_noise_candidate"
            else:
                candidate = min(
                    noise,
                    key=lambda row: (
                        round(abs(row["critical_friction"] - cutoff), 12),
                        row.get("training_seed") != target.get("training_seed"),
                        row["training_noise_scale"],
                        row.get("training_seed") or 0,
                        row["training_friction_min"],
                        row["training_friction_max"],
                    ),
                )
                difference = candidate["critical_friction"] - cutoff
                status = (
                    "matched"
                    if abs(difference) <= tolerance
                    or math.isclose(
                        abs(difference), tolerance, rel_tol=0, abs_tol=1e-12
                    )
                    else "outside_tolerance"
                )
            match = subset(target, (*EVALUATION, "nominal_friction")) | {
                "scope": scope,
                "status": status,
                "friction_training_seed": target.get("training_seed"),
                "friction_training_min": target["training_friction_min"],
                "friction_training_max": target["training_friction_max"],
                "friction_training_noise_scale": target["training_noise_scale"],
                "friction_critical_friction": cutoff,
                "friction_n_critical_seeds": target.get(
                    "n_critical_friction", int(cutoff is not None)
                ),
                "candidate_training_seed": candidate.get("training_seed")
                if candidate
                else None,
                "candidate_noise_scale": candidate["training_noise_scale"]
                if candidate
                else None,
                "candidate_training_friction_min": candidate["training_friction_min"]
                if candidate
                else None,
                "candidate_training_friction_max": candidate["training_friction_max"]
                if candidate
                else None,
                "candidate_critical_friction": candidate["critical_friction"]
                if candidate
                else None,
                "candidate_n_critical_seeds": candidate.get("n_critical_friction", 1)
                if candidate
                else 0,
                "equivalent_noise_scale": candidate["training_noise_scale"]
                if status == "matched"
                else None,
                "critical_friction_difference": difference,
                "critical_friction_absolute_difference": abs(difference)
                if difference is not None
                else None,
            }
            for metric in MATCH_METRICS:
                left = target[metric]
                right = candidate[metric] if candidate else None
                match[f"friction_{metric}"] = left
                match[f"candidate_{metric}"] = right
                match[f"{metric}_difference"] = (
                    right - left if left is not None and right is not None else None
                )
                match[f"friction_n_{metric}"] = target[f"n_{metric}"]
                match[f"candidate_n_{metric}"] = (
                    candidate[f"n_{metric}"] if candidate else 0
                )
            matches.append(match)
    return matches


def run_analysis(
    metrics_dir,
    output,
    nominal_friction=1.25,
    cost_metric=DEFAULT_COST,
    match_tolerance=0.05,
):
    metrics_dir, output = Path(metrics_dir).resolve(), Path(output).resolve()
    for name, value in (
        ("Nominal friction", nominal_friction),
        ("Matching tolerance", match_tolerance),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if cost_metric not in METRICS:
        raise ValueError(f"Unknown cost metric {cost_metric!r}")

    metadata = read_json(metrics_dir / "metrics_metadata.json")
    if metadata.get("protocol") != "go2trot-friction-metrics-v1":
        raise ValueError("Expected a Step 3 metrics directory")
    primary_maneuver = metadata.get("primary_maneuver", PRIMARY_MANEUVER)
    sources = {row["dataset"]: row for row in metadata["sources"]}
    raw = {}
    for name in sources:
        path = Path(name) / "metadata.json"
        raw[name] = read_json(path) if path.is_file() else {}
    kinds = {source.get("synthetic", False) for source in raw.values() if source}
    if len(kinds) > 1:
        raise ValueError(
            "Analyze synthetic and recorded evaluations in separate directories"
        )
    kind = ("synthetic" if True in kinds else "recorded") if kinds else "unverified"

    trials = defaultdict(list)
    for row in read_csv(metrics_dir / "per_trial.csv"):
        trials[trial_key(row)].append(row)
    critical = {
        (row["dataset"], float(row["noise_scale"])): row
        for row in read_csv(metrics_dir / "critical_friction.csv")
    }
    runs = {}
    profiles = {}
    identities = {
        dataset: dataset_details(dataset, raw[dataset], source)
        for dataset, source in sources.items()
    }
    for row in read_csv(metrics_dir / "summary.csv"):
        dataset, maneuver, scale, friction = trial_key(row)
        ident = identities[dataset]
        # One physical baseline dataset appears at the reference point of both curves.
        methods = (
            ("friction", "noise")
            if ident["method"] == "baseline"
            else (ident["method"],)
        )
        recorded = trials[trial_key(row)]
        condition = (
            {
                "test_friction": friction,
                "eval_seeds": ";".join(trial["eval_seed"] for trial in recorded),
            }
            | {key: int(row[key]) for key in CONDITION_COUNTS}
            | {"success_rate": number(row["success_rate"])}
            | {metric: number(row[f"mean_{metric}"]) for metric in METRICS}
            | {f"n_trials_{metric}": int(row[f"n_{metric}"]) for metric in METRICS}
        )
        for method in methods:
            run_key = (dataset, maneuver, scale, method)
            if run_key not in runs:
                runs[run_key] = ident | {
                    "method": method,
                    "maneuver": maneuver,
                    "test_noise_scale": scale,
                    "evaluation_profile": evaluation_profile(
                        raw[dataset], dataset, maneuver, scale, profiles
                    ),
                    "selected": False,
                    "selection_status": "unknown_training_provenance",
                    "conditions": [],
                }
            runs[run_key]["conditions"].append(condition)
    select_runs(runs.values())

    seed_conditions, seed_nominal, traces = [], [], []
    for (dataset, maneuver, scale, _), run in runs.items():
        run_trials = []
        for condition in run["conditions"]:
            seed_conditions.append(subset(run, SEED) | condition)
            for trial in trials[dataset, maneuver, scale, condition["test_friction"]]:
                run_trials.append(trial)
                trace_id = f"trace_{len(traces):06d}"
                traces.append(
                    subset(run, SEED)
                    | subset(trial, TRACE_INPUT_FIELDS)
                    | {
                        "trace_id": trace_id,
                        "test_friction": condition["test_friction"],
                        "raw_trace": str(Path(dataset) / trial["source_trace"])
                        if trial["source_trace"]
                        else None,
                        "derived_trace": str(metrics_dir / trial["derived_trace"])
                        if trial["derived_trace"]
                        else None,
                    }
                )
        cutoff = (
            critical.get((dataset, scale), {}) if maneuver == primary_maneuver else {}
        )
        seed_nominal.append(
            nominal_row(
                run, run_trials, cutoff, nominal_friction, cost_metric, primary_maneuver
            )
        )

    selected_nominal = [row for row in seed_nominal if row["selected"]]
    conditions = aggregate_conditions(row for row in seed_conditions if row["selected"])
    nominal = aggregate_nominal(selected_nominal)
    comparison = equivalent_noise(
        (
            row
            | {
                f"n_{metric}": row.get(
                    f"n_trials_{cost_metric if metric == 'nominal_cost' else metric}", 0
                )
                for metric in MATCH_METRICS
            }
            for row in selected_nominal
            if row["maneuver"] == primary_maneuver
        ),
        "seed",
        match_tolerance,
    )
    comparison.extend(
        equivalent_noise(
            (
                row
                | {
                    metric: row[f"mean_{metric}"]
                    for metric in ("critical_friction", *MATCH_METRICS)
                }
                for row in nominal
                if row["maneuver"] == primary_maneuver
            ),
            "configuration",
            match_tolerance,
        )
    )

    output.mkdir(parents=True, exist_ok=False)
    tables = (
        ("seed_conditions.csv", SEED_CONDITION_FIELDS, seed_conditions),
        ("condition_summary.csv", CONDITION_SUMMARY_FIELDS, conditions),
        ("seed_nominal.csv", SEED_NOMINAL_FIELDS, seed_nominal),
        ("nominal_summary.csv", NOMINAL_SUMMARY_FIELDS, nominal),
        ("equivalent_noise.csv", MATCH_FIELDS, comparison),
        ("traces.csv", TRACE_FIELDS, traces),
    )
    for name, fields, rows in tables:
        write_csv(
            output / name,
            ("data_kind", *fields),
            (row | {"data_kind": kind} for row in rows),
        )
    report = {
        "protocol": PROTOCOL,
        "data_kind": kind,
        "source_metrics": str(metrics_dir),
        "nominal_friction": nominal_friction,
        "cost_metric": cost_metric,
        "primary_maneuver": primary_maneuver,
        "match_tolerance": match_tolerance,
        "thresholds": metadata.get("thresholds"),
        "critical_friction_rule": metadata.get("critical_friction_rule"),
        "evaluation_profiles": profiles,
        "dataset_metadata_unavailable": [
            path for path, data in raw.items() if not data
        ],
        "runs": [subset(run, SEED) for run in runs.values()],
        "table_rows": {name: len(rows) for name, _, rows in tables},
    }
    (output / "analysis_metadata.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


SEED_CONDITION_FIELDS = (
    *SEED,
    "test_friction",
    "eval_seeds",
    *CONDITION_COUNTS,
    *CONDITION_METRICS,
    *(f"n_trials_{metric}" for metric in METRICS),
)
CONDITION_SUMMARY_FIELDS = (
    *EVALUATION,
    *CONFIGURATION,
    "test_friction",
    "n_training_seeds",
    *CONDITION_COUNTS,
    *stats_fields(CONDITION_METRICS),
)
SEED_NOMINAL_FIELDS = (
    *SEED,
    "nominal_friction",
    "nominal_expected_trials",
    "nominal_full_horizon_trials",
    "nominal_eval_seeds",
    "nominal_status",
    "critical_friction_status",
    "critical_eligible_conditions",
    "unscored_lower_friction_conditions",
    *NOMINAL_METRICS,
    *(f"n_trials_{metric}" for metric in METRICS),
)
NOMINAL_SUMMARY_FIELDS = (
    *EVALUATION,
    *CONFIGURATION,
    "nominal_friction",
    "n_training_seeds",
    "n_seeds_with_unscored_lower_friction",
    *stats_fields(NOMINAL_METRICS),
    *stats_fields(("paired_nominal_cost", "paired_critical_friction")),
)
MATCH_FIELDS = (
    *EVALUATION,
    "nominal_friction",
    "scope",
    "status",
    "friction_training_seed",
    "friction_training_min",
    "friction_training_max",
    "friction_training_noise_scale",
    "friction_critical_friction",
    "friction_n_critical_seeds",
    "candidate_training_seed",
    "candidate_noise_scale",
    "candidate_training_friction_min",
    "candidate_training_friction_max",
    "candidate_critical_friction",
    "candidate_n_critical_seeds",
    "equivalent_noise_scale",
    "critical_friction_difference",
    "critical_friction_absolute_difference",
    *(
        field
        for metric in MATCH_METRICS
        for field in (
            f"friction_{metric}",
            f"candidate_{metric}",
            f"{metric}_difference",
            f"friction_n_{metric}",
            f"candidate_n_{metric}",
        )
    ),
)
TRACE_INPUT_FIELDS = (
    "applied_friction",
    "eval_seed",
    "trial_id",
    "trial_status",
    "data_status",
    "issues",
    "survived",
    "success",
    "planned_steps",
    "valid_steps",
)
TRACE_FIELDS = (
    *SEED,
    "trace_id",
    "test_friction",
    *TRACE_INPUT_FIELDS,
    "raw_trace",
    "derived_trace",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics_dir", type=Path, help="Output directory from metrics.py"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory for analysis CSVs"
    )
    parser.add_argument("--nominal-friction", type=float, default=1.25)
    parser.add_argument("--cost-metric", choices=METRICS, default=DEFAULT_COST)
    parser.add_argument("--match-tolerance", type=float, default=0.05)
    args = parser.parse_args(argv)
    try:
        report = run_analysis(
            args.metrics_dir,
            args.output_dir,
            args.nominal_friction,
            args.cost_metric,
            args.match_tolerance,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Saved {report['data_kind']} analysis to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
