"""Compute offline performance metrics from Go2Trot Step 2 datasets."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from itertools import product
import json
import math
import operator
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


PROTOCOL = "go2trot-friction-eval-v1"
PRIMARY_MANEUVER = "stand_2_stop"
FAILURE_OUTCOMES = {"fall", "timeout", "numerical_failure"}
TERMINAL_FLAGS = ("fell", "terminated", "timed_out", "numerical_failure")
FORCE_METRICS = (
    "grf_ratio_p95",
    "grf_ratio_peak",
    "cone_utilization_p95",
    "cone_utilization_peak",
)
METRICS = (
    *(
        prefix + name
        for prefix in ("", "acceleration_", "braking_")
        for name in FORCE_METRICS
    ),
    "foot_slip_rate",
    "tracking_rmse_m_s",
    "moving_tracking_rmse_m_s",
    "yaw_tracking_rmse_rad_s",
    "rise_time_s",
    "stopping_distance_m",
    "max_pitch_excursion_rad",
)
TRIAL_FIELDS = (
    "dataset",
    "checkpoint_sha256",
    "training_friction_min",
    "training_noise_scale",
    "training_seed",
    "maneuver",
    "friction",
    "applied_friction",
    "noise_scale",
    "eval_seed",
    "trial_id",
    "trial_status",
    "source_trace",
    "derived_trace",
    "planned_steps",
    "trace_steps",
    "valid_steps",
    "data_status",
    "issues",
    "survived",
    "tracking_passed",
    "stop_required",
    "stopping_passed",
    "success",
    "stance_samples",
    "acceleration_stance_samples",
    "braking_stance_samples",
    "slip_eligible_samples",
    "slipping_samples",
    "tracking_samples",
    *METRICS,
)
CONDITION_FIELDS = (
    "dataset",
    "checkpoint_sha256",
    "training_friction_min",
    "training_noise_scale",
    "training_seed",
    "maneuver",
    "friction",
    "noise_scale",
    "expected_trials",
    "scored_trials",
    "successes",
    "failures",
    "unscored_trials",
    "success_rate",
    *(prefix + metric for metric in METRICS for prefix in ("mean_", "n_")),
    "hardware_readiness_status",
    "hardware_readiness_details",
)
CRITICAL_FIELDS = (
    "dataset",
    "checkpoint_sha256",
    "training_friction_min",
    "training_noise_scale",
    "training_seed",
    "noise_scale",
    "eligible_conditions",
    "critical_friction",
    "status",
)


@dataclass(frozen=True)
class Thresholds:
    min_normal_force_n: float = 5.0
    slip_speed_m_s: float = 0.1
    moving_command_m_s: float = 0.01
    max_moving_rmse_m_s: float = 0.75
    min_stage_speed_fraction: float = 0.5
    rise_fraction: float = 0.9
    stop_speed_m_s: float = 0.15
    max_yaw_rmse_rad_s: float | None = None

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            not 0 < self.rise_fraction <= 1
            or not 0 < self.min_stage_speed_fraction <= 1
        ):
            raise ValueError("Rise and stage speed fractions must be in (0, 1]")


class InvalidTrace(ValueError):
    """The trace cannot be aligned to its Step 2 schedule."""


def read_series(trace, name, shape, issues, *, boolean=False):
    values = trace.get(name)
    kind = "b" if boolean else "fiu"
    if values is None or values.shape != shape or values.dtype.kind not in kind:
        issues.append(f"{name} missing or invalid")
        return np.zeros(shape, dtype=bool) if boolean else np.full(shape, np.nan)
    if boolean:
        return values
    if not np.isfinite(values).all():
        issues.append(f"{name} contains nonfinite samples")
    return np.asarray(values, dtype=np.float64)


def aligned_samples(trace, steps, commands, dt):
    time = trace.get("time_s")
    if time is None or time.ndim != 1 or time.dtype.kind not in "fiu" or not len(time):
        raise InvalidTrace("time_s must be a nonempty numeric vector")
    count = len(time)
    if count > steps.sum():
        raise InvalidTrace("trace is longer than its scheduled horizon")
    stages = np.repeat(np.arange(len(steps)), steps)[:count]
    if not np.array_equal(trace.get("stage_index"), stages):
        raise InvalidTrace("stage_index does not match the scheduled prefix")
    sampled_commands = trace.get("command")
    if (
        sampled_commands is None
        or sampled_commands.shape != (count, 3)
        or sampled_commands.dtype.kind not in "fiu"
        or not np.allclose(sampled_commands, commands[stages], rtol=1e-6, atol=1e-7)
    ):
        raise InvalidTrace("command does not match the scheduled prefix")
    if not np.allclose(time, dt * np.arange(1, count + 1), atol=1e-8, rtol=1e-9):
        raise InvalidTrace("time_s does not match the post-step control clock")
    return time, stages


def command_phases(commands, speed_threshold):
    planar = commands[:, :2]
    speed = np.linalg.norm(planar, axis=1)
    moving = speed > speed_threshold
    directions = np.divide(
        planar, speed[:, None], out=np.zeros_like(planar), where=speed[:, None] > 0
    )
    accelerate = np.zeros(len(commands), dtype=bool)
    brake = accelerate.copy()
    # A new direction can simultaneously accelerate and brake relative to the
    # new and old command projections. The first stage has no recorded step input.
    accelerate[1:] = moving[1:] & (
        speed[1:] - (planar[:-1] * directions[1:]).sum(axis=1) > speed_threshold
    )
    brake[1:] = moving[:-1] & (
        speed[:-1] - (planar[1:] * directions[:-1]).sum(axis=1) > speed_threshold
    )
    stop_start = None
    if moving.any() and not moving[-1]:
        stop_start = int(np.flatnonzero(moving)[-1]) + 1
        brake[stop_start:] = True
    return moving, directions, speed, accelerate, brake, stop_start


def stance_metrics(force, contact, foot_velocity, physical, friction, cfg):
    loaded = contact & physical[:, None] & np.isfinite(force).all(axis=2)
    loaded &= force[..., 2] > cfg.min_normal_force_n
    ratios = np.full(contact.shape, np.nan)
    ratios[loaded] = (
        np.hypot(force[..., 0][loaded], force[..., 1][loaded]) / force[..., 2][loaded]
    )
    utilization = (
        ratios / friction
        if friction is not None and friction > 0
        else np.full_like(ratios, np.nan)
    )
    slip_eligible = loaded & np.isfinite(foot_velocity[..., :2]).all(axis=2)
    foot_speed = np.hypot(foot_velocity[..., 0], foot_velocity[..., 1])
    slipping = slip_eligible & (foot_speed > cfg.slip_speed_m_s)
    return {
        "stance_mask": loaded,
        "grf_ratio": ratios,
        "cone_utilization": utilization,
        "slip_eligible": slip_eligible,
        "slipping": slipping,
    }


def p95_and_peak(values):
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None, None
    return float(np.percentile(finite, 95, method="linear")), float(finite.max())


def rmse(error, mask):
    selected = error[mask]
    return float(np.sqrt(np.mean(selected**2))) if selected.size else None


def terminal_stop(speed, position, start, planned, physical, cfg):
    # Use the last moving sample as the position at the stop-command onset.
    # A stop must hold through the last sample of the zero-command window.
    if len(speed) != planned or not physical[start:].all():
        return None, None
    window = speed[start:]
    if not np.isfinite(window).all():
        return None, None
    if window[-1] > cfg.stop_speed_m_s:
        return False, None
    above = np.flatnonzero(window > cfg.stop_speed_m_s)
    stop = start + (above[-1] + 1 if above.size else 0)
    points = position[start - 1 : stop + 1, :2]
    distance = None
    if np.isfinite(points).all():
        segments = np.diff(points, axis=0)
        distance = float(np.hypot(segments[:, 0], segments[:, 1]).sum())
    return True, distance


def analyze_trace(
    trace, schedule, dt, foot_count, friction, outcome, cfg, *, primary=False
):
    """Return per-trial metrics and time-by-foot arrays for one Step 2 trace."""
    steps = np.array([stage[0] for stage in schedule])
    commands = np.array([stage[1] for stage in schedule], dtype=float)
    time, stages = aligned_samples(trace, steps, commands, dt)
    count, planned = len(time), int(steps.sum())
    issues = []
    flags = np.stack(
        [
            read_series(trace, flag, (count,), issues, boolean=True)
            for flag in TERMINAL_FLAGS
        ]
    )
    flags_available = not issues
    ended = np.flatnonzero(flags.any(axis=0))
    limit = int(ended[0]) + 1 if ended.size else count
    # A MuJoCo-recovered failure frame may contain finite reset-state telemetry.
    numerical_limit = flags[-1, limit - 1] or outcome == "numerical_failure"
    physical = np.arange(count) < (limit - int(numerical_limit))
    failed = outcome in FAILURE_OUTCOMES or bool(ended.size)
    complete = (
        outcome == "completed" and count == planned and flags_available and not failed
    )
    survived = False if failed else True if complete else None
    if outcome == "completed" and not complete:
        issues.append("completed outcome conflicts with trace or terminal flags")
    if outcome != "completed" and outcome not in FAILURE_OUTCOMES:
        issues.append("unknown trial outcome")
    if ended.size and limit != count:
        issues.append("ignored samples after first terminal frame")
    if friction is None:
        issues.append("applied friction missing")

    force = read_series(
        trace, "foot_ground_force_world_N", (count, foot_count, 3), issues
    )
    contact = read_series(
        trace, "foot_contact", (count, foot_count), issues, boolean=True
    )
    foot_velocity = read_series(
        trace, "foot_velocity_world_m_s", (count, foot_count, 3), issues
    )
    metrics = stance_metrics(force, contact, foot_velocity, physical, friction, cfg)
    moving, directions, command_speed, accel, brake, stop_start = command_phases(
        commands, cfg.moving_command_m_s
    )
    phases = {"": physical, "acceleration_": accel[stages], "braking_": brake[stages]}
    summary = {}
    for prefix, mask in phases.items():
        summary[prefix + "stance_samples"] = int(metrics["stance_mask"][mask].sum())
        for key in ("grf_ratio", "cone_utilization"):
            p95, peak = p95_and_peak(metrics[key][mask])
            summary[prefix + key + "_p95"] = p95
            summary[prefix + key + "_peak"] = peak
    slip_samples = int(metrics["slipping"].sum())
    slip_total = int(metrics["slip_eligible"].sum())
    summary.update(
        foot_slip_rate=slip_samples / slip_total if slip_total else None,
        slip_eligible_samples=slip_total,
        slipping_samples=slip_samples,
    )

    body = read_series(trace, "base_linear_velocity_body_m_s", (count, 3), issues)
    angular = read_series(trace, "base_angular_velocity_body_rad_s", (count, 3), issues)
    world = read_series(trace, "base_linear_velocity_world_m_s", (count, 3), issues)
    position = read_series(trace, "base_position_world_m", (count, 3), issues)
    orientation = read_series(trace, "base_roll_pitch_yaw_rad", (count, 3), issues)
    valid_velocity = physical & np.isfinite(body[:, :2]).all(axis=1)
    valid_yaw = physical & np.isfinite(angular[:, 2])
    valid_pitch = physical & np.isfinite(orientation[:, 1])
    tracking_error = np.linalg.norm(body[:, :2] - commands[stages, :2], axis=1)
    moving_rmse = rmse(tracking_error, valid_velocity & moving[stages])
    yaw_rmse = rmse(angular[:, 2] - commands[stages, 2], valid_yaw)
    summary.update(
        tracking_samples=int(valid_velocity.sum()),
        tracking_rmse_m_s=rmse(tracking_error, valid_velocity),
        moving_tracking_rmse_m_s=moving_rmse,
        yaw_tracking_rmse_rad_s=yaw_rmse,
        max_pitch_excursion_rad=float(np.abs(orientation[valid_pitch, 1]).max())
        if valid_pitch.any()
        else None,
    )
    projected = np.sum(body[:, :2] * directions[stages], axis=1)
    acceleration_stages = np.flatnonzero(accel)
    rise_time = None
    if acceleration_stages.size:
        stage = acceleration_stages[0]
        reached = np.flatnonzero(
            (stages == stage)
            & valid_velocity
            & (projected >= cfg.rise_fraction * command_speed[stage])
        )
        if reached.size:
            rise_time = float(time[reached[0]] - steps[:stage].sum() * dt)

    tracking_passed = None
    if not moving.any():
        tracking_passed = False
        issues.append("maneuver has no moving command")
    elif complete and valid_velocity[moving[stages]].all():
        tracking_passed = moving_rmse <= cfg.max_moving_rmse_m_s and all(
            projected[stages == stage].mean()
            >= cfg.min_stage_speed_fraction * command_speed[stage]
            for stage in np.flatnonzero(moving)
        )
        if cfg.max_yaw_rmse_rad_s is not None:
            tracking_passed = (
                bool(tracking_passed and yaw_rmse <= cfg.max_yaw_rmse_rad_s)
                if valid_yaw.all()
                else None
            )

    stop_required = primary or stop_start is not None
    stopping_passed, stopping_distance = None, None
    if stop_start is not None:
        stopping_passed, stopping_distance = terminal_stop(
            np.hypot(world[:, 0], world[:, 1]),
            position,
            int(steps[:stop_start].sum()),
            planned,
            physical,
            cfg,
        )
    elif primary:
        stopping_passed = False
        issues.append("primary maneuver lacks a terminal stop command")

    success = None
    if survived is False:
        success = False
    elif (
        complete
        and tracking_passed is not None
        and (not stop_required or stopping_passed is not None)
    ):
        success = bool(tracking_passed and (not stop_required or stopping_passed))

    summary.update(
        trace_steps=count,
        valid_steps=int(physical.sum()),
        data_status="partial" if issues else "valid",
        issues="; ".join(issues),
        survived=survived,
        tracking_passed=tracking_passed,
        stop_required=stop_required,
        stopping_passed=stopping_passed,
        success=success,
        rise_time_s=rise_time,
        stopping_distance_m=stopping_distance,
    )
    return summary, {
        "time_s": time,
        "stage_index": stages,
        "valid_step": physical,
        "acceleration_mask": physical & accel[stages],
        "braking_mask": physical & brake[stages],
        **metrics,
    }


def positive_schedule(stages):
    schedule = []
    for stage in stages:
        steps = stage["steps"]
        command = tuple(stage["command"])
        if type(steps) is not int or steps <= 0 or len(command) != 3:
            raise ValueError("Invalid Step 2 stage")
        if any(not math.isfinite(x) for x in command):
            raise ValueError("Nonfinite Step 2 stage command")
        schedule.append((steps, command))
    if not schedule:
        raise ValueError("Empty Step 2 schedule")
    return schedule


def load_series(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def trial_inputs(row, dataset, schedule, dt, foot_count, cfg, *, primary):
    if row is None:
        return {
            "trial_status": "missing_trial",
            "data_status": "unavailable",
            "issues": "trial absent from trials.csv",
        }, None
    outcome = row.get("status") or "unknown"
    source_trace = row.get("trace")
    result = {
        "trial_id": row.get("trial_id"),
        "trial_status": outcome,
        "source_trace": source_trace,
        "survived": False if outcome in FAILURE_OUTCOMES else None,
    }
    try:
        if not source_trace:
            raise InvalidTrace("trace path missing from trials.csv")
        path = (dataset / source_trace).resolve()
        if not path.is_relative_to(dataset) or path.suffix != ".npz":
            raise InvalidTrace("trace must be an NPZ within the dataset")
        applied = row.get("applied_friction")
        friction = float(applied) if applied else None
        if friction is not None and (not math.isfinite(friction) or friction < 0):
            raise InvalidTrace("applied friction must be finite and nonnegative")
        result["applied_friction"] = friction
        trace = load_series(path)
        if int(row["planned_steps"]) != sum(steps for steps, _ in schedule):
            raise InvalidTrace("planned_steps does not match metadata")
        if int(row["recorded_steps"]) != trace.get("time_s", np.empty(0)).size:
            raise InvalidTrace("recorded_steps does not match the trace")
        summary, derived = analyze_trace(
            trace,
            schedule,
            dt,
            foot_count,
            friction,
            outcome,
            cfg,
            primary=primary,
        )
    except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile) as exc:
        result.update(data_status="unavailable", issues=str(exc))
        return result, None
    return result | summary, derived


def summarize_condition(trials):
    expected = len(trials)
    scored = sum(row.get("success") is not None for row in trials)
    successes = sum(row.get("success") is True for row in trials)
    result = {
        "expected_trials": expected,
        "scored_trials": scored,
        "successes": successes,
        "failures": scored - successes,
        "unscored_trials": expected - scored,
        "success_rate": successes / expected if scored == expected else None,
    }
    for metric in METRICS:
        values = [row[metric] for row in trials if row.get(metric) is not None]
        result[f"mean_{metric}"] = float(np.mean(values)) if values else None
        result[f"n_{metric}"] = len(values)
    return result


def critical_friction(conditions, primary_maneuver=PRIMARY_MANEUVER):
    eligible = [
        row
        for row in conditions
        if row["maneuver"] == primary_maneuver
        and row["expected_trials"] == row["scored_trials"] == 10
    ]
    passed = [row["friction"] for row in eligible if row["successes"] >= 9]
    return {
        "eligible_conditions": len(eligible),
        "critical_friction": min(passed) if passed else None,
        "status": "qualified"
        if passed
        else "no_condition_passed"
        if eligible
        else "insufficient_trials",
    }


def assess_readiness(condition, critical, criteria):
    if not criteria:
        return "not_assessed", "No proposal thresholds supplied"
    values = condition | {"critical_friction": critical}
    missing, failed = [], []
    for rule, predicate in (("minimum", operator.lt), ("maximum", operator.gt)):
        for metric, limit in criteria.get(rule, {}).items():
            value = values[metric]
            if value is None:
                missing.append(metric)
            elif predicate(float(value), limit):
                failed.append(f"{metric}={value:g} violates {rule} {limit:g}")
    if missing:
        return "not_assessed", "Missing " + ", ".join(sorted(set(missing)))
    return (
        ("outside_thresholds", "; ".join(failed))
        if failed
        else ("meets_thresholds", "")
    )


def read_criteria(path):
    if path is None:
        return {}
    rules = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rules, dict) or not rules.keys() <= {"minimum", "maximum"}:
        raise ValueError("Readiness rubric needs minimum and/or maximum mappings")
    available = {
        "success_rate",
        "scored_trials",
        "successes",
        "failures",
        "critical_friction",
        *(f"mean_{metric}" for metric in METRICS),
    }
    for thresholds in rules.values():
        if not isinstance(thresholds, dict) or not thresholds.keys() <= available:
            raise ValueError("Readiness rubric names an unsupported metric")
        if any(
            type(x) not in (int, float) or not math.isfinite(x)
            for x in thresholds.values()
        ):
            raise ValueError("Readiness limits must be finite numbers")
    return {rule: thresholds for rule, thresholds in rules.items() if thresholds}


def read_dataset(dataset):
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    if metadata["protocol"] != PROTOCOL:
        raise ValueError(f"Unsupported dataset protocol in {dataset}")
    dt = float(metadata["dt_s"])
    if not math.isfinite(dt) or dt <= 0 or not metadata["foot_order"]:
        raise ValueError("Invalid control interval or foot ordering")
    schedules = {
        name: positive_schedule(stages)
        for name, stages in metadata["maneuvers"].items()
    }
    frictions = [float(x) for x in metadata["frictions"]]
    scales = [float(x) for x in metadata["noise_scales"]]
    seeds = metadata["trial_seeds"]
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Invalid Step 2 trial seed")
    for values in (frictions, scales, seeds):
        if (
            not values
            or len(set(values)) != len(values)
            or any(not math.isfinite(x) or x < 0 for x in values)
        ):
            raise ValueError("Invalid or repeated Step 2 conditions or seeds")
    rows = {}
    index = dataset / "trials.csv"
    if index.is_file():
        with index.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = (
                    row["maneuver"],
                    float(row["friction"]),
                    float(row["noise_scale"]),
                    int(row["eval_seed"]),
                )
                if key in rows:
                    raise ValueError(f"Duplicate trial in {dataset}: {key}")
                if not (
                    key[0] in schedules
                    and key[1] in frictions
                    and key[2] in scales
                    and key[3] in seeds
                ):
                    raise ValueError(
                        f"Trial outside the planned grid in {dataset}: {key}"
                    )
                rows[key] = row
    return metadata, dt, schedules, frictions, scales, seeds, rows


def dataset_identity(metadata, dataset):
    training = metadata.get("training") or {}
    bounds = training.get("friction_range") or [None]
    return {
        "dataset": str(dataset),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "training_friction_min": bounds[0],
        "training_noise_scale": training.get("noise_scale"),
        "training_seed": training.get("seed"),
    }


def evaluate_dataset(
    dataset, output, dataset_index, cfg, criteria, trial_writer, *, primary_maneuver
):
    meta, dt, schedules, frictions, scales, seeds, rows = read_dataset(dataset)
    identity = dataset_identity(meta, dataset)
    condition_rows = []
    derived_dir = output / "derived" / f"dataset_{dataset_index:04d}"
    derived_dir.mkdir(parents=True)
    ordinal = 0
    for name, friction, scale in product(schedules, frictions, scales):
        condition = identity | {
            "maneuver": name,
            "friction": friction,
            "noise_scale": scale,
        }
        trials = []
        schedule = schedules[name]
        for seed in seeds:
            result, derived = trial_inputs(
                rows.get((name, friction, scale, seed)),
                dataset,
                schedule,
                dt,
                len(meta["foot_order"]),
                cfg,
                primary=name == primary_maneuver,
            )
            if derived is not None:
                path = derived_dir / f"trial_{ordinal:06d}.npz"
                np.savez_compressed(path, **derived)
                result["derived_trace"] = path.relative_to(output).as_posix()
            trial_writer.writerow(
                condition
                | {
                    "eval_seed": seed,
                    "planned_steps": sum(steps for steps, _ in schedule),
                }
                | result
            )
            trials.append(result)
            ordinal += 1
        condition_rows.append(condition | summarize_condition(trials))

    critical_rows = []
    for scale in scales:
        selected = [row for row in condition_rows if row["noise_scale"] == scale]
        critical = critical_friction(selected, primary_maneuver)
        critical_rows.append(identity | {"noise_scale": scale} | critical)
        for row in selected:
            status, details = assess_readiness(
                row, critical["critical_friction"], criteria
            )
            row.update(
                hardware_readiness_status=status, hardware_readiness_details=details
            )
    return (
        condition_rows,
        critical_rows,
        identity
        | {
            "foot_order": meta["foot_order"],
            "control_dt_s": dt,
            "source_status": meta.get("status"),
            "indexed_trials": len(rows),
            "expected_trials": ordinal,
        },
    )


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_metrics(input_dir, output, cfg, criteria, *, primary_maneuver=PRIMARY_MANEUVER):
    input_dir, output = Path(input_dir).resolve(), Path(output).resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Dataset directory does not exist: {input_dir}")
    if (input_dir / "metadata.json").is_file():
        datasets = [input_dir]
    else:
        datasets = sorted(
            {path.parent.resolve() for path in input_dir.rglob("metadata.json")}
        )
    if not datasets:
        raise ValueError("No Step 2 datasets found")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "per_trial.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRIAL_FIELDS)
        writer.writeheader()
        summaries, critical, sources = [], [], []
        for index, dataset in enumerate(datasets):
            conditions, cutoffs, source = evaluate_dataset(
                dataset,
                output,
                index,
                cfg,
                criteria,
                writer,
                primary_maneuver=primary_maneuver,
            )
            summaries.extend(conditions)
            critical.extend(cutoffs)
            sources.append(source)
    write_csv(output / "summary.csv", CONDITION_FIELDS, summaries)
    write_csv(output / "critical_friction.csv", CRITICAL_FIELDS, critical)
    report = {
        "protocol": "go2trot-friction-metrics-v1",
        "sources": sources,
        "thresholds": asdict(cfg),
        "hardware_readiness_rubric": criteria or None,
        "primary_maneuver": primary_maneuver,
        "critical_friction_rule": (
            "Lowest tested coefficient with at least 9 successes "
            "in 10 scored primary trials"
        ),
    }
    (output / "metrics_metadata.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return len(sources)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="Step 2 dataset or sweep output directory"
    )
    parser.add_argument("--output-dir", type=Path, help="New metrics directory")
    parser.add_argument(
        "--primary-maneuver",
        default=PRIMARY_MANEUVER,
        help="Maneuver used for the stopping requirement and critical friction",
    )
    parser.add_argument(
        "--readiness-rubric", type=Path, help="JSON limits from the proposal"
    )
    defaults = Thresholds()
    for name, value in asdict(defaults).items():
        parser.add_argument(
            "--" + name.replace("_", "-"), dest=name, type=float, default=value
        )
    args = parser.parse_args(argv)
    source = args.input.resolve()
    output = (args.output_dir or source.with_name(source.name + "_metrics")).resolve()
    try:
        cfg = Thresholds(**{field: getattr(args, field) for field in asdict(defaults)})
        criteria = read_criteria(args.readiness_rubric)
        count = run_metrics(
            source, output, cfg, criteria, primary_maneuver=args.primary_maneuver
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Scored {count} dataset(s) in {output}")


if __name__ == "__main__":
    main()
