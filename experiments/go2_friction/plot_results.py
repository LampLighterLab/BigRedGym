"""Render publication figures from offline Go2Trot Step 4 analysis tables."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import re
from zipfile import BadZipFile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from analyze_results import (
    CONFIGURATION,
    EVALUATION,
    PROTOCOL,
    grouped,
    number,
    read_csv,
    read_json,
)


METHODS = ("friction", "noise")
METHOD_LABELS = {"friction": "Friction randomization", "noise": "Observation noise"}
METHOD_AXIS = {
    "friction": "Minimum training friction",
    "noise": "Training noise multiplier",
}
METHOD_FIELD = {"friction": "training_friction_min", "noise": "training_noise_scale"}
LINE_STYLE = {"friction": "-", "noise": "--"}
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")
FOOT_COLORS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7")
PHASE_COLORS = {"acceleration": "#1b9e77", "braking": "#d95f02"}
LABELS = {
    "success_rate": "Trial success rate",
    "foot_slip_rate": "Loaded stance-foot slip rate",
    "critical_friction": "Critical friction",
    "rise_time_s": "Acceleration rise time (s)",
    "stopping_distance_m": "Stopping distance (m)",
    "tracking_rmse_m_s": "Planar velocity-tracking RMSE (m/s)",
    "moving_tracking_rmse_m_s": "Moving planar tracking RMSE (m/s)",
    "yaw_tracking_rmse_rad_s": "Yaw-rate tracking RMSE (rad/s)",
    "max_pitch_excursion_rad": "Maximum pitch excursion (rad)",
}
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#dce1e7",
    "grid.linewidth": 0.65,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
}


def numeric(row, field):
    value = number(row.get(field))
    return np.nan if value is None else value


def config_key(row):
    return tuple(row[key] for key in CONFIGURATION)


def config_sort(key):
    method, low, high, scale = key
    return (
        METHODS.index(method),
        *(
            float(value)
            for value in (
                (low, high, scale) if method == "friction" else (scale, low, high)
            )
        ),
    )


def config_label(key):
    method, low, high, scale = key
    if method == "friction":
        label = f"Friction [{float(low):g}, {float(high):g}]"
        return label if float(scale) == 0 else f"{label}, noise ×{float(scale):g}"
    label = f"Noise ×{float(scale):g}"
    return (
        label
        if float(low) == float(high) == 1.25
        else f"{label}, friction [{float(low):g}, {float(high):g}]"
    )


def configuration_style(rows):
    configs = {config_key(row) for row in rows}
    result = {}
    for method, cmap in zip(METHODS, ("Blues", "Oranges"), strict=True):
        keys = sorted((key for key in configs if key[0] == method), key=config_sort)
        colors = plt.colormaps[cmap](np.linspace(0.57, 0.92, len(keys)))
        for key, color, marker in zip(keys, colors, itertools.cycle(MARKERS)):
            result[key] = (color, marker)
    return result


def set_title(fig, title, context):
    fig.suptitle(f"{title}\n{context}", fontsize=10, fontweight="bold")


def unavailable(ax, message="No defined values"):
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="#535e68",
        fontsize=9,
        wrap=True,
    )


def add_legend(fig, axes):
    entries = {}
    for ax in np.atleast_1d(axes).flat:
        handles, labels = ax.get_legend_handles_labels()
        entries.update(zip(labels, handles, strict=True))
    if entries:
        fig.legend(
            entries.values(),
            entries.keys(),
            loc="outside lower center",
            ncols=min(4, len(entries)),
        )


def axis_label(metric):
    if metric in LABELS:
        return LABELS[metric]
    return metric.replace("_", " ").capitalize()


def format_percent_axis(ax):
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    low, high = ax.get_ylim()
    ax.set_ylim(min(low, -0.025), max(high, 1.025))


def bars(ax, x, y, y_std, color, x_std=None):
    for direction, errors in (("yerr", y_std), ("xerr", x_std)):
        if errors is None:
            continue
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(errors)
        if mask.any():
            ax.errorbar(
                x[mask],
                y[mask],
                **{direction: errors[mask]},
                fmt="none",
                ecolor=color,
                elinewidth=1,
                capsize=2.5,
                zorder=2,
            )


def friction_series(ax, summary, seeds, grid, styles, metric, method=None):
    has_values = False
    for key, (color, marker) in styles.items():
        if method is not None and key[0] != method:
            continue
        conditions = {
            float(row["test_friction"]): row
            for row in summary
            if config_key(row) == key
        }
        x = np.asarray(grid, dtype=float)
        mean = np.array(
            [numeric(conditions.get(value, {}), f"mean_{metric}") for value in x]
        )
        sd = np.array(
            [numeric(conditions.get(value, {}), f"std_{metric}") for value in x]
        )
        if not np.isfinite(mean).any():
            continue
        has_values = True
        individual = [row for row in seeds if config_key(row) == key]
        ax.scatter(
            [float(row["test_friction"]) for row in individual],
            [numeric(row, metric) for row in individual],
            color=color,
            alpha=0.27,
            s=14,
            zorder=3,
        )
        ax.plot(
            x,
            mean,
            linestyle=LINE_STYLE[key[0]],
            marker=marker,
            color=color,
            linewidth=1.35,
            markersize=4,
            label=config_label(key),
            zorder=4,
        )
        bars(ax, x, mean, sd, color)
    if not has_values:
        unavailable(ax)
    if len(grid) > 1:
        ax.set_xlim(grid[0] - 0.035, grid[-1] + 0.035)


def friction_figure(summary, seeds, grid, styles, metric, context, method=None):
    fig, ax = plt.subplots(figsize=(7.1, 4.15), layout="constrained")
    friction_series(ax, summary, seeds, grid, styles, metric, method)
    title = (
        f"{METHOD_LABELS[method]} trial success"
        if method
        else "Foot slip across test friction"
    )
    set_title(fig, title, context)
    ax.set(xlabel="Test ground friction", ylabel=axis_label(metric))
    format_percent_axis(ax)
    add_legend(fig, ax)
    return fig


def grf_figure(summary, seeds, grid, styles, context):
    fig, axes = plt.subplots(
        3, 2, figsize=(8.7, 7.0), sharex=True, layout="constrained"
    )
    for row, phase in enumerate(("Full trial", "Acceleration", "Braking")):
        prefix = "" if row == 0 else f"{phase.lower()}_"
        for column, (statistic, suffix) in enumerate(
            (("95th percentile", "p95"), ("Peak", "peak"))
        ):
            ax = axes[row, column]
            friction_series(
                ax, summary, seeds, grid, styles, f"{prefix}grf_ratio_{suffix}"
            )
            ax.set_title(f"{phase} · {statistic}")
    fig.supxlabel("Test ground friction", fontsize=9)
    fig.supylabel("Horizontal-to-vertical GRF ratio", fontsize=9)
    set_title(fig, "Ground reaction force ratios across test friction", context)
    add_legend(fig, axes)
    return fig


def training_figure(summary, seeds, styles, metric, context, methods=METHODS):
    fig, axes = plt.subplots(
        1,
        len(methods),
        squeeze=False,
        sharey=True,
        layout="constrained",
        figsize=(7.6, 4.25) if len(methods) == 2 else (6.4, 4.1),
    )
    for ax, method in zip(axes[0], methods, strict=True):
        field = METHOD_FIELD[method]
        has_values = False
        for row in summary:
            key = config_key(row)
            if key[0] != method or number(row[f"mean_{metric}"]) is None:
                continue
            color, marker = styles[key]
            x = np.array([float(row[field])])
            y = np.array([numeric(row, f"mean_{metric}")])
            sd = np.array([numeric(row, f"std_{metric}")])
            samples = [sample for sample in seeds if config_key(sample) == key]
            ax.scatter(
                [float(sample[field]) for sample in samples],
                [numeric(sample, metric) for sample in samples],
                color=color,
                alpha=0.3,
                s=18,
            )
            ax.scatter(
                x,
                y,
                color=color,
                marker=marker,
                s=42,
                label=config_label(key),
                zorder=3,
            )
            bars(ax, x, y, sd, color)
            has_values = True
        if not has_values:
            unavailable(ax)
        if len(methods) > 1:
            ax.set_title(METHOD_LABELS[method])
        ax.set_xlabel(METHOD_AXIS[method])
    axes[0, 0].set_ylabel(axis_label(metric))
    set_title(fig, axis_label(metric), context)
    add_legend(fig, axes)
    return fig


def tradeoff_figure(summary, seeds, styles, cost_metric, context):
    fig, ax = plt.subplots(figsize=(7.1, 4.55), layout="constrained")
    has_values = False
    for row in summary:
        if not int(row["n_paired_critical_friction"]):
            continue
        key = config_key(row)
        color, marker = styles[key]
        samples = [sample for sample in seeds if config_key(sample) == key]
        ax.scatter(
            [numeric(sample, "nominal_cost") for sample in samples],
            [numeric(sample, "critical_friction") for sample in samples],
            color=color,
            alpha=0.28,
            s=18,
        )
        x = np.array([numeric(row, "mean_paired_nominal_cost")])
        y = np.array([numeric(row, "mean_paired_critical_friction")])
        bars(
            ax,
            x,
            y,
            np.array([numeric(row, "std_paired_critical_friction")]),
            color,
            np.array([numeric(row, "std_paired_nominal_cost")]),
        )
        ax.scatter(
            x, y, color=color, marker=marker, s=48, label=config_label(key), zorder=3
        )
        has_values = True
    if not has_values:
        unavailable(
            ax, "No training seed has both a nominal cost and a critical friction"
        )
    set_title(fig, "Robustness–optimality trade-off", context)
    ax.set(
        xlabel=f"Nominal cost · {axis_label(cost_metric)}", ylabel="Critical friction"
    )
    add_legend(fig, ax)
    return fig


def load_trace(row, profile):
    if not row["derived_trace"]:
        raise ValueError("No validated Step 3 trace")
    with np.load(row["derived_trace"], allow_pickle=False) as data:
        trace = {
            key: data[key]
            for key in (
                "time_s",
                "valid_step",
                "grf_ratio",
                "acceleration_mask",
                "braking_mask",
            )
        }
    time = trace["time_s"]
    dt = number(profile["dt_s"])
    feet = profile["foot_order"]
    if (
        dt is None
        or dt <= 0
        or time.ndim != 1
        or not len(time)
        or not np.isfinite(time).all()
        or np.any(np.diff(time) <= 0)
    ):
        raise ValueError("Invalid derived trace clock")
    if not feet or trace["grf_ratio"].shape != (len(time), len(feet)):
        raise ValueError("Derived GRF array does not match the clock and foot order")
    for key in ("valid_step", "acceleration_mask", "braking_mask"):
        if trace[key].shape != time.shape or trace[key].dtype.kind != "b":
            raise ValueError(f"Invalid derived trace mask {key}")
    trace["edges"] = np.r_[time[0] - dt, time]
    trace["foot_order"] = feet
    if row["raw_trace"]:
        try:
            with np.load(row["raw_trace"], allow_pickle=False) as raw:
                if not np.array_equal(raw["time_s"], trace["time_s"]):
                    raise ValueError("Raw and derived clocks differ")
                trace.update(
                    {
                        key: raw[key]
                        for key in (
                            "command",
                            "base_linear_velocity_body_m_s",
                            "base_angular_velocity_body_rad_s",
                            "base_roll_pitch_yaw_rad",
                        )
                        if key in raw
                    }
                )
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
            trace["raw_error"] = str(exc)
    return trace


def choose_trace(rows, reference_friction, trace_id, trace_friction):
    if trace_id is not None:
        return next((row for row in rows if row["trace_id"] == trace_id), None)
    candidates = [
        row
        for row in rows
        if row["derived_trace"]
        and (trace_friction is None or float(row["test_friction"]) == trace_friction)
    ]
    return min(
        candidates,
        key=lambda row: (
            row["selected"] != "True",
            abs(float(row["test_friction"]) - reference_friction),
            row["trial_status"] != "completed",
            -int(row["valid_steps"] or 0),
            row["trace_id"],
        ),
        default=None,
    )


def phase_windows(edges, mask):
    transitions = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [(edges[start], edges[stop]) for start, stop in transitions.reshape(-1, 2)]


def shade_phases(ax, trace):
    for phase, color in PHASE_COLORS.items():
        for index, (left, right) in enumerate(
            phase_windows(trace["edges"], trace[f"{phase}_mask"])
        ):
            ax.axvspan(
                left,
                right,
                facecolor=color,
                alpha=0.13,
                label=phase.capitalize() if index == 0 else None,
            )


def trace_context(context, row):
    if row is None:
        return context
    label = (
        config_label(config_key(row))
        if row["method"] in METHODS and all(row[key] for key in CONFIGURATION[1:])
        else "Unknown training configuration"
    )
    status = row["trial_status"].replace("_", " ")
    return (
        f"{context}\n{label} · Test friction {float(row['test_friction']):g} · "
        f"Training seed {row['training_seed'] or '?'} · "
        f"Eval seed {row['eval_seed']} · {status}"
    )


def grf_history(trace, row, context):
    fig, axes = plt.subplots(
        1, 2, sharey=True, figsize=(8.3, 4.4), layout="constrained"
    )
    set_title(fig, "Stance-foot ground reaction force ratios", context)
    for ax, phase in zip(axes, PHASE_COLORS, strict=True):
        ax.set(title=phase.capitalize(), xlabel="Simulation time (s)")
        if trace is None:
            unavailable(ax, "No validated trace for this selection")
            continue
        mask = trace[f"{phase}_mask"]
        windows = phase_windows(trace["edges"], mask)
        if not windows:
            unavailable(ax, f"No recorded {phase} samples")
            continue
        ax.set_xlim(windows[0][0], windows[-1][1])
        for index, foot in enumerate(trace["foot_order"]):
            color = FOOT_COLORS[index % len(FOOT_COLORS)]
            ratios = np.where(mask, trace["grf_ratio"][:, index], np.nan)
            if np.isfinite(ratios).any():
                ax.plot(
                    trace["time_s"],
                    ratios,
                    marker=".",
                    markersize=2.1,
                    linewidth=1.05,
                    color=color,
                    label=foot,
                )
        if not np.isfinite(trace["grf_ratio"][mask]).any():
            unavailable(ax, "No loaded stance-foot samples")
        friction = number(row["applied_friction"])
        if friction is not None:
            ax.axhline(
                friction,
                color="#272e38",
                linestyle="--",
                linewidth=1.1,
                label=f"Applied friction μ = {friction:g}",
            )
    axes[0].set_ylabel("Horizontal-to-vertical GRF ratio")
    add_legend(fig, axes)
    return fig


def logged_series(trace, name):
    values = trace.get(name)
    if (
        values is None
        or values.shape != (len(trace["time_s"]), 3)
        or values.dtype.kind not in "fiu"
    ):
        return np.full((len(trace["time_s"]), 3), np.nan)
    return np.where(trace["valid_step"][:, None] & np.isfinite(values), values, np.nan)


def velocity_history(trace, context):
    fig, axes = plt.subplots(
        3, 1, sharex=True, figsize=(7.7, 6.0), layout="constrained"
    )
    set_title(fig, "Commanded and measured body velocities", context)
    if trace is not None:
        command = logged_series(trace, "command")
        linear = logged_series(trace, "base_linear_velocity_body_m_s")
        angular = logged_series(trace, "base_angular_velocity_body_rad_s")
    for index, (ax, label) in enumerate(
        zip(axes, ("Forward (m/s)", "Lateral (m/s)", "Yaw rate (rad/s)"), strict=True)
    ):
        ax.set_ylabel(label)
        if trace is None:
            unavailable(ax, "No validated trace for this selection")
            continue
        actual = linear[:, index] if index < 2 else angular[:, 2]
        if np.isfinite(command[:, index]).any():
            ax.stairs(
                command[:, index],
                trace["edges"],
                baseline=None,
                color="#47515e",
                linewidth=1.15,
                linestyle="--",
                label="Command",
            )
        if np.isfinite(actual).any():
            ax.plot(
                trace["time_s"],
                actual,
                color=FOOT_COLORS[0],
                linewidth=1.25,
                label="Measured",
            )
        else:
            unavailable(ax, "Measured signal unavailable")
        ax.set_xlim(trace["edges"][0], trace["edges"][-1])
    axes[-1].set_xlabel("Simulation time (s)")
    add_legend(fig, axes)
    return fig


def pitch_history(trace, context):
    fig, ax = plt.subplots(figsize=(7.7, 4.3), layout="constrained")
    set_title(fig, "Base pitch through acceleration and braking", context)
    ax.set(xlabel="Simulation time (s)", ylabel="Base pitch (°)")
    if trace is None:
        unavailable(ax, "No validated trace for this selection")
        return fig
    shade_phases(ax, trace)
    pitch = np.rad2deg(logged_series(trace, "base_roll_pitch_yaw_rad")[:, 1])
    if np.isfinite(pitch).any():
        ax.plot(trace["time_s"], pitch, color=FOOT_COLORS[0], linewidth=1.3)
        ax.axhline(0, color="#47515e", linewidth=0.75, linestyle="--")
    else:
        unavailable(ax, "Pitch signal unavailable")
    ax.set_xlim(trace["edges"][0], trace["edges"][-1])
    add_legend(fig, ax)
    return fig


def save_pair(fig, folder, basename, dpi):
    try:
        for extension in ("png", "pdf"):
            fig.savefig(folder / f"{basename}.{extension}", dpi=dpi, facecolor="white")
    finally:
        plt.close(fig)


def safe_folder(group):
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", group["maneuver"])
    scale = float(group["test_noise_scale"])
    iteration = group["checkpoint_iteration"] or "unknown"
    return f"{slug}_noise_{scale:g}_iter_{iteration}_{group['evaluation_profile']}"


def run_plots(
    analysis_dir,
    output,
    *,
    maneuver=None,
    test_noise_scale=None,
    profile_id=None,
    trace_id=None,
    trace_friction=None,
    dpi=300,
):
    analysis_dir, output = Path(analysis_dir).resolve(), Path(output).resolve()
    for value in (test_noise_scale, trace_friction):
        if value is not None and (value < 0 or not math.isfinite(value)):
            raise ValueError(
                "Friction and noise selections must be finite and nonnegative"
            )
    if dpi <= 0:
        raise ValueError("DPI must be positive")
    report = read_json(analysis_dir / "analysis_metadata.json")
    if report.get("protocol") != PROTOCOL:
        raise ValueError("Expected a Step 4 analysis directory")
    tables = {}
    for name in (
        "condition_summary",
        "seed_conditions",
        "nominal_summary",
        "seed_nominal",
        "traces",
    ):
        rows = read_csv(analysis_dir / f"{name}.csv")
        tables[name] = {
            tuple(group[0][key] for key in EVALUATION): group
            for group in grouped(rows, EVALUATION)
        }
    groups = sorted(set().union(*(set(table) for table in tables.values())))
    groups = [
        key
        for key in groups
        if (maneuver is None or maneuver == key[0])
        and (test_noise_scale is None or test_noise_scale == float(key[1]))
        and (profile_id is None or profile_id == key[2])
        and (
            trace_id is None
            or any(row["trace_id"] == trace_id for row in tables["traces"].get(key, ()))
        )
    ]
    if not groups:
        raise ValueError("No evaluation group matches the plot selection")

    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "data_kind": report["data_kind"],
        "source_analysis": str(analysis_dir),
        "nominal_friction": report["nominal_friction"],
        "cost_metric": report["cost_metric"],
        "dpi": dpi,
        "groups": [],
    }
    with plt.rc_context(STYLE):
        for key in groups:
            group = dict(zip(EVALUATION, key, strict=True))
            summary = tables["condition_summary"].get(key, ())
            seeds = [
                row
                for row in tables["seed_conditions"].get(key, ())
                if row["selected"] == "True"
            ]
            nominal = tables["nominal_summary"].get(key, ())
            nominal_seeds = [
                row
                for row in tables["seed_nominal"].get(key, ())
                if row["selected"] == "True"
            ]
            profile = report["evaluation_profiles"][group["evaluation_profile"]]
            grid = profile["test_friction_grid"] or sorted(
                {float(row["test_friction"]) for row in summary}
            )
            styles = configuration_style(itertools.chain(summary, nominal))
            row = choose_trace(
                tables["traces"].get(key, ()),
                report["nominal_friction"],
                trace_id,
                trace_friction,
            )
            trace = None
            trace_issue = None
            if row is not None:
                try:
                    trace = load_trace(row, profile)
                    trace_issue = trace.get("raw_error")
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                    EOFError,
                    BadZipFile,
                ) as exc:
                    trace_issue = str(exc)
            heading = {
                "synthetic": "SYNTHETIC · ",
                "unverified": "UNVERIFIED PROVENANCE · ",
            }.get(report["data_kind"], "")
            context = (
                f"{heading}{group['maneuver'].replace('_', ' ')} · "
                f"Test noise ×{float(group['test_noise_scale']):g} · "
                f"Checkpoint iteration {group['checkpoint_iteration'] or 'unknown'}"
            )
            seed_context = f"{context}\nTraining seed mean ± sample SD"
            nominal_context = (
                f"{seed_context} · Nominal friction {report['nominal_friction']:g}"
            )
            history_context = trace_context(context, row)

            folder = output / safe_folder(group)
            folder.mkdir()
            figures = (
                (
                    "01_success_friction_training",
                    friction_figure,
                    (
                        summary,
                        seeds,
                        grid,
                        styles,
                        "success_rate",
                        seed_context,
                        "friction",
                    ),
                ),
                (
                    "02_success_noise_training",
                    friction_figure,
                    (
                        summary,
                        seeds,
                        grid,
                        styles,
                        "success_rate",
                        seed_context,
                        "noise",
                    ),
                ),
                (
                    "03_critical_friction_by_training_friction",
                    training_figure,
                    (
                        nominal,
                        nominal_seeds,
                        styles,
                        "critical_friction",
                        seed_context,
                        ("friction",),
                    ),
                ),
                (
                    "04_critical_friction_by_training_noise",
                    training_figure,
                    (
                        nominal,
                        nominal_seeds,
                        styles,
                        "critical_friction",
                        seed_context,
                        ("noise",),
                    ),
                ),
                (
                    "05_grf_ratios_by_test_friction",
                    grf_figure,
                    (summary, seeds, grid, styles, seed_context),
                ),
                ("06_grf_time_history", grf_history, (trace, row, history_context)),
                (
                    "07_foot_slip_by_test_friction",
                    friction_figure,
                    (summary, seeds, grid, styles, "foot_slip_rate", seed_context),
                ),
                (
                    "08_velocity_time_history",
                    velocity_history,
                    (trace, history_context),
                ),
                (
                    "09_rise_time_by_training_configuration",
                    training_figure,
                    (nominal, nominal_seeds, styles, "rise_time_s", nominal_context),
                ),
                (
                    "10_stopping_distance_by_training_configuration",
                    training_figure,
                    (
                        nominal,
                        nominal_seeds,
                        styles,
                        "stopping_distance_m",
                        nominal_context,
                    ),
                ),
                (
                    "11_tracking_rmse_by_training_configuration",
                    training_figure,
                    (
                        nominal,
                        nominal_seeds,
                        styles,
                        "tracking_rmse_m_s",
                        nominal_context,
                    ),
                ),
                (
                    "12_robustness_optimality_tradeoff",
                    tradeoff_figure,
                    (
                        nominal,
                        nominal_seeds,
                        styles,
                        report["cost_metric"],
                        nominal_context,
                    ),
                ),
                ("13_pitch_time_history", pitch_history, (trace, history_context)),
            )
            for name, draw, args in figures:
                save_pair(draw(*args), folder, name, dpi)
            manifest["groups"].append(
                group
                | {
                    "directory": folder.relative_to(output).as_posix(),
                    "trace_id": row["trace_id"] if row else None,
                    "trace_status": row["trial_status"] if row else None,
                    "trace_issue": trace_issue,
                    "figures": [name for name, _, _ in figures],
                }
            )
    (output / "figures.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "analysis_dir", type=Path, help="Output directory from analyze_results.py"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for PNG and PDF figures",
    )
    parser.add_argument("--maneuver")
    parser.add_argument("--test-noise-scale", type=float)
    parser.add_argument("--evaluation-profile")
    trace = parser.add_mutually_exclusive_group()
    trace.add_argument("--trace-id", help="Exact time-history trial ID from traces.csv")
    trace.add_argument(
        "--trace-friction",
        type=float,
        help="Only choose time histories at this friction",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    try:
        manifest = run_plots(
            args.analysis_dir,
            args.output_dir,
            maneuver=args.maneuver,
            test_noise_scale=args.test_noise_scale,
            profile_id=args.evaluation_profile,
            trace_id=args.trace_id,
            trace_friction=args.trace_friction,
            dpi=args.dpi,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    total = len(manifest["groups"])
    print(
        f"Saved {manifest['data_kind']} figures for {total} evaluation groups "
        f"to {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
