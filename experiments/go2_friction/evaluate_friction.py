"""Fixed-friction simulation evaluation for a trained BigRedGym go2trot policy.

Example (from BigRedGym root):
    uv run --frozen experiments/go2_friction/evaluate_friction.py \
      --experiment-name go2friction_lb_0p60_seed_42 \
      --device cpu --trials 10

Runs in MuJoCo CPU by default, using the same inference calls as scripts/play.py.
Does not update the network. Saves per-trial and aggregate CSV files.

IMPORTANT: These are simulation checks, NOT a physical-robot deployment script.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch


# (duration seconds, commanded vx, commanded vy, commanded yaw)
# Reversal and lateral magnitudes are explicit evaluation choices (1 m/s).
MANEUVERS = {
    "steady_0p5": ((4.0, 0.5, 0.0, 0.0),),
    "steady_1p0": ((4.0, 1.0, 0.0, 0.0),),
    "steady_1p5": ((4.0, 1.5, 0.0, 0.0),),
    "steady_2p0": ((4.0, 2.0, 0.0, 0.0),),
    "stand_2_stop": ((1.0, 0.0, 0.0, 0.0), (3.0, 2.0, 0.0, 0.0), (3.0, 0.0, 0.0, 0.0)),
    "forward_backward": ((1.0, 0.0, 0.0, 0.0), (3.0, 1.0, 0.0, 0.0), (3.0, -1.0, 0.0, 0.0)),
    "forward_sideways": ((1.0, 0.0, 0.0, 0.0), (3.0, 1.0, 0.0, 0.0), (3.0, 0.0, 1.0, 0.0)),
}
DEFAULT_FRICTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.25)


def parse_numbers(value: str) -> tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in value.split(",") if x.strip())
    if not vals or any(not math.isfinite(x) or x < 0 for x in vals):
        raise ValueError("Need a comma-separated list of nonnegative finite numbers")
    return vals


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def mean_or_nan(values):
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.mean(clean) if clean else math.nan


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["friction"], row["maneuver"])].append(row)
    out = []
    for (mu, name), trials in sorted(groups.items()):
        out.append({
            "friction": mu,
            "maneuver": name,
            "trials": len(trials),
            "successes": sum(bool(row["success"]) for row in trials),
            "success_rate": sum(bool(row["success"]) for row in trials) / len(trials),
            "mean_tracking_rmse_mps": mean_or_nan(r["tracking_rmse_mps"] for r in trials),
            "mean_friction_demand_p95": mean_or_nan(r["friction_demand_p95"] for r in trials),
            "mean_foot_slip_fraction": mean_or_nan(r["foot_slip_fraction"] for r in trials),
            "mean_stopping_distance_m": mean_or_nan(r["stopping_distance_m"] for r in trials),
            "mean_time_to_speed_s": mean_or_nan(r["time_to_speed_s"] for r in trials),
        })
    return out


def critical_friction(aggregates: list[dict]) -> float | None:
    """Proposal definition: smallest friction with >=9 successes in 10 tests."""
    valid = [
        row["friction"] for row in aggregates
        if row["maneuver"] == "stand_2_stop"
        and row["trials"] == 10 and row["successes"] >= 9
    ]
    return min(valid) if valid else None


def build_environment(args):
    import gym.envs  # noqa: F401  # required for task registration
    from gym.utils.helpers import set_seed
    from gym.utils.task_registry import task_registry

    env_cfg, train_cfg = task_registry.get_cfgs(
        "go2trot", original_cfg=args.original_cfg,
        experiment_name=args.experiment_name, load_run=args.load_run,
    )
    env_cfg.seed = args.eval_seed
    train_cfg.seed = args.eval_seed
    env_cfg.env.num_envs = 1
    # No random pushes; no automatic command changes during a trial.
    env_cfg.push_robots.toggle = False
    env_cfg.commands.resampling_time = 9999.0
    env_cfg.init_state.reset_mode = "reset_to_basic"
    env_cfg.env.episode_length_s = 20
    # Eliminate unrelated randomization during evaluation. A non-None range
    # must be present BEFORE constructing the backend so set_contact_friction
    # is enabled. All evaluations explicitly set the exact friction afterward.
    env_cfg.domain_randomization.startup.contact_friction_range = [1.25, 1.25]
    env_cfg.domain_randomization.startup.link_mass_scale_range = None
    env_cfg.domain_randomization.episode.scale_ranges = {}
    train_cfg.runner.device = args.device
    train_cfg.runner.resume = True
    train_cfg.runner.experiment_name = args.experiment_name
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    train_cfg.logging.enable_local_saving = False

    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    task_registry.set_log_dir_name(train_cfg)
    set_seed(args.eval_seed)
    env = task_registry.make_env("go2trot", env_cfg, args.device, True, "mujoco")
    runner = task_registry.make_alg_runner(env, train_cfg)
    runner.switch_to_eval()
    return env, runner


def measure_contact(env, fz_threshold: float, slip_threshold: float):
    """Per-stance-foot force ratios and foot slip from *world-frame* tensors.

    Rigid-body-state ordering is [position, quaternion, linear velocity,
    angular velocity], as defined by BigRedGym's MuJoCo CPU backend.
    """
    forces = env.contact_forces[0, env.feet_indices, :]  # (4,3)
    foot_states = env._rigid_body_state.reshape(env.num_envs, env.num_bodies, 13)
    foot_vxy = foot_states[0, env.feet_indices, 7:9]
    fz = forces[:, 2]
    stance = fz > fz_threshold
    tangential = torch.linalg.vector_norm(forces[:, :2], dim=1)
    ratios = (tangential[stance] / fz[stance]).detach().cpu().tolist()
    slips = (
        torch.linalg.vector_norm(foot_vxy[stance], dim=1) > slip_threshold
    ).detach().cpu().tolist()
    return ratios, slips


def evaluate_trial(env, runner, *, mu: float, maneuver: str, trial: int, args):
    from gym.utils.helpers import set_seed

    # Paired evaluation seeds: comparable resets across friction conditions.
    set_seed(args.eval_seed + trial)
    ids = torch.tensor([0], dtype=torch.long, device=env.device)
    values = torch.tensor([mu], dtype=env.contact_forces.dtype, device=env.device)
    env._backend.set_contact_friction(ids, values)
    env.reset()

    segments = MANEUVERS[maneuver]
    time_s = 0.0
    ratios, slips, tracking_sq = [], [], []
    stop_distance = 0.0
    speed_reached_at = math.nan
    terminal_speed = math.nan
    stop_reached = False
    falls = 0
    moving_errors = []
    stage_motion_projections = []
    final_stage_speed = math.nan
    total_steps = 0

    with torch.no_grad():
        for stage_index, (duration, vx, vy, yaw) in enumerate(segments):
            stage_start = time_s
            steps = round(duration / env.dt)
            projections = []
            for _ in range(steps):
                env.commands[0, :] = torch.tensor(
                    [vx, vy, yaw], dtype=env.commands.dtype, device=env.device
                )
                runner.set_actions(
                    runner.actor_cfg["actions"],
                    runner.get_inference_actions(),
                    runner.actor_cfg["disable_actions"],
                )
                env.step()
                total_steps += 1
                time_s += env.dt
                actual = env.base_lin_vel[0, :2].detach().cpu().tolist()
                err2 = (actual[0] - vx) ** 2 + (actual[1] - vy) ** 2
                tracking_sq.append(err2)
                requested_speed = math.hypot(vx, vy)
                if requested_speed > 0.01:
                    moving_errors.append(err2)
                    projections.append((actual[0] * vx + actual[1] * vy) / requested_speed)
                if (
                    maneuver == "stand_2_stop" and stage_index == 1
                    and not math.isfinite(speed_reached_at)
                    and actual[0] >= 1.8
                ):
                    speed_reached_at = time_s - stage_start
                if maneuver == "stand_2_stop" and stage_index == 2:
                    # Travel distance during braking, integrating world XY speed.
                    world_vxy = env.root_states[0, 7:9].detach().cpu().tolist()
                    terminal_speed = math.hypot(*world_vxy)
                    if not stop_reached:
                        stop_distance += terminal_speed * env.dt
                        if terminal_speed <= 0.15:
                            stop_reached = True
                r, s = measure_contact(env, args.min_normal_force, args.slip_threshold)
                ratios.extend(r)
                slips.extend(s)
                if bool(env.terminated[0].item()):
                    falls = 1
                    break
            requested_speed = math.hypot(vx, vy)
            if requested_speed > 0.01:
                stage_motion_projections.append((mean_or_nan(projections), requested_speed))
            final_stage_speed = math.hypot(*env.base_lin_vel[0, :2].detach().cpu().tolist())
            if falls:
                break

    rmse = math.sqrt(sum(tracking_sq) / len(tracking_sq)) if tracking_sq else math.nan
    moving_rmse = math.sqrt(sum(moving_errors) / len(moving_errors)) if moving_errors else math.nan
    p95 = math.nan
    if ratios:
        sorted_ratios = sorted(ratios)
        p95 = sorted_ratios[math.ceil(0.95 * len(sorted_ratios)) - 1]
    # Success requires both survival AND moving as commanded; otherwise a
    # motionless policy could be incorrectly counted as robust.
    moving_ok = (math.isfinite(moving_rmse) and moving_rmse <= args.max_moving_rmse
                 and len(stage_motion_projections) == sum(math.hypot(x[1], x[2]) > 0.01 for x in segments)
                 and all(math.isfinite(got) and got >= 0.5 * requested
                         for got, requested in stage_motion_projections))
    stop_ok = (maneuver != "stand_2_stop") or final_stage_speed <= args.stop_speed_tol
    success = not falls and moving_ok and stop_ok
    return {
        "experiment_name": args.experiment_name,
        "friction": mu,
        "maneuver": maneuver,
        "trial": trial,
        "success": int(success),
        "fell": falls,
        "time_s": round(time_s, 4),
        "steps": total_steps,
        "tracking_rmse_mps": rmse,
        "moving_tracking_rmse_mps": moving_rmse,
        "final_stage_speed_mps": final_stage_speed,
        "friction_demand_p95": p95,
        "friction_demand_peak": max(ratios) if ratios else math.nan,
        "cone_utilization_p95": p95 / mu if mu > 0 and math.isfinite(p95) else math.nan,
        "foot_slip_fraction": sum(slips) / len(slips) if slips else math.nan,
        "foot_stance_samples": len(slips),
        "stopping_distance_m": (stop_distance if stop_reached else math.nan)
        if maneuver == "stand_2_stop" else math.nan,
        "stopped_below_0p15_mps": int(stop_reached) if maneuver == "stand_2_stop" else "",
        "time_to_speed_s": speed_reached_at if maneuver == "stand_2_stop" else math.nan,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fixed-friction Go2 policy evaluation")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--load-run", default=-1, help="Training run subdirectory; default latest")
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--original-cfg", action="store_true", help="Load saved run configs")
    parser.add_argument("--device", default="cpu", help="Use cpu for verifiable MuJoCo contact readings")
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--frictions", default=",".join(map(str, DEFAULT_FRICTIONS)))
    parser.add_argument("--maneuvers", default=",".join(MANEUVERS))
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--min-normal-force", type=float, default=5.0)
    parser.add_argument("--slip-threshold", type=float, default=0.1)
    parser.add_argument("--max-moving-rmse", type=float, default=0.75,
                        help="Chosen success cutoff in m/s, not prescribed in proposal")
    parser.add_argument("--stop-speed-tol", type=float, default=0.35,
                        help="Chosen stopping cutoff in m/s, not prescribed in proposal")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    if args.device != "cpu":
        parser.error("This initial evaluator is validated for the MuJoCo CPU interface only")
    if args.trials < 1:
        parser.error("--trials must be positive")
    if args.min_normal_force <= 0 or args.slip_threshold < 0:
        parser.error("Contact thresholds must be nonnegative (normal force > 0)")
    try:
        frictions = sorted(set(parse_numbers(args.frictions)))
    except ValueError as exc:
        parser.error(str(exc))
    names = [x.strip() for x in args.maneuvers.split(",") if x.strip()]
    if any(x not in MANEUVERS for x in names):
        parser.error(f"Unknown maneuver; choose from: {', '.join(MANEUVERS)}")
    if not names:
        parser.error("Provide at least one maneuver")
    repo = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "results" / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    env, runner = build_environment(args)
    rows = []
    for mu in frictions:
        for name in names:
            for trial in range(args.trials):
                row = evaluate_trial(env, runner, mu=mu, maneuver=name, trial=trial, args=args)
                rows.append(row)
                print(f"mu={mu:.2f} {name} trial={trial + 1}/{args.trials} "
                      f"success={row['success']} slip={row['foot_slip_fraction']:.3f}", flush=True)
                save_csv(output_dir / "per_trial.csv", rows)

    agg = aggregate(rows)
    save_csv(output_dir / "summary.csv", agg)
    crit = critical_friction(agg)
    report = {
        "experiment_name": args.experiment_name,
        "critical_friction": crit,
        "definition": "minimum tested mu with >=9/10 successes in stand_2_stop",
        "frictions_tested": frictions,
        "trials_per_condition": args.trials,
        "no_pushes": True,
        "slip_threshold_mps": args.slip_threshold,
        "min_normal_force_N": args.min_normal_force,
        "success_cutoffs": {
            "max_moving_rmse_mps": args.max_moving_rmse,
            "stop_speed_tol_mps": args.stop_speed_tol,
        },
        "warning": (
            "A critical friction is reported only for 10-trial evaluations; "
            "success cutoffs are exploratory choices to be reviewed by the team. "
            "Measured force ratios and foot speeds are approximate contact proxies."
        ),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nResults:", output_dir)
    print("Critical tested friction:", crit)


if __name__ == "__main__":
    main()
