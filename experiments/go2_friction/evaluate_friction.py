"""Collect Go2Trot trial telemetry at fixed friction and actor noise levels."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import statistics

import mujoco
import numpy as np
import torch

from evaluation_contacts import FootGroundContacts
from gym.envs.base.domain_randomization import apply_domain_randomization_override
from gym.envs.go2.go2trot_config import Go2TrotRunnerCfg
from gym.utils.helpers import class_to_dict, get_load_path
from gym.utils.original_cfg import load_original_cfgs_from_run
from gym.utils.task_registry import task_registry
from train_friction import configure_observation_noise

# train_friction makes the repository's scripts directory available for imports.
from evaluate_go2trot import COMMANDS

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
DEFAULT_FRICTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.25)
PRIMARY_MANEUVER = "stand_2_stop"
NUMERICAL_WARNINGS = (
    mujoco.mjtWarning.mjWARN_BADQPOS,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQACC,
    mujoco.mjtWarning.mjWARN_BADCTRL,
)

# Each stage is (duration seconds, body-frame vx, vy, yaw rate).
MANEUVERS = {
    "stand_2_stop": (
        (1.0, *COMMANDS["stopped"]),
        (3.0, *COMMANDS["forward_fast"]),
        (3.0, *COMMANDS["stopped"]),
    ),
    "steady_0p5": ((4.0, *COMMANDS["forward_slow"]),),
    "steady_1p0": ((4.0, *COMMANDS["forward"]),),
    "steady_1p5": ((4.0, 1.5, 0.0, 0.0),),
    "steady_2p0": ((4.0, *COMMANDS["forward_fast"]),),
    "forward_backward": (
        (1.0, *COMMANDS["stopped"]),
        (3.0, *COMMANDS["forward"]),
        (3.0, -1.0, 0.0, 0.0),
    ),
    "forward_sideways": (
        (1.0, *COMMANDS["stopped"]),
        (3.0, *COMMANDS["forward"]),
        (3.0, 0.0, 1.0, 0.0),
    ),
}


def parse_numbers(value: str) -> tuple[float, ...]:
    vals = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not vals or any(not math.isfinite(item) or item < 0 for item in vals):
        raise ValueError("Need a comma-separated list of nonnegative finite numbers")
    return vals


# Kept for callers reading historical aggregate evaluation results.
def mean_or_nan(values):
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.mean(clean) if clean else math.nan


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["friction"], row["maneuver"])].append(row)
    out = []
    for (mu, name), trials in sorted(groups.items()):
        out.append(
            {
                "friction": mu,
                "maneuver": name,
                "trials": len(trials),
                "successes": sum(bool(row["success"]) for row in trials),
                "success_rate": sum(bool(row["success"]) for row in trials)
                / len(trials),
                "mean_tracking_rmse_mps": mean_or_nan(
                    r["tracking_rmse_mps"] for r in trials
                ),
                "mean_friction_demand_p95": mean_or_nan(
                    r["friction_demand_p95"] for r in trials
                ),
                "mean_foot_slip_fraction": mean_or_nan(
                    r["foot_slip_fraction"] for r in trials
                ),
                "mean_stopping_distance_m": mean_or_nan(
                    r["stopping_distance_m"] for r in trials
                ),
                "mean_time_to_speed_s": mean_or_nan(
                    r["time_to_speed_s"] for r in trials
                ),
            }
        )
    return out


def critical_friction(aggregates: list[dict]) -> float | None:
    """Proposal definition: smallest friction with >=9 successes in 10 tests."""
    valid = [
        row["friction"]
        for row in aggregates
        if row["maneuver"] == "stand_2_stop"
        and row["trials"] == 10
        and row["successes"] >= 9
    ]
    return min(valid) if valid else None


def select_maneuvers(names, config_path=None):
    maneuvers = dict(MANEUVERS)
    if config_path is not None:
        overrides = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            raise ValueError("Maneuver config must be a JSON object keyed by maneuver")
        for name, stages in overrides.items():
            if name not in maneuvers:
                raise ValueError(f"Unknown maneuver in config: {name}")
            if not isinstance(stages, list) or not stages:
                raise ValueError(f"{name} requires at least one stage")
            for stage in stages:
                if (
                    not isinstance(stage, list)
                    or len(stage) != 4
                    or any(type(value) not in (int, float) for value in stage)
                    or not all(math.isfinite(value) for value in stage)
                    or stage[0] <= 0
                ):
                    raise ValueError(
                        f"{name} stages need [positive seconds, vx, vy, yaw]"
                    )
            maneuvers[name] = tuple(tuple(stage) for stage in stages)
    if names == "all":
        return maneuvers
    selected = names.split(",")
    selected = [name.strip() for name in selected]
    if any(name not in maneuvers for name in selected):
        raise ValueError(f"Choose from {', '.join(maneuvers)} or all")
    if len(set(selected)) != len(selected):
        raise ValueError("Maneuvers must be unique")
    return {name: maneuvers[name] for name in selected}


def compile_maneuver(stages, dt):
    schedule = [(round(seconds / dt), tuple(command)) for seconds, *command in stages]
    if any(steps < 1 for steps, _ in schedule):
        raise ValueError(
            f"Every maneuver stage must round to at least one {dt:g} s step"
        )
    return schedule


def add_evaluation_arguments(parser, *, frictions=(1.25,), noise_scales=(0.0,)):
    parser.add_argument(
        "--frictions", "--friction", type=parse_numbers, default=frictions
    )
    parser.add_argument(
        "--noise-scales", "--noise-scale", type=parse_numbers, default=noise_scales
    )
    parser.add_argument("--maneuvers", default=PRIMARY_MANEUVER)
    parser.add_argument("--maneuver-config", type=Path, help="JSON stage overrides")
    parser.add_argument("--trials", type=int, help="Trials per condition (default: 10)")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--eval-seed", type=int, default=10000, help="First trial seed")
    seeds.add_argument("--eval-seeds", help="Explicit comma-separated trial seeds")
    parser.add_argument("--num-envs", type=int, default=1, help="Concurrent CPU worlds")
    parser.add_argument(
        "--device", default="cpu", help="Actor device, e.g. cpu or cuda:0"
    )
    parser.add_argument(
        "--gait-frequency", type=float, default=2.0, help="Fixed gait Hz"
    )
    parser.add_argument(
        "--original-cfg", action="store_true", help="Use saved config code"
    )


def evaluation_settings(parser, args):
    if args.num_envs < 1 or (args.trials is not None and args.trials < 1):
        parser.error("--num-envs and --trials must be positive")
    if not math.isfinite(args.gait_frequency) or args.gait_frequency <= 0:
        parser.error("--gait-frequency must be finite and positive")
    if any(
        len(set(values)) != len(values)
        for values in (args.frictions, args.noise_scales)
    ):
        parser.error("Friction coefficients and noise scales must be unique")
    try:
        device = torch.device(args.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Actor --device must be cpu or cuda")
        maneuvers = select_maneuvers(args.maneuvers, args.maneuver_config)
        if args.eval_seeds is None:
            trials = args.trials if args.trials is not None else 10
            seeds = range(args.eval_seed, args.eval_seed + trials)
            if not 0 <= seeds[0] <= seeds[-1] < 2**32:
                raise ValueError("Trial seeds must be in [0, 2**32 - 1]")
        else:
            seeds = tuple(int(seed) for seed in args.eval_seeds.split(","))
            if args.trials is not None and args.trials != len(seeds):
                raise ValueError("--trials must match the number of explicit seeds")
            if len(set(seeds)) != len(seeds) or any(
                not 0 <= seed < 2**32 for seed in seeds
            ):
                raise ValueError(
                    "Explicit seeds must be unique integers in [0, 2**32 - 1]"
                )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return tuple(seeds), maneuvers


def make_environment(checkpoint, args, seeds, maneuvers):
    import gym.envs  # noqa: F401

    if args.original_cfg:
        env_cfg, train_cfg = load_original_cfgs_from_run("go2trot", checkpoint.parent)
    else:
        env_cfg, train_cfg = deepcopy(task_registry.get_cfgs("go2trot"))
    env_cfg.seed = train_cfg.seed = seeds[0]
    env_cfg.env.num_envs = min(args.num_envs, len(seeds))
    env_cfg.init_state.reset_mode = "reset_to_basic"
    env_cfg.control.gait_freq = [args.gait_frequency, args.gait_frequency]
    env_cfg.push_robots.toggle = False
    apply_domain_randomization_override(env_cfg, "off")
    # The CPU backend must enable the friction setter during construction.
    env_cfg.domain_randomization.startup.contact_friction_range = [
        args.frictions[0]
    ] * 2

    train_cfg.runner.device = args.device
    train_cfg.runner.resume = False
    train_cfg.logging.enable_local_saving = False
    train_cfg.log_dir = None
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    dt = env_cfg.control.ctrl_dt
    schedules = {
        name: compile_maneuver(stages, dt) for name, stages in maneuvers.items()
    }
    horizon = max(
        sum(steps for steps, _ in schedule) for schedule in schedules.values()
    )
    env_cfg.env.episode_length_s = (horizon + 2) * dt
    env_cfg.commands.resampling_time = (horizon + 2) * dt
    env = task_registry.make_env("go2trot", env_cfg, "cpu", True, "mujoco")
    return env, train_cfg, schedules


def load_policy(env, train_cfg, checkpoint):
    runner = task_registry.make_alg_runner(env, train_cfg)
    runner.load(checkpoint, load_optimizer=False)
    runner.switch_to_eval()
    runner.alg.actor.requires_grad_(False)
    runner.alg.critic.requires_grad_(False)
    return runner


def observation_noise_std(runner, effective_std):
    """Convert physical standard deviations to the actor's scaled input units."""
    if not effective_std:
        return None
    widths = []
    for name in runner.actor_cfg["obs"]:
        width = runner.get_obs_size([name])
        std = torch.as_tensor(effective_std.get(name, 0.0), device=runner.device)
        scale = runner.env.scales.get(name, 1.0)
        if isinstance(scale, torch.Tensor):
            scale = scale.to(runner.device)
        widths.append((std / scale).expand(width))
    return torch.cat(widths)


def inference_actions(runner, std, generators):
    if std is None:
        return runner.get_inference_actions()
    clean = runner.get_obs(runner.actor_cfg["obs"])
    samples = torch.stack(
        [torch.randn(clean.shape[1], generator=generator) for generator in generators]
    ).to(clean)
    return runner.alg.actor.act_inference(clean + samples * std)


def reset_batch(env, friction, seeds):
    backend = env._backend
    applied = torch.full_like(backend.contact_friction, friction)
    backend.set_contact_friction(torch.arange(env.num_envs), applied)
    for world, data in enumerate(backend._datas):
        mujoco.mj_resetData(backend._model_for_env(world), data)
    env.reset()
    if not torch.equal(backend.contact_friction, applied):
        raise RuntimeError("Evaluation friction changed during reset")
    # The first draw sets the phase. Subsequent draws supply the actor noise.
    # Each trial's stream is paired across conditions and independent of batching.
    padded = tuple(seeds) + (seeds[0],) * (env.num_envs - len(seeds))
    generators = [torch.Generator(device="cpu").manual_seed(seed) for seed in padded]
    initial_phase = math.tau * torch.stack(
        [torch.rand((), generator=generator) for generator in generators]
    )
    env.phase[:, 0] = initial_phase
    env._update_phase_observation()
    env._update_gait_reference()
    return initial_phase, generators


def roll_pitch_yaw(quat):
    x, y, z, w = quat.T
    return np.stack(
        (
            np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
            np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
        ),
        axis=-1,
    )


def capture_step(env, contacts, step, stage, invalid_action):
    root = env.root_states.numpy().copy()
    rigid_bodies = env._backend.rigid_body_states.view(env.num_envs, env.num_bodies, 13)
    feet_velocity = rigid_bodies[:, env.feet_indices, 7:10].numpy()
    active_contacts, forces = contacts.read()
    terminated = env.terminated.numpy().copy()
    timed_out = env.timed_out.numpy().copy()
    numerical_failure = invalid_action.numpy() | ~np.isfinite(root).all(axis=1)
    numerical_failure |= ~np.isfinite(feet_velocity).all(axis=(1, 2))
    numerical_failure |= ~np.isfinite(forces).all(axis=(1, 2))
    # MuJoCo can recover from instability by resetting to finite state within
    # mj_step. The per-world warning counters also detect that step as failed.
    numerical_failure |= [
        any(data.warning[warning].number for warning in NUMERICAL_WARNINGS)
        for data in env._backend._datas
    ]
    return {
        "time_s": np.full(env.num_envs, step * env.dt),
        "stage_index": np.full(env.num_envs, stage, dtype=np.int32),
        "command": env.commands.numpy().copy(),
        "base_linear_velocity_body_m_s": env.base_lin_vel.numpy().copy(),
        "base_angular_velocity_body_rad_s": env.base_ang_vel.numpy().copy(),
        "base_linear_velocity_world_m_s": root[:, 7:10],
        "base_angular_velocity_world_rad_s": root[:, 10:13],
        "base_position_world_m": root[:, :3],
        "base_orientation_xyzw": root[:, 3:7],
        "base_roll_pitch_yaw_rad": roll_pitch_yaw(root[:, 3:7]),
        "foot_contact": active_contacts,
        "foot_ground_force_world_N": forces,
        "foot_velocity_world_m_s": feet_velocity,
        "fell": terminated,
        "terminated": terminated,
        "timed_out": timed_out,
        "numerical_failure": numerical_failure,
    }


@torch.inference_mode()
def collect_batch(env, runner, contacts, friction, schedule, seeds, std):
    initial_phase, generators = reset_batch(env, friction, seeds)
    active = torch.arange(env.num_envs) < len(seeds)
    lengths = np.zeros(env.num_envs, dtype=int)
    records = defaultdict(list)
    step = 0
    for stage, (steps, command) in enumerate(schedule):
        target = torch.tensor(command, dtype=env.commands.dtype).expand_as(env.commands)
        for _ in range(steps):
            if not active.any():
                break
            env.commands.copy_(target)
            actions = inference_actions(runner, std, generators).cpu()
            invalid_action = ~torch.isfinite(actions).all(dim=1)
            actions[invalid_action | ~active] = 0
            runner.set_actions(
                runner.actor_cfg["actions"],
                actions,
                runner.actor_cfg["disable_actions"],
            )
            env.step()
            if not torch.equal(env.commands, target):
                raise RuntimeError(
                    "Task resampled a commanded velocity during evaluation"
                )
            step += 1
            sample = capture_step(env, contacts, step, stage, invalid_action)
            for key, value in sample.items():
                records[key].append(value)
            lengths += active.numpy()
            ended = (
                sample["terminated"] | sample["timed_out"] | sample["numerical_failure"]
            )
            active &= ~torch.from_numpy(ended)

    trace = {key: np.stack(values) for key, values in records.items()}
    for slot in range(len(seeds)):
        trial = {key: value[: lengths[slot], slot] for key, value in trace.items()}
        yield trial, initial_phase[slot].item()


def write_metadata(path, metadata):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def trial_status(trace):
    for flag, status in (
        ("numerical_failure", "numerical_failure"),
        ("fell", "fall"),
        ("timed_out", "timeout"),
    ):
        if trace[flag][-1]:
            return status
    return "completed"


def checkpoint_output_path(checkpoint):
    return Path(checkpoint.parent.parent.name, checkpoint.parent.name, checkpoint.stem)


def record_evaluation(
    checkpoint, output, env, runner, train_cfg, schedules, args, seeds
):
    contacts = FootGroundContacts(env)
    effective_std = {}
    for scale in args.noise_scales:
        noise_cfg = Go2TrotRunnerCfg()
        configure_observation_noise(noise_cfg, scale)
        effective_std[scale] = noise_cfg.actor.noise
    stds = {
        scale: observation_noise_std(runner, values)
        for scale, values in effective_std.items()
    }
    with checkpoint.open("rb") as fh:
        checkpoint_sha = hashlib.file_digest(fh, "sha256").hexdigest()
    training_manifest = checkpoint.with_name("friction_experiment.json")
    training = (
        json.loads(training_manifest.read_text(encoding="utf-8"))
        if training_manifest.is_file()
        else None
    )
    metadata = {
        "protocol": "go2trot-friction-eval-v1",
        "status": "running",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_iteration": runner.it,
        "training": training,
        "backend": "mujoco_cpu",
        "mujoco_version": mujoco.__version__,
        "policy_device": args.device,
        "dt_s": env.dt,
        "num_envs": env.num_envs,
        "requested_num_envs": args.num_envs,
        "trial_seeds": seeds,
        "frictions": args.frictions,
        "noise_scales": args.noise_scales,
        "noise_distribution": "independent zero-mean Gaussian on actor inputs",
        "noise_std_physical": effective_std,
        "gait_frequency_hz": args.gait_frequency,
        "maneuvers": {
            name: [{"steps": steps, "command": command} for steps, command in schedule]
            for name, schedule in schedules.items()
        },
        "foot_order": list(env.robot_layout.body_groups["feet"]),
        "command_order": ["body_vx_m_s", "body_vy_m_s", "body_yaw_rate_rad_s"],
        "sample_timing": (
            "Elapsed control time after the reset warm-up step. Commands apply "
            "over the recorded interval. Base and foot kinematics are sampled "
            "at the interval end. Contacts and forces are from the solver's "
            "last physics substep in that interval."
        ),
        "contact_state": "Active foot-ground solver contact",
        "force_frame": "World XYZ, ground on foot, last physics substep",
        "foot_velocity_point": "Foot body origin",
        "fall_definition": "Go2Trot base-contact termination",
        "numerical_failure_definition": (
            "Nonfinite action or telemetry, or a MuJoCo instability warning. "
            "The terminal timestamp marks the attempted control interval."
        ),
        "env_config": class_to_dict(env.cfg),
        "runner_config": class_to_dict(train_cfg),
    }
    output.mkdir(parents=True, exist_ok=False)
    traces = output / "traces"
    traces.mkdir()
    manifest = output / "metadata.json"
    write_metadata(manifest, metadata)
    count = 0
    fields = (
        "trial_id",
        "maneuver",
        "friction",
        "applied_friction",
        "noise_scale",
        "trial_index",
        "eval_seed",
        "initial_phase_rad",
        "status",
        "planned_steps",
        "recorded_steps",
        "trace",
    )
    try:
        with (output / "trials.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            conditions = product(schedules.items(), args.frictions, stds.items())
            for (maneuver, schedule), friction, (scale, std) in conditions:
                planned = sum(steps for steps, _ in schedule)
                for start in range(0, len(seeds), env.num_envs):
                    batch = seeds[start : start + env.num_envs]
                    trials = collect_batch(
                        env, runner, contacts, friction, schedule, batch, std
                    )
                    for slot, (trace, phase) in enumerate(trials):
                        trial_id = f"trial_{count:06d}"
                        path = traces / f"{trial_id}.npz"
                        tmp = path.with_suffix(".tmp")
                        with tmp.open("wb") as target:
                            np.savez_compressed(target, **trace)
                        tmp.replace(path)
                        writer.writerow(
                            {
                                "trial_id": trial_id,
                                "maneuver": maneuver,
                                "friction": friction,
                                "applied_friction": env._backend.contact_friction[
                                    slot
                                ].item(),
                                "noise_scale": scale,
                                "trial_index": start + slot,
                                "eval_seed": batch[slot],
                                "initial_phase_rad": phase,
                                "status": trial_status(trace),
                                "planned_steps": planned,
                                "recorded_steps": len(trace["time_s"]),
                                "trace": path.relative_to(output).as_posix(),
                            }
                        )
                        fh.flush()
                        count += 1
    except BaseException as exc:
        metadata.update(
            status="interrupted_or_failed", error=repr(exc), trials_written=count
        )
        write_metadata(manifest, metadata)
        raise
    metadata.update(status="completed", trials_written=count)
    write_metadata(manifest, metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--checkpoint-path", type=Path)
    selector.add_argument("--experiment-name")
    parser.add_argument("--load-run", default=-1, help="Training run, default latest")
    parser.add_argument(
        "--checkpoint", type=int, default=-1, help="Iteration, default latest"
    )
    parser.add_argument("--output-dir", type=Path, help="New dataset directory")
    add_evaluation_arguments(parser)
    args = parser.parse_args(argv)
    seeds, maneuvers = evaluation_settings(parser, args)
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable for actor inference")
    if args.checkpoint_path is None:
        load_run = -1 if args.load_run in (-1, "-1") else args.load_run
        try:
            path = Path(get_load_path(args.experiment_name, load_run, args.checkpoint))
        except (IndexError, OSError, ValueError) as exc:
            parser.error(f"Could not locate checkpoint: {exc}")
    else:
        path = args.checkpoint_path
    checkpoint = path.resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    output = args.output_dir or RESULTS / "evaluation" / checkpoint_output_path(
        checkpoint
    )
    if output.exists():
        parser.error(f"Output directory already exists: {output}")

    env, train_cfg, schedules = make_environment(checkpoint, args, seeds, maneuvers)
    try:
        runner = load_policy(env, train_cfg, checkpoint)
        report = record_evaluation(
            checkpoint, output, env, runner, train_cfg, schedules, args, seeds
        )
    finally:
        env._backend.close()
    print(f"Saved {report['trials_written']} trials in {output}")


if __name__ == "__main__":
    main()
