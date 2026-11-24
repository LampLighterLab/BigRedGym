"""Generate labeled Step 2-format fixtures for offline analysis and plotting."""

import argparse
import csv
from itertools import product
import json
from pathlib import Path

import numpy as np


DT = 0.05
FRICTIONS = (0.2, 0.4, 0.8, 1.25)
TRAINING_SEEDS = (10, 20)
EVAL_SEEDS = tuple(range(100, 110))
FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
SCHEDULES = {
    "stand_2_stop": ((4, (0, 0, 0)), (10, (2, 0, 0)), (8, (0, 0, 0))),
    "steady_1p0": ((12, (1, 0, 0)),),
}
# For each training setting, the two entries are the seeds' cutoffs at test noise zero.
CONFIGS = (
    ("friction", 0.4, (0.4, 0.8)),
    ("friction", 0.8, (0.8, 1.25)),
    ("friction", 1.25, (0.2, 0.2)),
    ("noise", 0.0, (0.8, 1.25)),
    ("noise", 0.5, (0.4, 0.8)),
    ("noise", 2.0, (0.8, 0.8)),
)


def make_trace(schedule, friction, gain, trial_index, fell):
    lengths, targets = zip(*schedule, strict=True)
    command = np.repeat(np.array(targets, dtype=float), lengths, axis=0)
    stage = np.repeat(np.arange(len(schedule)), lengths)
    if fell:
        command, stage = command[:7], stage[:7]
    count = len(command)
    velocity = np.zeros_like(command)
    current = np.zeros(3)
    for index, target in enumerate(command):
        current += gain * (target - current)
        velocity[index] = current
    position = np.cumsum(DT * velocity, axis=0)
    position[:, 2] = 0.35

    pitch = 0.12 * (command[:, 0] - velocity[:, 0])
    attitude = np.column_stack((np.zeros(count), pitch, np.zeros(count)))
    zeros = np.zeros(count)
    orientation = np.column_stack((zeros, np.sin(pitch / 2), zeros, np.cos(pitch / 2)))
    step = np.arange(count)[:, None]
    contacts = (step + np.array([0, 1, 1, 0])[None, :]) % 4 < 3
    force = np.zeros((count, len(FEET), 3))
    force[:, :, 2] = 36 * contacts
    ratio = friction * (0.36 + 0.28 * abs(command[:, 0] - velocity[:, 0]) / 2)
    force[:, :, 0] = force[:, :, 2] * (
        ratio[:, None] * (1 + 0.06 * np.arange(len(FEET))) + 0.03 * (1 - gain)
    )
    foot_velocity = np.zeros_like(force)
    foot_velocity[:, :, 0] = np.where(
        contacts, np.where((step + trial_index) % 7 == 0, 0.15, 0.04), 0.4
    )
    terminal = np.zeros(count, dtype=bool)
    terminal[-1] = fell
    return {
        "time_s": DT * np.arange(1, count + 1),
        "stage_index": stage,
        "command": command,
        "base_linear_velocity_body_m_s": velocity,
        "base_angular_velocity_body_rad_s": np.zeros_like(velocity),
        "base_linear_velocity_world_m_s": velocity,
        "base_angular_velocity_world_rad_s": np.zeros_like(velocity),
        "base_position_world_m": position,
        "base_orientation_xyzw": orientation,
        "base_roll_pitch_yaw_rad": attitude,
        "foot_contact": contacts,
        "foot_ground_force_world_N": force,
        "foot_velocity_world_m_s": foot_velocity,
        "fell": terminal,
        "terminated": terminal,
        "timed_out": np.zeros_like(terminal),
        "numerical_failure": np.zeros_like(terminal),
    }


def write_dataset(dataset, method, value, cutoffs, seed_index, config_index):
    seed = TRAINING_SEEDS[seed_index]
    label = (
        f"go2friction_lb_{value:g}"
        if method == "friction"
        else f"go2noise_scale_{value:g}"
    )
    training = {
        "experiment_name": f"{label}_seed_{seed}",
        "seed": seed,
        "friction_range": [value if method == "friction" else 1.25, 1.25],
        "noise_scale": value if method == "noise" else 0,
    }
    (dataset / "traces").mkdir(parents=True)
    rows = []
    for (maneuver, schedule), friction, test_noise in product(
        SCHEDULES.items(), FRICTIONS, (0, 1)
    ):
        cutoff = cutoffs[seed_index]
        if test_noise:
            cutoff = FRICTIONS[min(FRICTIONS.index(cutoff) + 1, len(FRICTIONS) - 1)]
        successful_trials = (
            (10 if friction == FRICTIONS[-1] else 9) if friction >= cutoff else 4
        )
        if method == "noise" and value == 2 and seed_index == 1 and test_noise:
            successful_trials = 0
        for index, eval_seed in enumerate(EVAL_SEEDS):
            if (method, value, seed_index, maneuver, friction, test_noise, index) == (
                "friction",
                0.4,
                1,
                "stand_2_stop",
                0.2,
                1,
                9,
            ):
                continue
            fell = index >= successful_trials
            gain = 0.49 + 0.045 * config_index + 0.025 * seed_index + 0.005 * index
            trace = make_trace(schedule, friction, gain, index, fell)
            trial_id = f"trial_{len(rows):06d}"
            path = Path("traces") / f"{trial_id}.npz"
            np.savez_compressed(dataset / path, **trace)
            rows.append(
                {
                    "trial_id": trial_id,
                    "maneuver": maneuver,
                    "friction": friction,
                    "applied_friction": friction,
                    "noise_scale": test_noise,
                    "trial_index": index,
                    "eval_seed": eval_seed,
                    "initial_phase_rad": 0.0,
                    "status": "fall" if fell else "completed",
                    "planned_steps": sum(steps for steps, _ in schedule),
                    "recorded_steps": len(trace["time_s"]),
                    "trace": path.as_posix(),
                }
            )
    with (dataset / "trials.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "protocol": "go2trot-friction-eval-v1",
        "synthetic": True,
        "description": "SYNTHETIC offline fixtures generated from analytic signals.",
        "status": "completed",
        "checkpoint": f"synthetic/{training['experiment_name']}/model_100.pt",
        "checkpoint_sha256": f"synthetic-{config_index}-{seed}",
        "checkpoint_iteration": 100,
        "training": training,
        "backend": "synthetic",
        "dt_s": DT,
        "gait_frequency_hz": 2.0,
        "trial_seeds": EVAL_SEEDS,
        "frictions": FRICTIONS,
        "noise_scales": [0, 1],
        "noise_std_physical": {"0": {}, "1": {"dof_pos_obs": 0.01}},
        "maneuvers": {
            name: [{"steps": steps, "command": command} for steps, command in schedule]
            for name, schedule in SCHEDULES.items()
        },
        "foot_order": FEET,
        "trials_written": len(rows),
    }
    (dataset / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def generate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for config_index, (method, value, cutoffs) in enumerate(CONFIGS):
        for index, seed in enumerate(TRAINING_SEEDS):
            dataset = output / f"synthetic_{method}_{value:g}_seed_{seed}"
            write_dataset(dataset, method, value, cutoffs, index, config_index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="New folder for synthetic evaluations"
    )
    args = parser.parse_args()
    generate(args.output)
    print(f"Saved synthetic fixtures to {args.output.resolve()}")


if __name__ == "__main__":
    main()
