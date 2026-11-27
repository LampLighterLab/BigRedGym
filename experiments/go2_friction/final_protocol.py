"""The finalized training conditions and matched evaluation maneuvers."""

import json
import re


PREVIOUS_FINAL_RESEARCH_ID = "go2trot-final-friction-noise-v1"
FINAL_RESEARCH_ID = "go2trot-final-friction-noise-staged-v2"
FINAL_RESEARCH_IDS = (PREVIOUS_FINAL_RESEARCH_ID, FINAL_RESEARCH_ID)
FRICTION = 1.25
LOWER_BOUNDS = (1.25, 0.6, 0.4, 0.2)
NOISE_SCALES = (0.0, 1.0, 2.0, 4.0)
SEEDS = (42, 43)
TEMPLATE_ITERATION = 200
REWARD_TRANSITION_ITERATION = 200
ITERATIONS = 400
NUM_ENVS = 64
ROLLOUT_SIZE = 16384
BATCH_SIZE = 8192
SAVE_INTERVAL = 25
PRIMARY_MANEUVER = "stand_1_stop"
MANEUVERS = (PRIMARY_MANEUVER, "steady_0p5", "steady_1p0")
TEST_FRICTIONS = (0.2, 0.4, 0.6, 0.8, 1.25)
TEST_NOISE_SCALES = (0.0, 1.0)
EVAL_SEEDS = tuple(range(10000, 10010))


def reward_schedule():
    """Iteration labels refer to the PPO updates, starting at 1."""
    return [
        {
            "coordination_reward": "legacy",
            "first_iteration": 1,
            "last_iteration": REWARD_TRANSITION_ITERATION,
            "weights": {
                "trot_support": 3.0,
                "swing_contact": 3.0,
                "paired_contact": 0.0,
            },
        },
        {
            "coordination_reward": "paired-contact",
            "first_iteration": REWARD_TRANSITION_ITERATION + 1,
            "last_iteration": ITERATIONS,
            "weights": {
                "trot_support": 0.0,
                "swing_contact": 0.0,
                "paired_contact": 3.0,
            },
        },
    ]


def checkpoint_stage(iteration):
    if not 0 <= iteration <= ITERATIONS:
        raise ValueError(
            f"Iteration {iteration} is outside the finalized reward schedule"
        )
    legacy, paired = reward_schedule()
    return legacy if iteration <= REWARD_TRANSITION_ITERATION else paired


def staged_checkpoint_profile(manifest, iteration):
    details = manifest["final_research"]
    if details["reward_schedule"] != reward_schedule():
        raise ValueError("The run manifest has an incompatible reward schedule")
    stage = checkpoint_stage(iteration)
    mode = stage["coordination_reward"]
    profile = details["stages"][mode]["controller_profile"]
    weights = profile["train_config"]["critic"]["reward"]["weights"]
    if (
        profile["options"]["coordination_reward"] != mode
        or profile["options"]["reference_mode"] != "additive"
        or any(
            weights.get(name, 0.0) != value for name, value in stage["weights"].items()
        )
    ):
        raise ValueError(f"Checkpoint profile disagrees with the {mode} stage")
    if manifest["status"] == "completed" and (
        details["transition"]["iteration"] != REWARD_TRANSITION_ITERATION
        or details["transition"]["mode"] != "same_runner"
        or any(
            details["stages"][name]["status"] != "completed"
            for name in ("legacy", "paired-contact")
        )
    ):
        raise ValueError("The run manifest lacks a completed staged reward transition")
    return profile


def read_completed_run(directory, condition):
    path = directory / "friction_experiment.json"
    data = json.loads(path.read_text())
    details = data.get("final_research", {})
    expected = {
        "status": "completed",
        "experiment_name": condition["experiment_name"],
        "seed": condition["seed"],
        "max_iterations": ITERATIONS,
        "friction_range": [condition["lower_bound"], FRICTION],
        "noise_scale": condition["noise_scale"],
    }
    if (
        any(data.get(k) != v for k, v in expected.items())
        or details.get("study") != FINAL_RESEARCH_ID
        or details.get("run_name") != directory.name
        or details.get("method") != condition["method"]
        or details.get("fresh_optimizers") is not True
        or details.get("initial_iteration") != 0
    ):
        raise ValueError(f"Incompatible or incomplete staged run: {path}")
    staged_checkpoint_profile(data, REWARD_TRANSITION_ITERATION)
    if staged_checkpoint_profile(data, ITERATIONS) != data["controller_profile"]:
        raise ValueError(f"The finalized and paired-contact profiles differ: {path}")
    return data


def validate_run_name(name):
    if not name or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Supply --run-name with a filename label")


def final_training_runs(sweep=None):
    """The friction sweep owns the shared zero-noise baseline for each seed."""
    conditions = []
    if sweep in (None, "friction"):
        conditions.extend(
            (
                "baseline" if lower == FRICTION else "friction",
                lower,
                0.0,
            )
            for lower in LOWER_BOUNDS
        )
    if sweep in (None, "noise"):
        conditions.extend(("noise", FRICTION, scale) for scale in NOISE_SCALES if scale)
    for method, lower, scale in conditions:
        if method == "baseline":
            label = method
        else:
            prefix, value = (
                ("friction_lb", lower)
                if method == "friction"
                else ("noise_scale", scale)
            )
            label = f"{prefix}_{value:.2f}".replace(".", "p")
        for seed in SEEDS:
            yield {
                "experiment_name": f"go2_final_staged_{label}_seed_{seed}",
                "method": method,
                "lower_bound": lower,
                "noise_scale": scale,
                "seed": seed,
            }
