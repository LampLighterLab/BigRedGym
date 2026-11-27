"""Write self-contained controller checkpoints using the core runner format."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import torch

import final_protocol as final
from gait_controller import (
    ACTOR_HZ,
    MOTION_THRESHOLD,
    RAMP_S,
    WALK_PROBABILITIES,
    WALK_SPEEDS,
    ControllerOptions,
)
from gym import GYM_ROOT_DIR
from gym.utils.helpers import class_to_dict


PROFILE_VERSION = 1
PAIRED_WEIGHT_STUDY_ID = "go2trot-paired-contact-weight-ablation-v1"
WEIGHT3_CONTINUATION_ID = "go2trot-paired-contact-weight3-continuation-v1"
FRESH_STABILITY_ID = "go2trot-fresh-stability-pilots-v1"
FRESH_CONTROL_CONTINUATION_ID = "go2trot-fresh-control-continuation-v1"


def verify_finalized_template(profile, manifest):
    """Require the completed fresh Control run used for the final controller."""
    if profile is None or profile != manifest["controller_profile"]:
        raise ValueError(
            "Checkpoint controller profile does not match its run manifest"
        )
    fresh = manifest.get("fresh_pilot", {})
    env, train = profile["env_config"], profile["train_config"]
    options = profile["options"]
    weights = train["critic"]["reward"]["weights"]
    dr = env["domain_randomization"]
    curriculum = [
        {"command": [speed, 0.0, 0.0], "probability": probability}
        for speed, probability in zip(WALK_SPEEDS, WALK_PROBABILITIES, strict=True)
    ]
    if (
        fresh.get("study") != FRESH_STABILITY_ID
        or fresh.get("arm") != "control"
        or fresh.get("initial_iteration") != 0
        or fresh.get("fresh_optimizers") is not True
        or fresh.get("lin_vel_z_weight") != 0.0
        or not fresh.get("template_checkpoint_sha256")
        or manifest["status"] != "completed"
        or manifest["max_iterations"] != final.TEMPLATE_ITERATION
        or manifest["seed"] != 42
        or manifest["friction_range"] != [final.FRICTION] * 2
        or manifest["applied_friction_range"] != [final.FRICTION] * 2
        or manifest["noise_scale"] != 0.0
        or env["seed"] != 42
        or train["seed"] != 42
        or env["env"]["num_envs"] != final.NUM_ENVS
        or options["reference_mode"] != "additive"
        or options["physics_hz"] != 50
        or options["coordination_reward"] != "paired-contact"
        or options["command_curriculum"] != "stand-walk"
        or not options["clip_joint_targets"]
        or profile.get("command_probabilities") != curriculum
        or env["control"]["gait_freq"] != [2.0, 2.0]
        or dr["startup"]["contact_friction_range"] != [final.FRICTION] * 2
        or dr["startup"]["link_mass_scale_range"] is not None
        or dr["episode"]["scale_ranges"]
        or env["push_robots"]["toggle"]
        or train["actor"]["noise"]
        or train["actor"]["add_noise"]
        or train["runner"]["resume"]
        or train["runner"]["device"] != "cpu"
        or train["runner"]["algorithm_class_name"] != "PPO2"
        or train["runner"]["max_iterations"] != final.TEMPLATE_ITERATION
        or train["runner"]["save_interval"] != final.SAVE_INTERVAL
        or train["algorithm"]["schedule"] != "adaptive"
        or train["algorithm"]["rollout_size"] != final.ROLLOUT_SIZE
        or train["algorithm"]["batch_size"] != final.BATCH_SIZE
        or weights["paired_contact"] != 3.0
        or weights["lin_vel_z"] != 0.0
        or weights.get("trot_support", 0.0) != 0.0
        or weights.get("swing_contact", 0.0) != 0.0
    ):
        raise ValueError("Select the completed, unchanged fresh Control checkpoint")


def state_digest(state):
    """Hash parameter names, tensor layouts, and values independently of torch.save."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        description = (name, str(value.dtype), list(value.shape))
        digest.update(json.dumps(description).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def comparable_controller_profile(profile):
    expected = deepcopy(profile)
    # Class snapshots can include a placeholder for the initializer method.
    for name in ("env_config", "train_config"):
        expected[name].pop("init_member_classes", None)
    return expected


def controller_checkpoint_profile(env, train_cfg):
    profile = {
        "version": PROFILE_VERSION,
        "options": asdict(env.controller_options),
        "ramp_s": RAMP_S,
        "motion_threshold": MOTION_THRESHOLD,
        "env_config": class_to_dict(env.cfg),
        "train_config": class_to_dict(train_cfg),
        "actuated_joint_names": list(env.actuated_dof_names),
        "joint_limits_rad": env._actuated_dof_pos_limits.cpu().tolist(),
        "asset_sha256": asset_hash(env.cfg),
    }
    if env.controller_options.command_curriculum == "stand-walk":
        profile["command_probabilities"] = [
            {"command": [vx, 0.0, 0.0], "probability": probability}
            for vx, probability in zip(WALK_SPEEDS, WALK_PROBABILITIES, strict=True)
        ]
    # Normalize tuples and numeric scalars to the same format as the manifest.
    return json.loads(json.dumps(profile, allow_nan=False))


def asset_hash(env_cfg):
    path = Path(env_cfg.asset.file.format(GYM_ROOT_DIR=GYM_ROOT_DIR))
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_controller_profile(checkpoint, *, expected_iteration=None):
    if not checkpoint.is_file():
        return None
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if expected_iteration is not None and checkpoint_data["iter"] != expected_iteration:
        raise ValueError(f"Select the iteration-{expected_iteration} checkpoint")
    profile = checkpoint_data.get("controller_profile")
    if profile is None:
        manifest = checkpoint.with_name("friction_experiment.json")
        if manifest.is_file() and json.loads(manifest.read_text()).get(
            "controller_profile"
        ):
            raise ValueError("Checkpoint is missing its recorded controller profile")
        return None
    if not isinstance(profile, dict) or profile.get("version") != PROFILE_VERSION:
        raise ValueError("Unsupported controller profile version")
    if (
        profile.get("ramp_s") != RAMP_S
        or profile.get("motion_threshold") != MOTION_THRESHOLD
    ):
        raise ValueError("Controller phase settings differ from the saved profile")
    return profile


def _config_from_snapshot(snapshot, template):
    result = SimpleNamespace()
    for key, value in snapshot.items():
        prototype = getattr(template, key, None)
        if callable(prototype):
            continue
        # Core config containers are objects. Leaf mappings such as actor
        # noise, default joint angles and gain ranges remain dictionaries.
        if (
            isinstance(value, dict)
            and hasattr(prototype, "__dict__")
            and key != "noise"
        ):
            value = _config_from_snapshot(value, prototype)
        setattr(result, key, value)
    return result


def restore_controller_configs(profile, env_template, train_template):
    options = ControllerOptions(**profile["options"])
    env_cfg = _config_from_snapshot(profile["env_config"], env_template)
    train_cfg = _config_from_snapshot(profile["train_config"], train_template)
    if (
        env_cfg.control.ctrl_frequency != ACTOR_HZ
        or train_cfg.actor.frequency != ACTOR_HZ
        or env_cfg.control.desired_sim_frequency != options.physics_hz
        or env_cfg.control.decimation != options.physics_hz // ACTOR_HZ
        or env_cfg.control.ctrl_dt != 1 / ACTOR_HZ
        or env_cfg.sim_dt != 1 / options.physics_hz
    ):
        raise ValueError("Saved controller profile has inconsistent clock settings")
    if asset_hash(env_cfg) != profile["asset_sha256"]:
        raise ValueError("The robot URDF differs from the saved controller profile")
    return env_cfg, train_cfg, options


def verify_controller_layout(env, profile):
    if (
        list(env.actuated_dof_names) != profile["actuated_joint_names"]
        or env._actuated_dof_pos_limits.cpu().tolist() != profile["joint_limits_rad"]
    ):
        raise ValueError("Actuator order or joint limits differ from the saved profile")


def save_controller_checkpoints(runner, profile):
    """Bind the profiled saver while retaining ordinary runner.load support."""
    saved_iterations = set()

    def save():
        iteration = runner.it
        # OnPolicyRunner can request its final iteration twice.
        if iteration in saved_iterations:
            return
        directory = Path(runner.log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / f"model_{iteration}.pt"
        with NamedTemporaryFile(
            dir=directory, suffix=".pt.tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            critic_optimizer_state = runner.alg.critic_optimizer.state_dict()
            torch.save(
                {
                    "actor_state_dict": runner.alg.actor.state_dict(),
                    "critic_state_dict": runner.alg.critic.state_dict(),
                    "optimizer_state_dict": runner.alg.optimizer.state_dict(),
                    "critic_optimizer_state_dict": critic_optimizer_state,
                    "iter": iteration,
                    "controller_profile": profile() if callable(profile) else profile,
                },
                temporary,
            )
            os.link(
                temporary, checkpoint
            )  # Publish atomically without replacing a checkpoint.
        finally:
            temporary.unlink()
        saved_iterations.add(iteration)

    runner.save = save
