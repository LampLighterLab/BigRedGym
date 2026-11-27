"""Train one Go2trot policy at a friction range, actor noise scale, and seed.

Place in BigRedGym/experiments/go2_friction/ and launch with:
    uv run --frozen experiments/go2_friction/train_friction.py --task go2trot \
        --min-friction 0.6 --max-friction 1.25 --seed 42 \
        --max_iterations 200 --num_envs 256 --headless --disable_wandb

Reuses BigRedGym's *actual* scripts/train.py setup and policy runner.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch

from ablation_telemetry import log_command_metrics
from controller_checkpoint import (
    comparable_controller_profile,
    controller_checkpoint_profile,
    read_controller_profile,
    restore_controller_configs,
    save_controller_checkpoints,
    state_digest,
    verify_finalized_template,
)
from controller_telemetry import contact_force_source
import final_protocol as final
from gait_controller import (
    add_controller_arguments,
    configure_controller,
    controller_options,
    register_controller,
)
from gym.utils.helpers import class_to_dict, get_load_path
from staged_rewards import configure_stage, staged_coordination_rewards
from sweep_friction import (
    REFERENCE_FRICTION,
    experiment_name,
    noise_experiment_name,
    value_name,
)

# Reuse BigRedGym core training code without copying or modifying it.
CORE_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(CORE_SCRIPTS))
from train import make_train_parser, setup, train  # noqa: E402


def configure_friction(
    env_cfg, minimum: float, maximum: float, *, friction_only: bool
) -> None:
    """Change only the requested experimental condition.

    Other DR axes can be disabled completely to isolate friction. No mutation
    of BigRedGym's version-controlled configuration source is required.
    """
    if not math.isfinite(maximum) or not 0.0 <= minimum <= maximum:
        raise ValueError(f"Invalid friction range [{minimum}, {maximum}]")
    dr = env_cfg.domain_randomization
    dr.startup.contact_friction_range = [float(minimum), float(maximum)]
    if friction_only:
        dr.startup.link_mass_scale_range = None
        dr.episode.scale_ranges = {}


def configure_observation_noise(train_cfg, scale: float, *, default_std=None) -> dict:
    if default_std is None:
        defaults = train_cfg.actor.noise
        default_std = {
            "base_ang_vel": defaults.base_ang_vel,
            "projected_gravity": defaults.gravity_vec,
            "dof_pos_obs": defaults.dof_pos_obs,
            "dof_vel": defaults.dof_vel,
        }
    # The on-policy rollout reads this map directly. An empty map produces
    # zero perturbation while preserving the normal rollout's RNG draws.
    train_cfg.actor.noise = (
        {
            name: (torch.as_tensor(std, dtype=torch.float64) * scale).tolist()
            for name, std in default_std.items()
        }
        if scale
        else {}
    )
    train_cfg.actor.add_noise = bool(train_cfg.actor.noise)
    return default_std


def reward_stance_foot_slip(self):
    """Penalize horizontal foot sliding during stance."""
    stance = torch.relu(torch.sin(self._leg_phases()))

    # Only penalize feet actually touching the ground
    normal_force = self.contact_forces[:, self.feet_indices, 2]
    in_contact = (normal_force > 5.0).float()

    # World-frame horizontal foot velocity
    foot_vel_xy = self._rigid_body_lin_vel[:, self.feet_indices, :2]
    speed_squared = foot_vel_xy.square().sum(dim=-1)

    # Normalize by 0.5 m/s and cap large penalties
    slip = (speed_squared / 0.25).clamp(max=4.0)

    return -(stance * in_contact * slip).mean(dim=1)


def reward_swing_foot_clearance(self):
    """Penalize feet that fail to lift during swing."""
    swing = torch.relu(-torch.sin(self._leg_phases()))

    # Initial target: foot-link origin 8 cm above flat ground
    target_z = 0.08
    foot_z = self._rigid_body_pos[:, self.feet_indices, 2]

    deficit = ((target_z - foot_z) / target_z).clamp(min=0.0, max=1.0)

    # Don't require stepping when commanded to stand still
    moving = (
        (self.commands[:, :2].norm(dim=1) > 0.2) | (self.commands[:, 2].abs() > 0.2)
    ).float()

    return -(swing * deficit.square()).mean(dim=1) * moving


def main(argv=None, *, continuation=None) -> None:
    parser = make_train_parser()
    parser.add_argument("--min-friction", type=float, required=True)
    parser.add_argument("--max-friction", type=float, default=REFERENCE_FRICTION)
    parser.add_argument(
        "--noise-scale",
        type=float,
        help="Multiplier for the actor's IMU and joint noise (default: 0, disabled)",
    )
    parser.add_argument(
        "--rollout-size", type=int, help="Override transitions per PPO rollout"
    )
    parser.add_argument(
        "--finalized-template",
        type=Path,
        help="Use the fresh Control configuration for both finalized reward stages",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and verify a finalized training run without updating or saving",
    )

    parser.add_argument(
        "--gait-rewards",
        action="store_true",
        help="Enable foot-slip and swing-clearance penalties",
    )

    parser.add_argument(
        "--easy-gait",
        action="store_true",
        help="Use a simplified forward-walking training curriculum",
    )

    parser.add_argument(
        "--keep-other-randomization",
        action="store_true",
        help="Keep mass and PD-gain randomization instead of friction-only training.",
    )
    add_controller_arguments(parser)
    args = parser.parse_args(argv)
    if args.task != "go2trot":
        parser.error("This experiment is designed for --task go2trot")
    if args.seed is None:
        parser.error("Supply --seed so results are reproducible")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be an integer in [0, 2**32 - 1]")
    if not (
        math.isfinite(args.max_friction) and 0 <= args.min_friction <= args.max_friction
    ):
        parser.error("Friction bounds must be finite and satisfy 0 <= min <= max")
    noise_scale = 0.0 if args.noise_scale is None else args.noise_scale
    if not math.isfinite(noise_scale) or noise_scale < 0:
        parser.error("--noise-scale must be finite and nonnegative")
    if args.rollout_size is not None and args.rollout_size < 1:
        parser.error("--rollout-size must be positive")
    if args.original_cfg:
        parser.error("Do not use --original_cfg with this experiment.")
    if args.resume and args.load_run is None:
        parser.error("Supply --load_run when resuming training.")
    try:
        options = controller_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.finalized_template is not None:
        if options is not None or continuation is not None:
            parser.error("The finalized template supplies all controller options")
        try:
            run_finalized_training(args, noise_scale)
        except (OSError, KeyError, ValueError) as exc:
            parser.error(str(exc))
        return
    if args.dry_run:
        parser.error("--dry-run requires --finalized-template")
    if options is not None and (args.resume or continuation is not None):
        parser.error("Controller pilots start from fresh policy weights")
    if (
        options is not None
        and options.command_curriculum == "stand-walk"
        and args.easy_gait
    ):
        parser.error("Choose one walking curriculum")
    if args.experiment_name is None:
        fixed_reference = args.min_friction == args.max_friction == REFERENCE_FRICTION
        if args.noise_scale is not None and fixed_reference:
            args.experiment_name = noise_experiment_name(noise_scale, args.seed)
        else:
            args.experiment_name = experiment_name(
                args.min_friction, args.seed, upper=args.max_friction
            )
            if noise_scale:
                args.experiment_name += f"_noise_scale_{value_name(noise_scale)}"
        if options is not None:
            args.experiment_name += f"_{options.reference_mode}_{options.physics_hz}hz"

    if args.resume:
        iteration = -1 if args.checkpoint is None else args.checkpoint
        try:
            source = Path(get_load_path(args.experiment_name, args.load_run, iteration))
            resumed_profile = read_controller_profile(source)
        except (IndexError, OSError, ValueError) as exc:
            parser.error(f"Could not load resume checkpoint: {exc}")
        if resumed_profile is not None:
            parser.error("Select a legacy checkpoint for unprofiled continuation")

    # Task registration must occur before asking for the configuration.
    import gym.envs  # noqa: F401
    from gym.utils.task_registry import task_registry

    env_cfg, train_cfg = deepcopy(task_registry.get_cfgs(args.task))
    default_noise = configure_observation_noise(train_cfg, noise_scale)
    if args.rollout_size is not None:
        train_cfg.algorithm.rollout_size = args.rollout_size

    if args.gait_rewards:
        from gym.envs.go2.go2trot import Go2Trot

        Go2Trot._reward_stance_foot_slip = reward_stance_foot_slip
        Go2Trot._reward_swing_foot_clearance = reward_swing_foot_clearance

        train_cfg.critic.reward.weights.stance_foot_slip = 0.5
        train_cfg.critic.reward.weights.swing_foot_clearance = 1.0
    configure_friction(
        env_cfg,
        args.min_friction,
        args.max_friction,
        friction_only=not args.keep_other_randomization,
    )

    # Penalize head and calf contact with the ground
    env_cfg.asset.penalize_contacts_on = ["Head_", "calf"]
    # Positive weight because _reward_collision returns negative values
    train_cfg.critic.reward.weights.collision = 0.5
    train_cfg.runner.save_interval = 25

    if args.easy_gait:
        if not args.gait_rewards:
            parser.error("--easy-gait requires --gait-rewards")

        # Learn forward walking before turning or moving sideways.
        env_cfg.commands.ranges.lin_vel_x = [0.5, 1.0]
        env_cfg.commands.ranges.lin_vel_y = 0.0
        env_cfg.commands.ranges.yaw_vel = 0.0
        env_cfg.commands.var = 0.0

        # Start with a consistent, relatively slow trot.
        env_cfg.control.gait_freq = [1.5, 1.5]

        # No external disturbances during initial gait learning.
        env_cfg.push_robots.toggle = False

        # Give foot lifting and avoiding sliding more importance.
        train_cfg.critic.reward.weights.stance_foot_slip = 1.0
        train_cfg.critic.reward.weights.swing_foot_clearance = 3.0

        # Encourage a level, supported body.
        train_cfg.critic.reward.weights.orientation = 1.5
        train_cfg.critic.reward.weights.min_base_height = 1.0

        # Stronger incentives for alternating diagonal leg support
        train_cfg.critic.reward.weights.trot_support = 6.0
        train_cfg.critic.reward.weights.swing_contact = 5.0
    if continuation is not None:
        continuation.preflight(args, env_cfg, train_cfg)
        if continuation.dry_run:
            return
    if options is not None:
        configure_controller(env_cfg, train_cfg, options)
    run_training(
        args,
        env_cfg,
        train_cfg,
        default_noise,
        noise_scale,
        options=options,
        training_hook=continuation.training if continuation is not None else None,
    )


def run_finalized_training(args, noise_scale):
    import gym.envs  # noqa: F401
    from gym import GYM_ROOT_DIR
    from gym.utils.logging_and_saving import local_code_save_helper
    from gym.utils.task_registry import task_registry

    final.validate_run_name(args.run_name)
    condition = next(
        (
            run
            for run in final.final_training_runs()
            if (run["lower_bound"], run["noise_scale"], run["seed"])
            == (args.min_friction, noise_scale, args.seed)
        ),
        None,
    )
    if condition is None or args.max_friction != final.FRICTION:
        raise ValueError("The condition is outside the finalized friction/noise matrix")
    experiment = condition["experiment_name"]
    if (
        args.backend != "mujoco"
        or args.device != "cpu"
        or args.resume
        or args.load_run is not None
        or args.checkpoint is not None
        or args.easy_gait
        or args.keep_other_randomization
        or args.gait_rewards
        or args.num_envs not in (None, final.NUM_ENVS)
        or args.max_iterations not in (None, final.ITERATIONS)
        or args.rollout_size not in (None, final.ROLLOUT_SIZE)
        or args.batch_size not in (None, final.BATCH_SIZE)
        or args.save_interval not in (None, final.SAVE_INTERVAL)
        or args.experiment_name not in (None, experiment)
    ):
        raise ValueError(
            "Finalized runs require the saved profile and fixed CPU budget"
        )

    template = args.finalized_template.resolve()
    source = json.loads(template.with_name("friction_experiment.json").read_text())
    profile = read_controller_profile(
        template, expected_iteration=final.TEMPLATE_ITERATION
    )
    verify_finalized_template(profile, source)
    with template.open("rb") as stream:
        template_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    directory = Path(GYM_ROOT_DIR) / "logs" / experiment / args.run_name
    if directory.exists():
        raise FileExistsError(f"Run directory already exists: {directory}")

    env_template, train_template = task_registry.get_cfgs("go2trot")
    env, cfg, options = restore_controller_configs(
        profile, env_template, train_template
    )
    cfg.runner.max_iterations = final.ITERATIONS
    configure_friction(env, args.min_friction, args.max_friction, friction_only=True)
    default_noise = configure_observation_noise(deepcopy(train_template), 0.0)
    if default_noise != source["observation_noise"]["default_std"]:
        raise ValueError("Configured actor noise defaults differ from the Control run")
    configure_observation_noise(cfg, noise_scale, default_std=default_noise)
    args.experiment_name = experiment
    args.headless = args.disable_wandb = True

    expected = comparable_controller_profile(profile)
    ec, tc = expected["env_config"], expected["train_config"]
    ec["seed"] = tc["seed"] = args.seed
    ec["domain_randomization"]["startup"]["contact_friction_range"] = [
        args.min_friction,
        args.max_friction,
    ]
    tc["actor"].update(noise=deepcopy(cfg.actor.noise), add_noise=bool(noise_scale))
    tc["log_dir"] = str(directory)
    tc["runner"].update(
        run_name=args.run_name,
        experiment_name=experiment,
        max_iterations=final.ITERATIONS,
    )
    legacy, paired = final.reward_schedule()
    legacy_mode, paired_mode = (
        stage["coordination_reward"] for stage in (legacy, paired)
    )
    initial_profile = deepcopy(expected)
    initial_profile["options"]["coordination_reward"] = legacy_mode
    initial_profile["train_config"]["critic"]["reward"]["weights"].update(
        legacy["weights"]
    )
    options = configure_stage(cfg, options, legacy)

    def check_initialization(runner, actual):
        env = runner.env
        applied = env.domain_randomizer.contact_friction
        if actual != initial_profile:
            raise ValueError(
                "Finalized configuration changed outside the planned factors"
            )
        within_bounds = (applied >= args.min_friction) & (applied <= final.FRICTION)
        if (
            runner.it != 0
            or runner.alg.optimizer.state
            or runner.alg.critic_optimizer.state
            or runner.num_learning_iterations != final.ITERATIONS
            or runner.num_steps_per_env * env.num_envs != final.ROLLOUT_SIZE
            or not within_bounds.all()
            or (args.min_friction < final.FRICTION and applied.unique().numel() < 2)
        ):
            raise ValueError("Unexpected rollout, friction, or nonfresh initialization")
        return {
            "study": final.FINAL_RESEARCH_ID,
            "method": condition["method"],
            "run_name": args.run_name,
            "template_checkpoint": str(template),
            "template_checkpoint_sha256": template_hash,
            "initial_iteration": runner.it,
            "fresh_optimizers": True,
            "initial_actor_sha256": state_digest(runner.alg.actor.state_dict()),
            "initial_critic_sha256": state_digest(runner.alg.critic.state_dict()),
            "reward_schedule": final.reward_schedule(),
            "friction_by_world": applied.tolist(),
            "stages": {
                legacy_mode: {"status": "running", "controller_profile": actual},
                paired_mode: {"status": "pending"},
            },
        }

    def check_condition(runner, details):
        if (
            runner.env.domain_randomizer.contact_friction.tolist()
            != details["friction_by_world"]
            or runner.actor_cfg["noise"] != tc["actor"]["noise"]
        ):
            raise ValueError("Applied friction or actor noise changed between stages")

    def transition(runner, manifest):
        profile = controller_checkpoint_profile(runner.env, cfg)
        if profile != expected:
            raise ValueError(
                "The paired-contact stage changed settings outside its rewards"
            )
        details = manifest["final_research"]
        check_condition(runner, details)
        boundary = {
            "iteration": runner.it,
            "mode": "same_runner",
            "adaptive_learning_rate": runner.alg.learning_rate,
            "actor_learning_rates": [
                group["lr"] for group in runner.alg.optimizer.param_groups
            ],
            "critic_learning_rates": [
                group["lr"] for group in runner.alg.critic_optimizer.param_groups
            ],
        }
        if not args.dry_run:
            checkpoint = directory / f"model_{runner.it}.pt"
            with checkpoint.open("rb") as stream:
                boundary["checkpoint_sha256"] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
        details["transition"] = boundary
        details["stages"][legacy_mode]["status"] = "completed"
        details["stages"][paired_mode].update(
            status="running", controller_profile=profile
        )
        manifest["controller_profile"] = profile
        manifest["train_config"] = profile["train_config"]
        if not args.dry_run:
            write_training_manifest(directory / "friction_experiment.json", manifest)

    @contextmanager
    def training(runner, manifest, log_dir):
        details = check_initialization(runner, manifest["controller_profile"])
        manifest["final_research"] = details
        try:
            with (
                staged_coordination_rewards(
                    runner, cfg, lambda: transition(runner, manifest)
                ),
                log_command_metrics(runner, log_dir / "command_metrics.csv"),
            ):
                yield
            if (
                not details.get("transition")
                or runner.it != final.ITERATIONS
                or any(
                    not (log_dir / f"model_{i}.pt").is_file()
                    for i in range(0, final.ITERATIONS + 1, final.SAVE_INTERVAL)
                )
            ):
                raise RuntimeError(
                    "Finalized run missed the transition or its checkpoints"
                )
            check_condition(runner, details)
        except BaseException:
            mode = paired_mode if details.get("transition") else legacy_mode
            details["stages"][mode]["status"] = "interrupted_or_failed"
            raise
        details["stages"][paired_mode]["status"] = "completed"

    def reserve_directory(config):
        if not args.dry_run:
            directory.mkdir(parents=True, exist_ok=False)
        config.log_dir = str(directory)

    with patch.object(task_registry, "set_log_dir_name", reserve_directory):
        if args.dry_run:
            with patch.object(local_code_save_helper, "save_local_files_to_logs"):
                cfg, runner = setup_training(args, env, cfg, options)
            try:
                details = check_initialization(
                    runner, controller_checkpoint_profile(runner.env, cfg)
                )
                with (
                    patch.object(runner, "save"),
                    staged_coordination_rewards(
                        runner,
                        cfg,
                        lambda: transition(runner, {"final_research": details}),
                    ),
                ):
                    runner.it = final.REWARD_TRANSITION_ITERATION
                    runner.save()
                applied = runner.env.domain_randomizer.contact_friction
                print(
                    json.dumps(
                        {
                            **condition,
                            **{
                                key: value
                                for key, value in details.items()
                                if key not in ("stages", "friction_by_world")
                            },
                            "output": str(directory),
                            "mode": "dry-run",
                            "iterations": runner.num_learning_iterations,
                            "num_envs": runner.env.num_envs,
                            "effective_noise_std": runner.actor_cfg["noise"],
                            "applied_friction_range": [
                                applied.min().item(),
                                applied.max().item(),
                            ],
                        },
                        indent=2,
                    )
                )
            finally:
                runner.env._backend.close()
        else:
            run_training(
                args,
                env,
                cfg,
                default_noise,
                noise_scale,
                options=options,
                training_hook=training,
            )


def setup_training(args, env_cfg, train_cfg, options):
    if options is not None:
        args = copy(args)
        args.task = register_controller(env_cfg, train_cfg, options)
    return setup(args, env_cfg=env_cfg, train_cfg=train_cfg)


def write_training_manifest(path, manifest):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_training(
    args,
    env_cfg,
    train_cfg,
    default_noise,
    noise_scale,
    *,
    options=None,
    training_hook=None,
):
    """Run the shared experiment pipeline with already resolved training configs."""
    print(
        f"Training {args.experiment_name}: "
        f"friction=[{args.min_friction}, {args.max_friction}], "
        f"noise scale={noise_scale}, seed={args.seed}"
    )
    train_cfg, runner = setup_training(args, env_cfg, train_cfg, options)
    log_dir = Path(train_cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    applied_friction = runner.env.domain_randomizer.contact_friction
    manifest = {
        "experiment_name": args.experiment_name,
        "task": args.task,
        "seed": args.seed,
        "friction_range": [args.min_friction, args.max_friction],
        "applied_friction_range": [
            applied_friction.min().item(),
            applied_friction.max().item(),
        ],
        "friction_only": not args.keep_other_randomization and noise_scale == 0.0,
        "noise_scale": noise_scale,
        "observation_noise": {
            "distribution": "gaussian",
            "default_std": default_noise,
            "effective_std": runner.actor_cfg["noise"],
        },
        "gait_rewards_enabled": args.gait_rewards,
        "max_iterations": train_cfg.runner.max_iterations,
        "rollout_size": train_cfg.algorithm.rollout_size,
        "num_envs": env_cfg.env.num_envs,
        "device": args.device,
        "backend": args.backend,
        "train_pushes_enabled": bool(env_cfg.push_robots.toggle),
        "friction_resampling": "once per simulated environment at startup",
        "env_config": class_to_dict(env_cfg),
        "train_config": class_to_dict(train_cfg),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    path = log_dir / "friction_experiment.json"
    if options is not None:
        profile = controller_checkpoint_profile(runner.env, train_cfg)
        manifest["controller_profile"] = profile
        manifest["controller_telemetry"] = {
            "command_metrics": "command_metrics.csv",
            "force_source": contact_force_source(runner.env),
        }
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, indent=2) + "\n")
        save_controller_checkpoints(runner, lambda: manifest["controller_profile"])

    training_context = nullcontext()
    if training_hook is not None:
        training_context = training_hook(runner, manifest, log_dir)
    elif options is not None:
        training_context = log_command_metrics(runner, log_dir / "command_metrics.csv")
    try:
        with training_context:
            write_training_manifest(path, manifest)
            train(train_cfg, runner)
    except BaseException as exc:
        manifest["status"] = "interrupted_or_failed"
        manifest["error"] = repr(exc)
        write_training_manifest(path, manifest)
        raise
    else:
        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_training_manifest(path, manifest)
        print(f"Finished! Log/checkpoints: {log_dir}")


if __name__ == "__main__":
    main()
