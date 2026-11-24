"""Compare frozen Go2Trot/Go2Height switching and live action blending."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from gym.envs.base.domain_randomization import apply_domain_randomization_override
from gym.utils.helpers import get_load_path, set_seed
from gym.utils.task_registry import task_registry


METHODS = (
    ("b_alone", None),
    ("direct", 0.0),
    ("blend_025", 0.25),
    ("blend_050", 0.50),
    ("blend_100", 1.00),
)
SPEEDS = (0.20, 0.40)
CASES = tuple(
    (f"{method}_v{speed:.2f}", method, duration, speed)
    for method, duration in METHODS
    for speed in SPEEDS
)


def first_sustained(mask, sustain_steps, dt):
    for start in range(max(0, len(mask) - sustain_steps + 1)):
        if bool(mask[start : start + sustain_steps].all()):
            return round(start * dt, 3)
    return None


def blend_alpha(method, duration, elapsed):
    if method == "b_alone":
        return 1.0
    if elapsed < 0:
        return 0.0
    if duration == 0.0:
        return 1.0
    return min(elapsed / duration, 1.0)


def configure(env_cfg, train_cfg, *, num_envs, seconds, seed):
    env_cfg.env.num_envs = num_envs
    env_cfg.env.episode_length_s = seconds + 1
    env_cfg.commands.resampling_time = seconds + 1
    env_cfg.push_robots.toggle = False
    env_cfg.init_state.reset_mode = "reset_to_basic"
    apply_domain_randomization_override(env_cfg, "off")
    env_cfg.seed = seed
    train_cfg.runner.device = "cpu"
    train_cfg.runner.resume = False
    train_cfg.logging.enable_local_saving = False
    train_cfg.log_dir = None
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)


def summarize(trace, *, dt, replicas, transition_step):
    measured = slice(transition_step + round(0.5 / dt), None)
    pre = slice(transition_step - round(0.5 / dt), transition_step)
    sustain_steps = round(0.5 / dt)
    transition_window = slice(
        transition_step - 1, transition_step + round(1.0 / dt) + 1
    )
    action_step = trace["target"].diff(dim=0).abs().amax(dim=-1)
    results = {}
    for case_index, (name, method, duration, speed) in enumerate(CASES):
        selection = slice(case_index * replicas, (case_index + 1) * replicas)
        alive = trace["alive"][:, selection]
        valid = alive[measured]
        pre_valid = alive[pre]

        settle = []
        success = []
        for replica in range(replicas):
            ok = (
                alive[transition_step:, replica]
                & (
                    (
                        trace["height"][transition_step:, selection][:, replica] - 0.20
                    ).abs()
                    <= 0.025
                )
                & (
                    trace["velocity"][transition_step:, selection][:, replica, 0].abs()
                    <= 0.10
                )
                & (
                    trace["velocity"][transition_step:, selection][:, replica, 2].abs()
                    <= 0.15
                )
            )
            value = first_sustained(ok, sustain_steps, dt)
            settle.append(value)
            success.append(
                bool(alive[-1, replica]) and value is not None and value <= 2.0
            )

        def rmse(value, mask):
            return (
                float(value[mask].square().mean().sqrt()) if bool(mask.any()) else None
            )

        height_error = trace["height"][measured, selection] - 0.20
        pre_vx_error = trace["velocity"][pre, selection, 0] - speed
        results[name] = {
            "method": method,
            "blend_duration_s": duration,
            "commanded_initial_vx_m_s": speed,
            "mean_initial_vx_m_s": round(
                float(trace["velocity"][pre, selection, 0][pre_valid].mean()), 4
            ),
            "initial_vx_rmse_m_s": round(rmse(pre_vx_error, pre_valid), 4),
            "survival_fraction": round(float(alive[-1].float().mean()), 4),
            "success_rate": round(sum(success) / replicas, 4),
            "settling_time_s_by_replica": settle,
            "steady_height_rmse_m": round(rmse(height_error, valid), 4),
            "max_tilt_deg": round(
                float(trace["tilt_deg"][:, selection][alive].max()), 2
            ),
            "peak_joint_target_step_rad": round(
                float(action_step[transition_window, selection].max()), 4
            ),
        }
    return results


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trot_run", default="Nov16_20-08-42_trot_v1")
    parser.add_argument("--trot_checkpoint", type=int, default=500)
    parser.add_argument("--height_run", default="Oct16_18-01-11_height_v1")
    parser.add_argument("--height_checkpoint", type=int, default=500)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--transition_seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=5110)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.replicas < 1
        or args.transition_seconds < 1.0
        or args.seconds - args.transition_seconds < 2.5
    ):
        parser.error("leave at least 1 s before and 2.5 s after the transition")

    import gym.envs  # noqa: F401

    trot_env_cfg, trot_train_cfg = copy.deepcopy(
        task_registry.get_cfgs("go2trot", original_cfg=True, load_run=args.trot_run)
    )
    height_env_cfg, height_train_cfg = copy.deepcopy(
        task_registry.get_cfgs("go2height", original_cfg=True, load_run=args.height_run)
    )
    num_envs = len(CASES) * args.replicas
    configure(
        trot_env_cfg,
        trot_train_cfg,
        num_envs=num_envs,
        seconds=args.seconds,
        seed=args.seed,
    )
    configure(
        height_env_cfg,
        height_train_cfg,
        num_envs=num_envs,
        seconds=args.seconds,
        seed=args.seed,
    )
    compatibility = {
        "action_names_equal": (
            trot_train_cfg.actor.actions == height_train_cfg.actor.actions
        ),
        "action_scaling_equal": (
            trot_env_cfg.scaling.dof_pos_target == height_env_cfg.scaling.dof_pos_target
        ),
        "control_frequency_equal": (
            trot_env_cfg.control.ctrl_frequency == height_env_cfg.control.ctrl_frequency
        ),
    }
    if not all(compatibility.values()):
        raise ValueError(f"incompatible frozen policies: {compatibility}")
    set_seed(args.seed)

    trot_path = Path(get_load_path("go2trot", args.trot_run, args.trot_checkpoint))
    height_path = Path(
        get_load_path("go2height", args.height_run, args.height_checkpoint)
    )
    env = task_registry.make_env("go2height", height_env_cfg, "cpu", True, "mujoco")
    try:
        trot_runner = task_registry.make_alg_runner(env, trot_train_cfg)
        height_runner = task_registry.make_alg_runner(env, height_train_cfg)
        trot_runner.load(trot_path, load_optimizer=False)
        height_runner.load(height_path, load_optimizer=False)
        trot_runner.switch_to_eval()
        height_runner.switch_to_eval()

        speeds = torch.tensor(
            [case[3] for case in CASES], device=env.device
        ).repeat_interleave(args.replicas)
        phase = torch.arange(args.replicas, device=env.device) * (
            2 * torch.pi / args.replicas
        )
        env.phase[:, 0] = phase.repeat(len(CASES))
        env.phase_frequency[:] = 2.0
        env._update_phase_observation()
        env._update_gait_reference()

        total_steps = round(args.seconds / env.dt)
        transition_step = round(args.transition_seconds / env.dt)
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        records = {
            key: [] for key in ("alive", "height", "velocity", "tilt_deg", "target")
        }
        for step in range(total_steps):
            elapsed = (step - transition_step) * env.dt
            before = step < transition_step
            env.commands.zero_()
            if before:
                env.commands[:, 0] = speeds
                env.base_height_command[:] = 0.35
            else:
                env.base_height_command[:] = 0.20

            trot_actions = trot_runner.get_inference_actions()
            height_actions = height_runner.get_inference_actions()
            case_alpha = torch.tensor(
                [
                    blend_alpha(method, duration, elapsed)
                    for _, method, duration, _ in CASES
                ],
                dtype=trot_actions.dtype,
                device=env.device,
            ).repeat_interleave(args.replicas)[:, None]
            actions = trot_actions + case_alpha * (height_actions - trot_actions)
            height_runner.set_actions(height_runner.actor_cfg["actions"], actions)
            requested_target = env.dof_pos_target.clone()
            env.step()
            alive &= ~env.terminated
            gravity = env.projected_gravity
            tilt = torch.rad2deg(
                torch.acos((-gravity[:, 2] / gravity.norm(dim=1)).clamp(-1.0, 1.0))
            )
            records["alive"].append(alive.clone())
            records["height"].append(env.base_height[:, 0].clone())
            records["velocity"].append(
                torch.cat((env.base_lin_vel[:, :2], env.base_ang_vel[:, 2:3]), dim=1)
            )
            records["tilt_deg"].append(tilt)
            records["target"].append(requested_target)
    finally:
        env._backend.close()

    trace = {key: torch.stack(value).cpu() for key, value in records.items()}
    results = summarize(
        trace, dt=env.dt, replicas=args.replicas, transition_step=transition_step
    )
    method_scores = {}
    for method, _ in METHODS:
        selected = [case for case in results.values() if case["method"] == method]
        method_scores[method] = round(
            sum(case["success_rate"] for case in selected) / len(selected), 4
        )
    report = {
        "protocol": "go2-frozen-composition-v1",
        "policies_frozen": True,
        "trot_checkpoint": str(trot_path.resolve()),
        "trot_sha256": hashlib.sha256(trot_path.read_bytes()).hexdigest(),
        "height_checkpoint": str(height_path.resolve()),
        "height_sha256": hashlib.sha256(height_path.read_bytes()).hexdigest(),
        "compatibility": compatibility,
        "criteria": {
            "height_error_m_max": 0.025,
            "forward_speed_m_s_max": 0.10,
            "yaw_error_rad_s_max": 0.15,
            "settling_time_s_max": 2.0,
            "sustain_time_s": 0.5,
        },
        "arguments": vars(args) | {"output": str(args.output)},
        "method_mean_success_rate": method_scores,
        "cases": results,
        "preliminary_only": True,
        "ready_for_hardware": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"method_mean_success_rate": method_scores, "report": str(args.output)}
        )
    )


if __name__ == "__main__":
    main()
