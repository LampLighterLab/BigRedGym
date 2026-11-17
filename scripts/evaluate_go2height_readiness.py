"""Evaluate Go2Height height, forward-speed, and yaw tracking in MuJoCo."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from gym.envs.base.domain_randomization import apply_domain_randomization_override
from gym.utils.helpers import get_load_path, set_seed
from gym.utils.task_registry import task_registry


CASES = (
    ("fixed_h020", (0.0, 0.0, 0.0), 0.20, 0.20),
    ("fixed_h035", (0.0, 0.0, 0.0), 0.35, 0.35),
    ("forward_020", (0.2, 0.0, 0.0), 0.25, 0.25),
    ("forward_040", (0.4, 0.0, 0.0), 0.25, 0.25),
    ("yaw_pos_050", (0.0, 0.0, 0.5), 0.25, 0.25),
    ("yaw_neg_050", (0.0, 0.0, -0.5), 0.25, 0.25),
    ("step_stop_035_to_020", (0.0, 0.0, 0.0), 0.35, 0.20),
    ("step_forward_035_to_020", (0.2, 0.0, 0.0), 0.35, 0.20),
)


def first_sustained(mask, sustain_steps, dt):
    """Return first time a boolean trace stays true for the requested duration."""
    for start in range(max(0, len(mask) - sustain_steps + 1)):
        if bool(mask[start : start + sustain_steps].all()):
            return round(start * dt, 3)
    return None


def summarize(trace, *, dt, replicas, transition_step):
    results = {}
    measured = slice(transition_step + round(0.5 / dt), None)
    sustain_steps = round(0.5 / dt)
    for case_index, (name, command, _initial_height, final_height) in enumerate(CASES):
        selection = slice(case_index * replicas, (case_index + 1) * replicas)
        alive = trace["alive"][:, selection]
        valid = alive[measured]
        height_error = trace["height"][measured, selection] - final_height
        vx_error = trace["velocity"][measured, selection, 0] - command[0]
        yaw_error = trace["velocity"][measured, selection, 2] - command[2]
        if not bool(valid.any()):
            results[name] = {
                "passed": False,
                "survival_fraction": float(alive[-1].float().mean()),
                "failure": "no valid post-transition samples",
            }
            continue

        def rmse(value):
            return float(value[valid].square().mean().sqrt())

        settle = []
        after = slice(transition_step, None)
        for replica in range(replicas):
            ok = (
                alive[after, replica]
                & (
                    (trace["height"][after, selection][:, replica] - final_height).abs()
                    <= 0.025
                )
                & (
                    (
                        trace["velocity"][after, selection][:, replica, 0] - command[0]
                    ).abs()
                    <= 0.10
                )
                & (
                    (
                        trace["velocity"][after, selection][:, replica, 2] - command[2]
                    ).abs()
                    <= 0.15
                )
            )
            settle.append(first_sustained(ok, sustain_steps, dt))

        metrics = {
            "command": {
                "vx_m_s": command[0],
                "yaw_rad_s": command[2],
                "initial_height_m": _initial_height,
                "final_height_m": final_height,
            },
            "survival_fraction": round(float(alive[-1].float().mean()), 4),
            "height_rmse_m": round(rmse(height_error), 4),
            "vx_rmse_m_s": round(rmse(vx_error), 4),
            "yaw_rmse_rad_s": round(rmse(yaw_error), 4),
            "mean_vx_m_s": round(
                float(trace["velocity"][measured, selection, 0][valid].mean()), 4
            ),
            "mean_yaw_rad_s": round(
                float(trace["velocity"][measured, selection, 2][valid].mean()), 4
            ),
            "max_tilt_deg": round(
                float(trace["tilt_deg"][:, selection][alive].max()), 2
            ),
            "settling_time_s_by_replica": settle,
        }
        is_transition = _initial_height != final_height
        metrics["passed"] = (
            metrics["survival_fraction"] == 1.0
            and metrics["height_rmse_m"] <= 0.025
            and metrics["vx_rmse_m_s"] <= 0.10
            and metrics["yaw_rmse_rad_s"] <= 0.15
            and metrics["max_tilt_deg"] <= 15.0
            and (
                not is_transition
                or all(value is not None and value <= 2.0 for value in settle)
            )
        )
        results[name] = metrics
    return results


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load_run", default="Oct16_18-01-11_height_v1")
    parser.add_argument("--checkpoint", type=int, default=500)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--transition_seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=5110)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replicas < 1 or not 1.0 <= args.transition_seconds <= args.seconds - 1.0:
        parser.error("use at least one replica and leave one second around transition")

    import gym.envs  # noqa: F401

    env_cfg, train_cfg = copy.deepcopy(
        task_registry.get_cfgs("go2height", original_cfg=True, load_run=args.load_run)
    )
    env_cfg.env.num_envs = len(CASES) * args.replicas
    env_cfg.env.episode_length_s = args.seconds + 1
    env_cfg.commands.resampling_time = args.seconds + 1
    env_cfg.push_robots.toggle = False
    env_cfg.init_state.reset_mode = "reset_to_basic"
    apply_domain_randomization_override(env_cfg, "off")
    env_cfg.seed = args.seed
    train_cfg.runner.device = "cpu"
    train_cfg.runner.resume = False
    train_cfg.logging.enable_local_saving = False
    train_cfg.log_dir = None
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    set_seed(args.seed)

    checkpoint = Path(get_load_path("go2height", args.load_run, args.checkpoint))
    env = task_registry.make_env("go2height", env_cfg, "cpu", True, "mujoco")
    try:
        runner = task_registry.make_alg_runner(env, train_cfg)
        runner.load(checkpoint, load_optimizer=False)
        runner.switch_to_eval()
        commands = torch.tensor(
            [case[1] for case in CASES], device=env.device
        ).repeat_interleave(args.replicas, dim=0)
        initial_heights = torch.tensor(
            [case[2] for case in CASES], device=env.device
        ).repeat_interleave(args.replicas)[:, None]
        final_heights = torch.tensor(
            [case[3] for case in CASES], device=env.device
        ).repeat_interleave(args.replicas)[:, None]
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
        records = {key: [] for key in ("alive", "height", "velocity", "tilt_deg")}
        for step in range(total_steps):
            env.commands[:] = commands
            env.base_height_command[:] = (
                initial_heights if step < transition_step else final_heights
            )
            actions = runner.get_inference_actions()
            runner.set_actions(runner.actor_cfg["actions"], actions)
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
    finally:
        env._backend.close()

    trace = {key: torch.stack(value).cpu() for key, value in records.items()}
    results = summarize(
        trace, dt=env.dt, replicas=args.replicas, transition_step=transition_step
    )
    report = {
        "protocol": "go2height-readiness-v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "thresholds": {
            "survival_fraction": 1.0,
            "height_rmse_m_max": 0.025,
            "vx_rmse_m_s_max": 0.10,
            "yaw_rmse_rad_s_max": 0.15,
            "max_tilt_deg_max": 15.0,
            "settling_time_s_max": 2.0,
            "sustain_time_s": 0.5,
        },
        "arguments": vars(args) | {"output": str(args.output)},
        "cases": results,
        "passed_cases": sum(case["passed"] for case in results.values()),
        "total_cases": len(results),
        "ready_for_hardware": all(case["passed"] for case in results.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "ready_for_hardware": report["ready_for_hardware"],
                "passed_cases": report["passed_cases"],
                "total_cases": report["total_cases"],
                "report": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
