"""Sim-to-real readiness check for go2sym policies.

Runs one checkpoint under a set of perturbation conditions that mimic the
real Go2 deployment pipeline (``go2_deploy``) and reports, per condition,
survival, velocity tracking, left/right asymmetry, torque saturation and
joint-velocity margin, plus pass/fail against fixed thresholds.

Conditions (all with the training domain randomization and observation
noise, sampled per replica):
- nominal:       training conditions
- deploy_gains:  Kp, Kd from go2_deploy/deploy_config.py (30, 2.0 vs 20, 0.5)
- ema:           the deployment action smoothing a <- alpha a_new + (1-alpha) a
                 with alpha = DeployConfig.ema_smoothing_factor (0.1)
- latency_1/2:   the PD law acts on the target from 1 / 2 control steps
                 (10 / 20 ms) earlier, while the policy observes the target it
                 commanded, as on the robot (Go2 action_latency_steps_range);
                 every other condition uses no delay
- noise_x2:      observation noise doubled
- push:          random base-velocity pushes (0.5 m/s, every 2 s)
- deploy_full:   deploy gains + ema + latency_1 together (the real pipeline
                 as currently configured, plus one step of transport delay)
- deploy_matched: the pipeline we would actually deploy with: training gains,
                 no smoothing, one step of delay, plus pushes

Usage:
    uv run --frozen scripts/evaluate_sim2real.py --load_run <run_dir> \\
        --device cuda:0 --output eval/<run>_sim2real.json
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from gym.utils.helpers import get_load_path, set_seed
from gym.utils.task_registry import task_registry
from learning.utils.symmetry import build_mirror

COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "fwd_0.5": (0.5, 0.0, 0.0),
    "fwd_1.0": (1.0, 0.0, 0.0),
    "turn_L": (0.5, 0.0, 1.0),
    "turn_R": (0.5, 0.0, -1.0),
    "strafe_L": (0.0, 0.5, 0.0),
    "strafe_R": (0.0, -0.5, 0.0),
}
PAIRS = [("turn_L", "turn_R"), ("strafe_L", "strafe_R")]
MOVING = [n for n in COMMANDS if n != "stand"]

# Deployment values (go2_deploy/deploy_config.py at the time of writing).
DEPLOY_KP, DEPLOY_KD, DEPLOY_EMA = 30.0, 2.0, 0.1

CONDITIONS = {
    "nominal": {},
    "deploy_gains": {"kp": DEPLOY_KP, "kd": DEPLOY_KD},
    "ema": {"ema": DEPLOY_EMA},
    "latency_1": {"latency": 1},
    "latency_2": {"latency": 2},
    "noise_x2": {"noise": 2.0},
    "push": {"push": True},
    "deploy_full": {"kp": DEPLOY_KP, "kd": DEPLOY_KD, "ema": DEPLOY_EMA, "latency": 1},
    "deploy_matched": {"latency": 1, "push": True},
}

# Pass thresholds (sim-to-real readiness criteria).
THRESHOLDS = {
    "min_survival": 0.95,  # every command case
    "max_lin_rmse": 0.25,  # m/s, mean over moving commands
    "max_yaw_rmse": 0.35,  # rad/s, mean over moving commands
    "max_torque_saturation": 0.05,  # fraction of joint-steps at the limit
    "max_qdot_ratio_p99": 0.9,  # 99th pct of |qdot| / velocity limit
}


def _scaled_noise(noise_cfg, factor):
    return {
        k: (factor * torch.as_tensor(v)).tolist() if k != "scale" else v
        for k, v in noise_cfg.items()
    }


@torch.no_grad()
def run_condition(name, cond, args, path):
    import gym.envs  # noqa: F401

    env_cfg, train_cfg = copy.deepcopy(task_registry.get_cfgs("go2sym"))
    env_cfg.env.num_envs = len(COMMANDS) * args.replicas
    env_cfg.env.episode_length_s = args.seconds + 1
    env_cfg.commands.resampling_time = args.seconds + 1
    env_cfg.init_state.reset_mode = "reset_to_range"
    env_cfg.push_robots.toggle = bool(cond.get("push"))
    env_cfg.push_robots.interval_s = 2.0
    env_cfg.push_robots.max_push_vel_xy = 0.5
    k = int(cond.get("latency", 0))
    env_cfg.control.action_latency_steps_range = [k, k] if k else None
    if "kp" in cond:
        env_cfg.control.stiffness = {k: cond["kp"] for k in env_cfg.control.stiffness}
        env_cfg.control.damping = {k: cond["kd"] for k in env_cfg.control.damping}
    env_cfg.seed = args.seed
    train_cfg.runner.device = args.device
    train_cfg.runner.resume = False
    train_cfg.logging.enable_local_saving = False
    train_cfg.log_dir = None
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    set_seed(args.seed)
    env = task_registry.make_env("go2sym", env_cfg, args.device, True, "mujoco")
    try:
        runner = task_registry.make_alg_runner(env, train_cfg)
        runner.load(path, load_optimizer=False)
        runner.switch_to_eval()
        actor = runner.alg.actor
        obs_names = runner.actor_cfg["obs"]
        act_names = runner.actor_cfg["actions"]
        noise = _scaled_noise(runner.actor_cfg["noise"], cond.get("noise", 1.0))
        commands = torch.tensor(
            [COMMANDS[n] for n in COMMANDS], device=env.device
        ).repeat_interleave(args.replicas, dim=0)
        alpha = cond.get("ema")
        vel_limit = env.dof_vel_limits.index_select(0, env.actuated_dof_indices)

        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        rec = {k: [] for k in ("alive", "vel", "sat", "qdot_ratio")}
        for _ in range(round(args.seconds / env.dt)):
            env.commands[:] = commands
            obs = runner.get_noisy_obs(obs_names, noise)
            applied = actor.act_inference(obs)
            if alpha is not None:
                previous = env.get_states(act_names).to(applied.device)
                applied = alpha * applied + (1.0 - alpha) * previous
            runner.set_actions(act_names, applied)
            env.step()
            alive &= ~env.terminated
            rec["alive"].append(alive.clone())
            rec["vel"].append(
                torch.cat((env.base_lin_vel[:, :2], env.base_ang_vel[:, 2:3]), 1)
            )
            rec["sat"].append(
                (env.torques.abs() >= 0.99 * env.actuated_torque_limits)
                .float()
                .mean(dim=-1)
            )
            qdot = env.dof_vel.index_select(1, env.actuated_dof_indices)
            rec["qdot_ratio"].append((qdot.abs() / vel_limit).max(dim=-1).values)
        trace = {k: torch.stack(v).cpu() for k, v in rec.items()}
        mirror_cmd = build_mirror(env, ["commands"])  # sanity: pairs are mirrors
    finally:
        env._backend.close()

    for left, right in PAIRS:
        assert torch.allclose(
            mirror_cmd.to("cpu")(torch.tensor(COMMANDS[left])),
            torch.tensor(COMMANDS[right]),
        )

    warm = round(args.warmup / env_cfg.control.ctrl_dt)
    cases = {}
    for i, cname in enumerate(COMMANDS):
        sel = slice(i * args.replicas, (i + 1) * args.replicas)
        a = trace["alive"][:, sel]
        valid = a.clone()
        valid[:warm] = False
        err = trace["vel"][:, sel] - torch.tensor(COMMANDS[cname])
        c = {"survival": float(a[-1].float().mean())}
        if valid.sum() > 10:
            c["lin_rmse"] = float(
                err[..., :2].norm(dim=-1)[valid].square().mean().sqrt()
            )
            c["yaw_rmse"] = float(err[..., 2][valid].square().mean().sqrt())
            c["torque_saturation"] = float(trace["sat"][:, sel][valid].mean())
            c["qdot_ratio_p99"] = float(
                torch.quantile(trace["qdot_ratio"][:, sel][valid], 0.99)
            )
        cases[cname] = c

    def mean(key, names):
        vals = [cases[n][key] for n in names if key in cases[n]]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "min_survival": min(c["survival"] for c in cases.values()),
        "lin_rmse": mean("lin_rmse", MOVING),
        "yaw_rmse": mean("yaw_rmse", MOVING),
        "torque_saturation": mean("torque_saturation", list(COMMANDS)),
        "qdot_ratio_p99": max(
            (c["qdot_ratio_p99"] for c in cases.values() if "qdot_ratio_p99" in c),
            default=None,
        ),
        "pair_yaw_gap": sum(
            abs(cases[left].get("yaw_rmse", 0) - cases[right].get("yaw_rmse", 0))
            for left, right in PAIRS
        )
        / len(PAIRS),
    }
    checks = {
        "survival": summary["min_survival"] >= THRESHOLDS["min_survival"],
        "lin_rmse": summary["lin_rmse"] is not None
        and summary["lin_rmse"] <= THRESHOLDS["max_lin_rmse"],
        "yaw_rmse": summary["yaw_rmse"] is not None
        and summary["yaw_rmse"] <= THRESHOLDS["max_yaw_rmse"],
        "torque_saturation": summary["torque_saturation"] is not None
        and summary["torque_saturation"] <= THRESHOLDS["max_torque_saturation"],
        "qdot": summary["qdot_ratio_p99"] is not None
        and summary["qdot_ratio_p99"] <= THRESHOLDS["max_qdot_ratio_p99"],
    }
    return {
        "condition": cond,
        "summary": summary,
        "checks": checks,
        "pass": all(checks.values()),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_name", default="go2sym")
    parser.add_argument("--load_run", required=True)
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replicas", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument(
        "--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS)
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = Path(
        get_load_path(
            name=args.experiment_name,
            load_run=args.load_run,
            checkpoint=args.checkpoint,
        )
    )
    results = {}
    for name in args.conditions:
        results[name] = run_condition(name, CONDITIONS[name], args, path)
        s = results[name]["summary"]
        print(
            f"{name:13s} pass={results[name]['pass']!s:5s} "
            f"surv={s['min_survival']:.2f} lin={s['lin_rmse']} yaw={s['yaw_rmse']} "
            f"sat={s['torque_saturation']} qdot99={s['qdot_ratio_p99']}"
        )
    report = {
        "protocol": "go2sym-sim2real-v2",
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "arguments": vars(args) | {"output": str(args.output)},
        "thresholds": THRESHOLDS,
        "pass_all": all(r["pass"] for r in results.values()),
        "conditions": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
