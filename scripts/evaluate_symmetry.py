"""Fixed-command evaluation for the Go2 left-right symmetry comparison.

Runs a trained ``go2sym`` policy deterministically on a fixed set of commands
(straight walking at several speeds, mirrored turn and lateral pairs, and
standing), with domain randomization, pushes and observation noise off, and
reports, per command case:

- survival and velocity-tracking RMSE (vx, vy, yaw rate);
- drift: mean body-frame lateral velocity and yaw rate (straight cases);
- policy symmetry error ||mu(M_o s) - M_a mu(s)||^2 on the visited states;
- gait: per-foot duty factor, relative contact phase of each foot w.r.t. FL,
  and the nearest gait template (trot, pace, bound, pronk);
- behaviour symmetry: normalised error between the joint trajectory and its
  mirror image shifted by 0 or half a stride (the better shift is reported);
- torque saturation: fraction of joint-steps at >= 99% of the torque limit;
- mechanical cost of transport  sum_t sum_j |tau_j qdot_j| / (m g distance);
- PCA "synergy" of joint torques and of joint position targets: explained
  variance ratios and the number of components for 90% variance;

plus mirrored-pair tracking asymmetries (left vs right turns / strafes).

Usage (on the GPU machine, from the repo root):
    uv run --frozen scripts/evaluate_symmetry.py --load_run <run_dir> \\
        --device cuda:0 --output eval/<run_name>.json
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch

from gym.envs.base.domain_randomization import apply_domain_randomization_override
from gym.utils.helpers import get_load_path, set_seed
from gym.utils.task_registry import task_registry
from learning.utils.symmetry import build_mirror

# name: (vx, vy, yaw_rate)
COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "fwd_0.5": (0.5, 0.0, 0.0),
    "fwd_1.0": (1.0, 0.0, 0.0),
    "fwd_1.5": (1.5, 0.0, 0.0),
    "fwd_2.0": (2.0, 0.0, 0.0),
    "turn_L_0.5_1.0": (0.5, 0.0, 1.0),
    "turn_R_0.5_1.0": (0.5, 0.0, -1.0),
    "turn_L_1.0_1.0": (1.0, 0.0, 1.0),
    "turn_R_1.0_1.0": (1.0, 0.0, -1.0),
    "spin_L_1.5": (0.0, 0.0, 1.5),
    "spin_R_1.5": (0.0, 0.0, -1.5),
    "strafe_L_0.5": (0.0, 0.5, 0.0),
    "strafe_R_0.5": (0.0, -0.5, 0.0),
    "diag_L_0.5": (0.5, 0.5, 0.0),
    "diag_R_0.5": (0.5, -0.5, 0.0),
}
MIRROR_PAIRS = [
    ("turn_L_0.5_1.0", "turn_R_0.5_1.0"),
    ("turn_L_1.0_1.0", "turn_R_1.0_1.0"),
    ("spin_L_1.5", "spin_R_1.5"),
    ("strafe_L_0.5", "strafe_R_0.5"),
    ("diag_L_0.5", "diag_R_0.5"),
]
STRAIGHT = ["fwd_0.5", "fwd_1.0", "fwd_1.5", "fwd_2.0"]

FOOT_ORDER = ["FL", "FR", "RL", "RR"]
# Relative contact phase of (FR, RL, RR) w.r.t. FL, in cycles.
GAIT_TEMPLATES = {
    "trot": (0.5, 0.5, 0.0),
    "pace": (0.5, 0.0, 0.5),
    "bound": (0.0, 0.5, 0.5),
    "pronk": (0.0, 0.0, 0.0),
}
CONTACT_THRESHOLD_N = 5.0
# A replica whose every foot is in contact more than this fraction of the time
# is not stepping; its gait is not classified.
STATIONARY_DUTY = 0.9
# Farther than this (cycles) from every template -> "irregular".
GAIT_TOLERANCE = 0.125


# * Pure analysis functions on CPU traces -------------------------------------


def circular_distance(a, b):
    """Distance between phases in cycles, in [0, 0.5]."""
    d = (a - b) % 1.0
    return min(d, 1.0 - d)


def stride_frequency(contact, dt, f_min=0.5, f_max=6.0):
    """Dominant frequency of a [time, feet] 0/1 contact signal (Hz) or None."""
    x = contact.double() - contact.double().mean(dim=0)
    if x.abs().sum() == 0 or len(x) < 8:
        return None
    spectrum = torch.fft.rfft(x, dim=0).abs().square().sum(dim=1)
    freq = torch.fft.rfftfreq(len(x), dt)
    band = (freq >= f_min) & (freq <= f_max)
    if not band.any():
        return None
    return float(freq[band][spectrum[band].argmax()])


def relative_phases(contact, dt, frequency):
    """Contact phase of each foot relative to foot 0, in cycles [0, 1)."""
    t = torch.arange(len(contact), dtype=torch.double) * dt
    basis = torch.exp(-2j * math.pi * frequency * t)
    coeff = (contact.double().to(torch.complex128) * basis[:, None]).sum(dim=0)
    phase = torch.angle(coeff) / (2 * math.pi)
    rel = (-(phase - phase[0])) % 1.0  # positive = lags FL
    return [float(p) for p in rel[1:]]


def classify_gait(rel):
    distances = {
        name: max(circular_distance(r, t) for r, t in zip(rel, template))
        for name, template in GAIT_TEMPLATES.items()
    }
    best = min(distances, key=distances.get)
    if distances[best] > GAIT_TOLERANCE:
        return "irregular", distances[best]
    return best, distances[best]


def mirror_shift_error(q, joint_mirror, shift):
    """Normalised || M q(t + shift) - q(t) ||^2 over a [time, joints] trace."""
    if shift >= len(q) - 1:
        return float("nan")
    a = q[: len(q) - shift] if shift else q
    b = joint_mirror(q[shift:]) if shift else joint_mirror(q)
    var = (q - q.mean(dim=0)).square().sum(dim=1).mean()
    return float((b - a).square().sum(dim=1).mean() / var.clamp_min(1e-12))


def pca_explained(x):
    """Explained-variance ratios of centred [samples, features] data."""
    x = x.double() - x.double().mean(dim=0)
    s = torch.linalg.svdvals(x)
    var = s.square()
    return (var / var.sum().clamp_min(1e-12)).tolist()


def components_for(ratios, level=0.9):
    total = 0.0
    for k, r in enumerate(ratios, start=1):
        total += r
        if total >= level:
            return k
    return len(ratios)


def summarize_case(trace, dt, warmup_steps, command, joint_mirror, body_weight):
    alive = trace["alive"].bool()
    valid = alive.clone()
    valid[:warmup_steps] = False
    out = {"survival": float(alive[-1].float().mean())}
    if valid.sum() < 10:
        return out | {"insufficient_samples": True}

    def masked_mean(x):
        return float(x[valid].mean())

    vel = trace["velocity"]  # [t, r, 3] = vx, vy, yaw rate (body frame)
    err = vel - command
    out["vx_rmse"] = float(err[..., 0][valid].square().mean().sqrt())
    out["vy_rmse"] = float(err[..., 1][valid].square().mean().sqrt())
    out["yaw_rmse"] = float(err[..., 2][valid].square().mean().sqrt())
    out["lin_rmse"] = float(err[..., :2].norm(dim=-1)[valid].square().mean().sqrt())
    out["mean_vx"] = masked_mean(vel[..., 0])
    out["mean_vy"] = masked_mean(vel[..., 1])
    out["mean_yaw_rate"] = masked_mean(vel[..., 2])
    out["symmetry_error"] = masked_mean(trace["sym_err"])

    # * Mechanical cost of transport, per surviving replica then averaged
    power = (trace["torque"] * trace["dof_vel"]).abs().sum(dim=-1)  # [t, r]
    speed = vel[..., :2].norm(dim=-1)
    cots = []
    for r in range(alive.shape[1]):
        m = valid[:, r]
        if not alive[-1, r] or m.sum() < 10:
            continue
        work = float(power[m, r].sum() * dt)
        distance = float(speed[m, r].sum() * dt)
        if distance > 0.5:
            cots.append(work / (body_weight * distance))
    out["cot"] = sum(cots) / len(cots) if cots else None
    out["mean_power_w"] = masked_mean(power)
    # * Fraction of joint-steps at the torque limit (sim-to-real readiness)
    out["torque_saturation"] = masked_mean(trace["saturated"])

    # * Gait, behaviour symmetry and PCA on surviving replicas
    contact = trace["load"] > CONTACT_THRESHOLD_N  # [t, r, 4]
    out["duty"] = contact[valid].float().mean(dim=0).tolist()
    gaits, shift_err = [], {"0": [], "half": []}
    stationary = 0
    for r in range(alive.shape[1]):
        if not alive[-1, r]:
            continue
        c = contact[warmup_steps:, r]
        if c.float().mean(dim=0).min() > STATIONARY_DUTY:
            stationary += 1
            continue
        q = trace["dof_pos"][warmup_steps:, r]
        f = stride_frequency(c, dt)
        if f is None:
            continue
        rel = relative_phases(c, dt, f)
        name, dist = classify_gait(rel)
        gaits.append({"freq_hz": f, "rel_phase": rel, "gait": name, "dist": dist})
        half = max(1, round(0.5 / (f * dt)))
        shift_err["0"].append(mirror_shift_error(q, joint_mirror, 0))
        shift_err["half"].append(mirror_shift_error(q, joint_mirror, half))
    out["gaits"] = gaits
    out["stationary_replicas"] = stationary
    if gaits:
        names = [g["gait"] for g in gaits]
        out["gait"] = max(set(names), key=names.count)
        out["stride_freq_hz"] = sum(g["freq_hz"] for g in gaits) / len(gaits)
        e0 = sum(shift_err["0"]) / len(shift_err["0"])
        eh = sum(shift_err["half"]) / len(shift_err["half"])
        out["mirror_shift_error"] = {"0": e0, "half": eh}
        out["behaviour_symmetry_error"] = min(e0, eh)
        out["behaviour_symmetry_shift"] = "0" if e0 <= eh else "half"

    for key in ("torque", "target"):
        x = trace[key][valid]
        ratios = pca_explained(x)
        out[f"pca_{key}_ratios"] = ratios
        out[f"pca_{key}_k90"] = components_for(ratios, 0.9)
    return out


def summarize(cases):
    pairs = {}
    for left, right in MIRROR_PAIRS:
        a, b = cases[left], cases[right]
        if "lin_rmse" not in a or "lin_rmse" not in b:
            continue
        pairs[f"{left}|{right}"] = {
            "lin_rmse_gap": abs(a["lin_rmse"] - b["lin_rmse"]),
            "yaw_rmse_gap": abs(a["yaw_rmse"] - b["yaw_rmse"]),
            "survival_gap": abs(a["survival"] - b["survival"]),
        }
    straight = [cases[n] for n in STRAIGHT if "lin_rmse" in cases[n]]

    def mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    return {
        "mean_symmetry_error": mean([c.get("symmetry_error") for c in cases.values()]),
        "mean_survival": mean([c["survival"] for c in cases.values()]),
        "mean_lin_rmse": mean([c.get("lin_rmse") for c in cases.values()]),
        "mean_torque_saturation": mean(
            [c.get("torque_saturation") for c in cases.values()]
        ),
        "straight_abs_vy": mean([abs(c["mean_vy"]) for c in straight]),
        "straight_abs_yaw_rate": mean([abs(c["mean_yaw_rate"]) for c in straight]),
        "pair_lin_rmse_gap": mean([p["lin_rmse_gap"] for p in pairs.values()]),
        "pair_yaw_rmse_gap": mean([p["yaw_rmse_gap"] for p in pairs.values()]),
        "straight_behaviour_symmetry_error": mean(
            [c.get("behaviour_symmetry_error") for c in straight]
        ),
        "straight_gaits": {n: cases[n].get("gait") for n in STRAIGHT},
        "straight_cot": {n: cases[n].get("cot") for n in STRAIGHT},
        "straight_pca_torque_k90": {
            n: cases[n].get("pca_torque_k90") for n in STRAIGHT
        },
        "straight_pca_target_k90": {
            n: cases[n].get("pca_target_k90") for n in STRAIGHT
        },
        "pairs": pairs,
    }


# * Simulation ------------------------------------------------------------------


@torch.no_grad()
def evaluate(env, runner, *, replicas, seconds, warmup):
    names = list(COMMANDS)
    commands = torch.tensor(
        [COMMANDS[n] for n in names], device=env.device
    ).repeat_interleave(replicas, dim=0)
    obs_mirror = build_mirror(env, runner.actor_cfg["obs"]).to(env.device)
    act_mirror = build_mirror(env, runner.actor_cfg["actions"]).to(env.device)
    joint_mirror = build_mirror(env, ["dof_pos"])
    actor = runner.alg.actor

    env.commands[:] = commands
    alive = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    keys = ("alive", "velocity", "sym_err", "torque", "dof_vel", "dof_pos")
    records = {k: [] for k in keys + ("target", "load", "saturated")}
    for _ in range(round(seconds / env.dt)):
        env.commands[:] = commands  # pin commands against resampling
        obs = runner.get_obs(runner.actor_cfg["obs"])
        mu = actor.act_inference(obs)
        mu_m = actor.act_inference(obs_mirror(obs))
        sym_err = (mu_m - act_mirror(mu)).square().sum(dim=-1)
        runner.set_actions(runner.actor_cfg["actions"], mu)
        env.step()
        alive &= ~env.terminated
        values = {
            "alive": alive,
            "velocity": torch.cat(
                (env.base_lin_vel[:, :2], env.base_ang_vel[:, 2:3]), 1
            ),
            "sym_err": sym_err,
            "torque": env.torques,
            "saturated": (env.torques.abs() >= 0.99 * env.actuated_torque_limits)
            .float()
            .mean(dim=-1),
            "dof_vel": env.dof_vel,
            "dof_pos": env.dof_pos,
            "target": env.dof_pos_target,
            "load": env.contact_forces[:, env.feet_indices, 2],
        }
        for key, value in values.items():
            records[key].append(value.detach().clone())
    trace = {k: torch.stack(v).cpu() for k, v in records.items()}
    for key, value in trace.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise RuntimeError(f"Nonfinite values in {key}")

    body_weight = float(env._backend.link_mass[0].sum()) * 9.81
    joint_mirror = joint_mirror.to("cpu")
    cases = {}
    for i, name in enumerate(names):
        sel = slice(i * replicas, (i + 1) * replicas)
        cases[name] = summarize_case(
            {k: v[:, sel] for k, v in trace.items()},
            env.dt,
            round(warmup / env.dt),
            torch.tensor(COMMANDS[name]),
            joint_mirror,
            body_weight,
        ) | {"command": list(COMMANDS[name])}
    return cases, body_weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_name", default="go2sym")
    parser.add_argument("--load_run", required=True)
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument(
        "--init",
        choices=("basic", "range"),
        default="basic",
        help="basic: every replica starts from the same symmetric standing "
        "pose; range: the task's randomized initial states",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seconds - args.warmup < 3:
        parser.error("need at least 3 measured seconds")
    import gym.envs  # noqa: F401

    env_cfg, train_cfg = copy.deepcopy(task_registry.get_cfgs("go2sym"))
    env_cfg.env.num_envs = len(COMMANDS) * args.replicas
    env_cfg.env.episode_length_s = args.seconds + 1
    env_cfg.commands.resampling_time = args.seconds + 1
    env_cfg.push_robots.toggle = False
    env_cfg.control.action_latency_steps_range = None  # no action delay
    env_cfg.init_state.reset_mode = (
        "reset_to_basic" if args.init == "basic" else "reset_to_range"
    )
    apply_domain_randomization_override(env_cfg, "off")
    env_cfg.seed = args.seed
    train_cfg.runner.device = args.device
    train_cfg.runner.resume = False
    train_cfg.logging.enable_local_saving = False
    train_cfg.log_dir = None
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    set_seed(args.seed)
    path = Path(
        get_load_path(
            name=args.experiment_name,
            load_run=args.load_run,
            checkpoint=args.checkpoint,
        )
    )
    env = task_registry.make_env("go2sym", env_cfg, args.device, True, "mujoco")
    try:
        runner = task_registry.make_alg_runner(env, train_cfg)
        runner.load(path, load_optimizer=False)
        runner.switch_to_eval()
        cases, body_weight = evaluate(
            env,
            runner,
            replicas=args.replicas,
            seconds=args.seconds,
            warmup=args.warmup,
        )
        foot_order = list(env.robot_layout.body_groups["feet"])
    finally:
        env._backend.close()
    if [f.split("_")[0] for f in foot_order] != FOOT_ORDER:
        raise RuntimeError(f"Unexpected foot order {foot_order}")
    report = {
        "protocol": "go2sym-eval-v2",
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "arguments": vars(args) | {"output": str(args.output)},
        "dt": env_cfg.control.ctrl_dt,
        "body_weight_n": body_weight,
        "summary": summarize(cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.output)} | report["summary"], indent=2))


if __name__ == "__main__":
    main()
