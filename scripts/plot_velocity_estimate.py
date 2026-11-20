"""Plot estimated vs true forward velocity of a Go2 estimator policy at
several ground frictions (no rendering). Writes <out>.png and <out>.csv.

Usage:
    python scripts/plot_velocity_estimate.py --task go2_se_go2env \
        --experiment_name go2_se --load_run <run> --mu 0.8 0.4 0.2
"""

import argparse
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "scripts")
import record_go2_se as rec  # noqa: E402

TRUE, EST = "#2a78d6", "#eb6834"  # reference palette slots 1, 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def smooth(x, window):
    x = torch.tensor(x)
    k = torch.ones(window) / window
    pad = torch.cat((x[:1].repeat(window // 2), x, x[-1:].repeat(window - 1 - window // 2)))
    return torch.nn.functional.conv1d(pad.view(1, 1, -1), k.view(1, 1, -1)).flatten()


@torch.no_grad()
def rollout(args, mu):
    env, runner = rec.make(args, mu)
    env._reset_idx(torch.ones(1, dtype=torch.bool))
    scale = env.scales["base_lin_vel"]
    scale = scale[0] if torch.is_tensor(scale) and scale.ndim else scale
    t, vt, ve = [], [], []
    for i in range(int(args.seconds / env.dt)):
        rec.set_command(env, args.vx)
        runner.set_actions(
            runner.actor_cfg["actions"],
            runner.get_inference_actions(),
            runner.actor_cfg["disable_actions"],
        )
        env.step()
        runner.reset_envs()
        t.append(i * env.dt)
        vt.append(env.base_lin_vel[0, 0].item())
        ve.append((runner.get_estimate()[0, 0] * scale).item())
    return t, vt, ve, env.dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="go2_se_go2env")
    p.add_argument("--experiment_name", default="go2_se")
    p.add_argument("--load_run", required=True)
    p.add_argument("--checkpoint", type=int, default=-1)
    p.add_argument("--mu", type=float, nargs="+", default=[0.8, 0.4, 0.2])
    p.add_argument("--vx", type=float, default=1.0)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--smooth_s", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="velocity_estimate_clean")
    args = p.parse_args()

    runs = {mu: rollout(args, mu) for mu in args.mu}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "text.color": INK})
    fig, axes = plt.subplots(len(runs), 1, figsize=(9, 2.6 * len(runs)), sharex=True, sharey=True)
    axes = [axes] if len(runs) == 1 else axes
    rows = []
    for ax, (mu, (t, vt, ve, dt)) in zip(axes, runs.items()):
        w = max(1, round(args.smooth_s / dt))
        ve_s = smooth(ve, w)
        err = (torch.tensor(ve) - torch.tensor(vt)).abs()
        steady = torch.tensor(t) > 1.0
        ax.plot(t, ve, color=EST, lw=0.8, alpha=0.3)
        ax.plot(t, vt, color=TRUE, lw=2, label="true velocity")
        ax.plot(t, ve_s, color=EST, lw=2, label=f"estimated ({args.smooth_s:.1f} s avg)")
        ax.axhline(args.vx, color=MUTED, lw=1, ls="--", label="command")
        ax.set_title(
            f"μ = {mu}:  mean |error| {err[steady].mean():.2f} m/s,  "
            f"true speed {torch.tensor(vt)[steady].mean():.2f} m/s, "
            f"estimated {torch.tensor(ve)[steady].mean():.2f} m/s",
            loc="left", fontsize=10, color=INK,
        )
        ax.set_ylabel("forward velocity [m/s]", color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED)
        rows += [(mu, ti, a, b) for ti, a, b in zip(t, vt, ve)]
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.35), ncol=3, frameon=False)
    axes[-1].set_xlabel("time [s]", color=MUTED)
    fig.suptitle(
        f"Concurrent estimator: estimated vs true forward velocity (command {args.vx} m/s)",
        x=0.01, ha="left", fontsize=12, color=INK, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    with open(args.out + ".csv", "w", newline="") as f:
        csv.writer(f).writerows([("mu", "time_s", "true_vx", "estimated_vx")] + rows)
    print(f"-> {args.out}.png, {args.out}.csv")


if __name__ == "__main__":
    main()
