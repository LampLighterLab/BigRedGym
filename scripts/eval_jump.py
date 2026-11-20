"""Jump-landing evaluation for go2_se_jump / go2_se_jump_oracle (Step 3).

Walks at a fixed command, triggers jumps at fixed times, and reports per
jump: success (no fall), height gained, airtime, landing slip (foot travel
while in contact in the 0.5 s after touchdown), velocity estimation error
in the 1 s after touchdown, and recovery time (until the speed stays within
0.2 m/s of the command for 0.5 s). Randomization is off except friction.

Usage:
    python scripts/eval_jump.py --task go2_se_jump --load_run jump1 \
        --mu 0.8 0.4 0.2 --vx 1.0
"""

import argparse
import sys

import torch

sys.path.insert(0, "scripts")
import record_go2_se as rec  # noqa: E402


@torch.no_grad()
def run(args, mu):
    env, runner = rec.make(args, mu)
    env.random_jumps = False
    if args.oracle_input:
        runner.oracle = True
    env._reset_idx(torch.ones(1, dtype=torch.bool))
    sc = env.scales["base_lin_vel"]
    sc = sc[0] if torch.is_tensor(sc) and sc.ndim else sc
    dt = env.dt
    jump_steps = {round(t / dt) for t in args.jump_times}
    log = []  # per step: t, z, vx, vx_est, n_contact, slip, fell
    fell_at = None
    for i in range(int(args.seconds / dt)):
        rec.set_command(env, args.vx)
        if i in jump_steps:
            env.trigger_jump()
        runner.set_actions(
            runner.actor_cfg["actions"],
            runner.get_inference_actions(),
            runner.actor_cfg["disable_actions"],
        )
        env.step()
        fell = bool(env.terminated[0])
        slip = (env.foot_contact[0] * env.foot_vel_xy[0]).sum().item() * dt
        log.append(
            (
                i * dt,
                env.root_states[0, 2].item(),
                env.base_lin_vel[0, 0].item(),
                (runner.get_estimate()[0, 0] * sc).item(),
                int(env.foot_contact[0].sum().item()),
                slip,
            )
        )
        if fell:
            fell_at = i * dt
            break
        runner.reset_envs()

    results = []
    for t0 in args.jump_times:
        s0 = round(t0 / dt)
        if s0 >= len(log):
            results.append(dict(t0=t0, success=False))
            continue
        z0 = log[s0][1]
        window = log[s0 : s0 + round(1.0 / dt)]
        airborne = [r for r in window if r[4] == 0]
        # * touchdown: first contact after the first airborne step
        first_air = next((k for k, r in enumerate(window) if r[4] == 0), None)
        td = None
        if first_air is not None:
            td = next(
                (s0 + k for k in range(first_air, len(window)) if window[k][4] > 0), None
            )
        res = dict(
            t0=t0,
            height=max(r[1] for r in window) - z0,
            airtime=len(airborne) * dt,
            success=fell_at is None or fell_at > t0 + 2.0,
        )
        if td is not None and td + round(1.0 / dt) <= len(log):
            after = log[td : td + round(1.0 / dt)]
            res["slip"] = sum(r[5] for r in log[td : td + round(0.5 / dt)])
            res["est_err"] = sum(abs(r[3] - r[2]) for r in after) / len(after)
            need = round(0.5 / dt)
            ok, rec_t = 0, None
            for k in range(td, len(log)):
                ok = ok + 1 if abs(log[k][2] - args.vx) < 0.2 else 0
                if ok >= need:
                    rec_t = (k - need + 1 - td) * dt
                    break
            res["recovery"] = rec_t
        results.append(res)
    return results, fell_at


def fmt(x, spec=".2f"):
    return "  -  " if x is None else format(x, spec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="go2_se_jump")
    p.add_argument("--experiment_name", default=None)
    p.add_argument("--load_run", required=True)
    p.add_argument("--checkpoint", type=int, default=-1)
    p.add_argument("--mu", type=float, nargs="+", default=[0.8, 0.4, 0.2])
    p.add_argument("--vx", type=float, default=1.0)
    p.add_argument("--jump_times", type=float, nargs="+", default=[2.0, 5.0, 8.0])
    p.add_argument("--seconds", type=float, default=11.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--oracle_input",
        action="store_true",
        help="feed the true states instead of the estimate to the same policy",
    )
    args = p.parse_args()
    if args.experiment_name is None:
        args.experiment_name = args.task

    print(f"{args.task} {args.load_run} vx={args.vx}" + (" [true-state input]" if args.oracle_input else ""))
    print("  mu   jump  ok  height  air   slip  est_err  recovery")
    for mu in args.mu:
        results, fell_at = run(args, mu)
        for r in results:
            print(
                f"  {mu:.1f}  {r['t0']:4.1f}  {'y' if r['success'] else 'n'}   "
                f"{fmt(r.get('height'))}   {fmt(r.get('airtime'))}  {fmt(r.get('slip'))}  "
                f"{fmt(r.get('est_err'))}     {fmt(r.get('recovery'))}"
            )
        if fell_at is not None:
            print(f"  {mu:.1f}  fell at t={fell_at:.2f}s")


if __name__ == "__main__":
    main()
