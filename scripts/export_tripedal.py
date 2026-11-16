"""Export a three-legged (one leg locked) policy for the real robot.

The trained network outputs targets for all 12 joints, in the simulator's
scaled units; in simulation the locked leg's three outputs are ignored and a
separate PD controller holds that leg in its lock pose. This export makes
that explicit, so the robot-side code cannot send policy outputs to the
locked leg:

- policy.pt / policy.onnx: takes the observation vector exactly as in
  training and returns ABSOLUTE joint position targets [rad] for the NINE
  free joints only (order in deploy.json "free_joints"). The locked leg's
  outputs are dropped inside the exported model.
- deploy.json: everything the robot side needs: free and locked joint
  names, PD gains and default pose of the free joints, the locked leg's
  hold command (pose, stiffness, damping, as in simulation), the
  observation layout with each part's size and scale, and the control rate.

On the robot, every control step:
    obs = concat(part / scale for each part in deploy.json "observation")
    targets = policy(obs)                      # 9 free joints, rad
    free joints:   q_des = targets, kp, kd     # deploy.json "free_joints"
    locked joints: q_des = lock pose, kp = 40, kd = 1  # never the policy
The locked leg's "dof_pos_target" observation entries must be 0 (as in
training); its joint positions and speeds are the measured ones.

Usage:
    uv run --frozen scripts/export_tripedal.py --task go2osc_wide_even2.5_deploy --load_run <run> [--out exports/<name>]
"""

import argparse
import json
import os
from pathlib import Path

import torch

from gym.utils.helpers import GYM_ROOT_DIR


class FreeJointPolicy(torch.nn.Module):
    """The actor followed by the action scaling and default pose, keeping
    only the free joints' outputs."""

    def __init__(self, actor, free, scale, default):
        super().__init__()
        self.actor = actor
        self.register_buffer("free", torch.as_tensor(free, dtype=torch.long))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float))
        self.register_buffer("default", torch.as_tensor(default, dtype=torch.float))

    def forward(self, obs):
        action = self.actor.act_inference(obs)
        target = action * self.scale + self.default
        return target.index_select(-1, self.free)


def tolist(x):
    return x.detach().cpu().flatten().tolist() if torch.is_tensor(x) else list(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--load_run", required=True)
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import play

    import gym.envs.go2.go2osc_config  # noqa: F401  (registers the study tasks)

    play_argv = [
        "--task",
        args.task,
        "--load_run",
        args.load_run,
        "--headless",
        "--no-keyboard",
        "--no-record",
        "--num_envs",
        "1",
        "--device",
        "cpu",
        "--checkpoint",
        str(args.checkpoint),
    ]
    with torch.no_grad():
        env, runner, load_path = play.setup(play.get_play_args(play_argv))
    names = list(env.actuated_dof_names)
    locked = env._locked_dofs.tolist()
    if not locked:
        raise SystemExit(f"{args.task} has no locked leg")
    free = [i for i in range(len(names)) if i not in locked]
    actions = runner.actor_cfg["actions"]
    if actions != ["dof_pos_target"]:
        raise SystemExit(f"unexpected actions {actions}")
    scale = env.scales["dof_pos_target"]
    scale = tolist(scale) if hasattr(scale, "__len__") else [float(scale)] * len(names)
    default = tolist(env.default_dof_pos.index_select(1, env.actuated_dof_indices)[0])
    # Nominal gains from the config (training randomizes them per robot).
    control = env.cfg.control

    def gain(table, name):
        hits = [v for k, v in table.items() if k in name]
        if len(hits) != 1:
            raise SystemExit(f"no unique gain for {name} in {table}")
        return float(hits[0])

    p_gains = [gain(control.stiffness, n) for n in names]
    d_gains = [gain(control.damping, n) for n in names]

    actor = runner.alg.actor
    actor.eval()
    policy = FreeJointPolicy(actor, free, scale, default).to("cpu").eval()
    obs = runner.get_obs(runner.actor_cfg["obs"]).cpu()
    with torch.no_grad():
        full = actor.act_inference(obs)[0] * torch.tensor(scale) + torch.tensor(default)
        mine = policy(obs[0])
    assert torch.allclose(full[free], mine, atol=1e-6), (
        "export does not match the actor"
    )

    out = Path(args.out or Path("exports") / f"{args.task}__{args.load_run}")
    out.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.trace(policy, obs[0])
    torch.jit.save(traced, out / "policy.pt")
    files = ["policy.pt"]
    try:
        torch.onnx.export(
            traced,
            obs[0],
            out / "policy.onnx",
            input_names=["obs"],
            output_names=["free_joint_targets"],
        )
        files.append("policy.onnx")
    except Exception as e:  # ONNX export is optional
        print(f"ONNX export skipped: {e}")

    lock = env.cfg.leg_lock
    layout = []
    for part in runner.actor_cfg["obs"]:
        size = env.get_state(part).shape[1]
        s = env.scales.get(part, 1.0)
        layout.append(
            {
                "name": part,
                "size": int(size),
                "scale": tolist(s) if hasattr(s, "__len__") else float(s),
            }
        )
    meta = {
        "task": args.task,
        "checkpoint": os.path.relpath(load_path, GYM_ROOT_DIR),
        "control_dt": float(env.dt),
        "joint_order": names,
        "free_joints": {
            "names": [names[i] for i in free],
            "kp": [p_gains[i] for i in free],
            "kd": [d_gains[i] for i in free],
            "default_pos": [default[i] for i in free],
            "note": "policy output = absolute position targets [rad] in this order",
        },
        "locked_joints": {
            "names": [names[i] for i in locked],
            "hold_pos": list(lock.poses[lock.pose]),
            "kp": float(lock.stiffness),
            "kd": float(lock.damping),
            "note": "never driven by the policy; hold this pose with this PD "
            "every control step, as in simulation",
        },
        "observation": layout,
        "observation_size": int(obs.shape[1]),
        "observation_note": "each part = measured value / scale, concatenated in "
        "this order; locked joints' dof_pos_target entries are 0; "
        "phase_obs and osc_omega come from the oscillators, which "
        "must run on the robot with the measured foot loads",
    }
    (out / "deploy.json").write_text(json.dumps(meta, indent=2))
    print(
        f"Exported {out}: {', '.join(files + ['deploy.json'])} "
        f"({len(free)} free joints, locked: {meta['locked_joints']['names']})"
    )


if __name__ == "__main__":
    main()
