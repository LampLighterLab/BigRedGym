"""Turn a walking go2_se_paper checkpoint into a go2_se_jump starting point.

The jump task adds a 2-dim observation (jump_command) to the actor (after
`commands`, before the estimate the runner appends) and to the end of the
critic's observations. This pads the first layer of both networks, and the
matching Adam moments, with zero columns, so the warm-started policy behaves
exactly like the walking policy until it learns to use the new input.

Usage:
    python scripts/warm_start_jump.py \
        --src logs/go2_se_paper/<run>/model_<it>.pt --dst_experiment go2_se_jump
    python scripts/train.py --task go2_se_jump --resume --load_run warm_start ...
"""

import argparse
import os

import torch


def pad_columns(t, at, n):
    zeros = torch.zeros(t.shape[0], n, dtype=t.dtype, device=t.device)
    return torch.cat((t[:, :at], zeros, t[:, at:]), dim=1)


def pad_net(ckpt, net, opt, at, n):
    """Pad <net>_state_dict['NN.0.weight'] and its Adam moments in <opt>."""
    w = ckpt[f"{net}_state_dict"]["NN.0.weight"]
    old = tuple(w.shape)
    at = old[1] if at is None else at
    ckpt[f"{net}_state_dict"]["NN.0.weight"] = pad_columns(w, at, n)
    for state in ckpt[opt]["state"].values():
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state and tuple(state[key].shape) == old:
                state[key] = pad_columns(state[key], at, n)
    return old, at


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst_experiment", default="go2_se_jump")
    p.add_argument("--run_name", default="warm_start")
    p.add_argument("--n_new", type=int, default=2, help="size of jump_command")
    args = p.parse_args()

    ckpt = torch.load(args.src, weights_only=True, map_location="cpu")
    # * the estimate (appended last to the actor obs) has as many entries
    # * as the estimator has outputs
    est = ckpt["estimator_state_dict"]
    last = [k for k in est if k.endswith("weight") and est[k].ndim == 2][-1]
    n_estimate = est[last].shape[0]
    n_actor_in = ckpt["actor_state_dict"]["NN.0.weight"].shape[1]

    a = pad_net(ckpt, "actor", "optimizer_state_dict", n_actor_in - n_estimate, args.n_new)
    c = pad_net(ckpt, "critic", "critic_optimizer_state_dict", None, args.n_new)
    print(f"actor  {a[0]} -> +{args.n_new} columns at {a[1]}")
    print(f"critic {c[0]} -> +{args.n_new} columns at {c[1]}")

    out_dir = os.path.join("logs", args.dst_experiment, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"model_{ckpt['iter']}.pt")
    torch.save(ckpt, out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
