"""Left-right mirror operators for observation and action vectors.

The mirror is the reflection y -> -y in the base frame (x forward, y left,
z up). For a vector built by concatenating named states (as
``env.get_states(obs_list)`` does), the mirror is a signed permutation:

    mirrored = x[..., perm] * sign

Each named state transforms according to its *kind*:

- ``"joint"``: one entry per joint (or a whole multiple of the joint count,
  e.g. a stacked history). Joints swap with their opposite-side partner, and
  joints whose axis is the x-axis (hip ab/adduction) change sign.
- ``"polar"``: a base-frame 3-vector such as linear velocity or projected
  gravity: (x, y, z) -> (x, -y, z).
- ``"axial"``: a base-frame angular 3-vector: (x, y, z) -> (-x, y, -z).
- ``"invariant"``: unchanged (e.g. base height).
- a list of +-1: explicit per-element signs with no permutation
  (e.g. commands [vx, vy, yaw_rate] -> [1, -1, -1]).

Because every scale in ``env.scales`` is checked to be mirror-invariant, the
mirror commutes with BigRedGym's observation scaling.
"""

import torch

GO2_STATE_KINDS = {
    "base_height": "invariant",
    "base_lin_vel": "polar",
    "projected_gravity": "polar",
    "base_ang_vel": "axial",
    "commands": [1.0, -1.0, -1.0],
    "dof_pos": "joint",
    "dof_pos_obs": "joint",
    "dof_vel": "joint",
    "dof_pos_target": "joint",
    "dof_pos_history": "joint",
    "torques": "joint",
}

GO2_SIDE_PAIRS = [("FL_", "FR_"), ("RL_", "RR_")]
GO2_SIGN_FLIP_KEYWORDS = ["hip_joint"]

_POLAR = [1.0, -1.0, 1.0]
_AXIAL = [-1.0, 1.0, -1.0]


def _swap_side(name, side_pairs):
    for left, right in side_pairs:
        if left in name:
            return name.replace(left, right)
        if right in name:
            return name.replace(right, left)
    return name


def joint_mirror(joint_names, side_pairs, sign_flip_keywords):
    """Return (perm, sign) lists mirroring a per-joint vector."""
    index = {name: i for i, name in enumerate(joint_names)}
    perm, sign = [], []
    for name in joint_names:
        partner = _swap_side(name, side_pairs)
        if partner not in index:
            raise ValueError(f"Joint {name!r} has no mirror partner {partner!r}")
        perm.append(index[partner])
        flip = any(key in name for key in sign_flip_keywords)
        sign.append(-1.0 if flip else 1.0)
    return perm, sign


class Mirror:
    """Signed permutation acting on the last dimension of a tensor."""

    def __init__(self, perm, sign, device="cpu"):
        perm = torch.as_tensor(perm, dtype=torch.long).cpu()
        sign = torch.as_tensor(sign, dtype=torch.float32).cpu()
        # * Validate on CPU, then move: the checks must not depend on device.
        if perm.shape != sign.shape:
            raise ValueError("perm and sign must have the same length")
        if not torch.equal(torch.sort(perm).values, torch.arange(len(perm))):
            raise ValueError("perm is not a permutation")
        self.perm = perm.to(device)
        self.sign = sign.to(device)

    def __call__(self, x):
        return x[..., self.perm] * self.sign.to(x.dtype)

    def __len__(self):
        return len(self.perm)

    def to(self, device):
        self.perm = self.perm.to(device)
        self.sign = self.sign.to(device)
        return self

    def matrix(self):
        """Dense matrix M with mirrored = x @ M.T (for inspection/tests)."""
        n = len(self.perm)
        M = torch.zeros(n, n)
        M[torch.arange(n), self.perm.cpu()] = self.sign.cpu()
        return M


def _state_mirror(kind, dim, joint_perm, joint_sign, name):
    if kind == "invariant":
        return list(range(dim)), [1.0] * dim
    if kind in ("polar", "axial"):
        if dim != 3:
            raise ValueError(f"{name}: {kind} state must have dim 3, got {dim}")
        return [0, 1, 2], list(_POLAR if kind == "polar" else _AXIAL)
    if kind == "joint":
        n = len(joint_perm)
        if dim % n != 0:
            raise ValueError(f"{name}: dim {dim} is not a multiple of {n} joints")
        perm, sign = [], []
        for block in range(dim // n):
            perm += [block * n + p for p in joint_perm]
            sign += list(joint_sign)
        return perm, sign
    if isinstance(kind, (list, tuple)):
        if len(kind) != dim:
            raise ValueError(f"{name}: {len(kind)} signs for dim {dim}")
        return list(range(dim)), [float(s) for s in kind]
    raise ValueError(f"{name}: unknown mirror kind {kind!r}")


def _check_scale_invariant(env, name, perm, sign):
    scale = getattr(env, "scales", {}).get(name)
    if scale is None:
        return
    scale = torch.as_tensor(scale, dtype=torch.float32).detach().cpu().flatten()
    if scale.numel() == 1:
        return
    if scale.numel() != len(perm):
        return  # broadcast scale; nothing to permute
    perm_t = torch.as_tensor(perm)
    if not torch.allclose(scale[perm_t], scale):
        raise ValueError(f"Scale of {name!r} is not mirror-invariant: {scale}")


def build_mirror(
    env,
    state_list,
    state_kinds=None,
    side_pairs=None,
    sign_flip_keywords=None,
    joint_names=None,
):
    """Build the Mirror for ``env.get_states(state_list)``."""
    state_kinds = GO2_STATE_KINDS if state_kinds is None else state_kinds
    side_pairs = GO2_SIDE_PAIRS if side_pairs is None else side_pairs
    if sign_flip_keywords is None:
        sign_flip_keywords = GO2_SIGN_FLIP_KEYWORDS
    if joint_names is None:
        joint_names = list(env.dof_names)
        actuated = list(getattr(env, "actuated_dof_names", joint_names))
        if actuated != joint_names:
            raise ValueError("Mirror assumes every DOF is actuated, in DOF order")
    joint_perm, joint_sign = joint_mirror(joint_names, side_pairs, sign_flip_keywords)

    perm, sign, offset = [], [], 0
    for name in state_list:
        if name not in state_kinds:
            raise KeyError(f"No mirror rule for state {name!r}; add it to state_kinds")
        dim = env.get_states([name]).shape[-1]
        p, s = _state_mirror(state_kinds[name], dim, joint_perm, joint_sign, name)
        _check_scale_invariant(env, name, p, s)
        perm += [offset + i for i in p]
        sign += s
        offset += dim
    return Mirror(perm, sign, device=getattr(env, "device", "cpu"))
