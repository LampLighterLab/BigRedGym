"""Left-right mirror operators for the Go2 symmetry experiments."""

from types import SimpleNamespace

import pytest
import torch

from gym.envs.go2.go2_config import GO2_DOF_NAMES, Go2Cfg, Go2RunnerCfg
from learning.utils.symmetry import Mirror, build_mirror, joint_mirror
from learning.utils.symmetry import GO2_SIDE_PAIRS, GO2_SIGN_FLIP_KEYWORDS

STATE_DIMS = {
    "base_height": 1,
    "base_lin_vel": 3,
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "commands": 3,
    "dof_pos_obs": 12,
    "dof_vel": 12,
    "dof_pos_target": 12,
    "dof_pos_history": 36,
}


def _fake_env(scales=None):
    def get_states(names):
        return torch.zeros(1, sum(STATE_DIMS[n] for n in names))

    return SimpleNamespace(
        dof_names=list(GO2_DOF_NAMES),
        actuated_dof_names=list(GO2_DOF_NAMES),
        scales=scales or {},
        device="cpu",
        get_states=get_states,
    )


def test_go2_joint_mirror_swaps_sides_and_flips_hips():
    perm, sign = joint_mirror(GO2_DOF_NAMES, GO2_SIDE_PAIRS, GO2_SIGN_FLIP_KEYWORDS)
    mirrored = [GO2_DOF_NAMES[p] for p in perm]
    assert mirrored[:6] == [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
    ]
    assert mirrored[6:9] == ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"]
    assert sign == [-1.0, 1.0, 1.0] * 4


def test_vector_and_command_rules():
    env = _fake_env()
    m = build_mirror(
        env, ["base_lin_vel", "base_ang_vel", "projected_gravity", "commands"]
    )
    x = torch.arange(1.0, 13.0)
    expected = torch.tensor(
        [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, 10.0, -11.0, -12.0]
    )
    torch.testing.assert_close(m(x), expected)


@pytest.mark.parametrize(
    "states",
    [Go2RunnerCfg.actor.obs, Go2RunnerCfg.critic.obs, Go2RunnerCfg.actor.actions],
)
def test_go2_mirrors_are_orthogonal_involutions(states):
    env = _fake_env(scales=Go2Cfg.scaling.__dict__)
    m = build_mirror(env, states)
    x = torch.randn(64, len(m))
    torch.testing.assert_close(m(m(x)), x)
    M = m.matrix()
    torch.testing.assert_close(M @ M.T, torch.eye(len(m)))


def test_stacked_joint_history_mirrors_each_block():
    env = _fake_env()
    m = build_mirror(env, ["dof_pos_history"])
    single = build_mirror(env, ["dof_pos_target"])
    x = torch.randn(5, 36)
    expected = torch.cat([single(x[:, 12 * i : 12 * (i + 1)]) for i in range(3)], 1)
    torch.testing.assert_close(m(x), expected)


def test_unknown_state_and_asymmetric_scale_are_rejected():
    env = _fake_env()
    STATE_DIMS["mystery"] = 2
    try:
        with pytest.raises(KeyError):
            build_mirror(env, ["mystery"])
    finally:
        del STATE_DIMS["mystery"]
    bad = _fake_env(scales={"dof_vel": torch.arange(1.0, 13.0)})
    with pytest.raises(ValueError, match="mirror-invariant"):
        build_mirror(bad, ["dof_vel"])


def test_mirror_rejects_non_permutation():
    with pytest.raises(ValueError):
        Mirror([0, 0], [1.0, 1.0])


# * Physics: the mirror image of a state must evolve into the mirror image -----


def _mirror_root_states(root, dof_mirror_unused=None):
    """World-frame reflection y -> -y of [pos, quat(xyzw), lin_vel, ang_vel]."""
    out = root.clone()
    out[:, 1] *= -1  # y
    out[:, 3] *= -1  # quat x
    out[:, 5] *= -1  # quat z
    out[:, 8] *= -1  # vy
    out[:, 10] *= -1  # wx
    out[:, 12] *= -1  # wz
    return out


def _max_mirror_error(sign_flip_keywords):
    """Step a random state and its mirror image together on MuJoCo CPU.

    Returns the largest |obs(mirrored state) - M_o obs(state)| over 30 steps,
    in scaled observation units.
    """
    from gym.envs.go2.go2 import Go2
    from gym.utils.task_registry import select_backend, task_registry

    device = "cpu"
    cfg = Go2Cfg()
    runner_cfg = Go2RunnerCfg()
    cfg.seed = 3
    cfg.env.num_envs = 2
    cfg.push_robots.toggle = False
    cfg.init_state.reset_mode = "reset_to_basic"
    cfg.domain_randomization.startup.contact_friction_range = None
    cfg.domain_randomization.startup.link_mass_scale_range = None
    cfg.domain_randomization.episode.scale_ranges = {}
    task_registry.convert_frequencies_to_params(cfg, runner_cfg)
    backend = select_backend(cfg, device, "mujoco")
    env = Go2(cfg, device, True, backend)

    joint = build_mirror(env, ["dof_pos"], sign_flip_keywords=sign_flip_keywords)
    obs_states = [s for s in Go2RunnerCfg.critic.obs if s != "commands"]
    obs_mirror = build_mirror(env, obs_states, sign_flip_keywords=sign_flip_keywords)

    g = torch.Generator().manual_seed(0)
    root = torch.zeros(1, 13)
    root[0, :3] = torch.tensor([0.0, 0.03, 0.36])
    q = torch.tensor([0.03, -0.05, 0.2, 1.0])  # small roll/pitch, some yaw
    root[0, 3:7] = q / q.norm()
    root[0, 7:10] = torch.tensor([0.6, 0.25, -0.05])
    root[0, 10:13] = torch.tensor([0.3, -0.2, 0.7])
    dof_pos = env.default_dof_pos[:1] + 0.15 * torch.randn(1, 12, generator=g)
    dof_vel = 0.5 * torch.randn(1, 12, generator=g)
    target = 0.2 * torch.randn(1, 12, generator=g)

    env.root_states[:] = torch.cat((root, _mirror_root_states(root)))
    env.root_states[:, :3] += env.env_origins
    env.dof_pos[:] = torch.cat((dof_pos, joint(dof_pos)))
    env.dof_vel[:] = torch.cat((dof_vel, joint(dof_vel)))
    backend.reset_state(torch.ones(2, dtype=torch.bool))

    worst = 0.0
    for _ in range(30):
        env.dof_pos_target[:] = torch.cat((target, joint(target)))
        env.step()
        obs = env.get_states(obs_states)
        worst = max(worst, (obs[1] - obs_mirror(obs[0])).abs().max().item())
    return worst


def test_go2_physics_is_mirror_equivariant():
    """The mirror image of a state evolves into the mirror image.

    The Go2 URDF is symmetric up to a small base product of inertia
    (ixy = 1.2e-4, iyz = -3e-5 kg m^2, ~0.5% of ixx) and MuJoCo's contact
    solver is not exactly reflection-invariant, so a residual of ~1e-2 in
    scaled units remains, mostly in roll rate around touchdown. A wrong
    mirror rule (here: no hip sign flip) is an order of magnitude worse.
    """
    assert _max_mirror_error(GO2_SIGN_FLIP_KEYWORDS) < 2e-2
    assert _max_mirror_error([]) > 2e-1


# * Device handling: env tensors live on cuda:0 during training --------------

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def _device_env(device):
    env = _fake_env(
        scales={
            k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in vars(Go2Cfg.scaling).items()
            if not k.startswith("_")
        }
    )
    env.device = device
    return env


@pytest.mark.parametrize("device", DEVICES)
def test_build_mirror_on_device(device):
    m = build_mirror(_device_env(device), Go2RunnerCfg.actor.obs)
    assert m.perm.device == torch.device(device)
    x = torch.randn(4, len(m), device=device)
    torch.testing.assert_close(m(m(x)), x)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["none", "augment", "mirror_loss"])
def test_ppo2sym_update_runs_on_device(device, mode):
    """One full PPO2Sym update on random data, on every available device."""
    from tensordict import TensorDict

    from learning.algorithms.ppo2_sym import PPO2Sym
    from learning.modules import Actor, Critic

    env = _device_env(device)
    actor_m = build_mirror(env, Go2RunnerCfg.actor.obs)
    critic_m = build_mirror(env, Go2RunnerCfg.critic.obs)
    action_m = build_mirror(env, Go2RunnerCfg.actor.actions)
    actor = Actor(len(actor_m), len(action_m), [32, 32], normalize_obs=False)
    critic = Critic(len(critic_m), [32, 32], normalize_obs=False)
    alg = PPO2Sym(
        actor,
        critic,
        symmetry_mode=mode,
        mirror_loss_coef=1.0,
        batch_size=64,
        max_gradient_steps=3,
        schedule="adaptive",
        device=device,
    )
    alg.set_mirrors(actor_m, critic_m, action_m)

    steps, envs = 8, 16

    def rand(*shape):
        return torch.randn(steps, envs, *shape, device=device)

    critic_obs = rand(len(critic_m))
    data = TensorDict(
        {
            "actor_obs": rand(len(actor_m)),
            "critic_obs": critic_obs,
            "next_critic_obs": critic_obs,
            "actions": rand(len(action_m)),
            "rewards": rand(),
            "timed_out": torch.zeros(steps, envs, dtype=torch.bool, device=device),
            "dones": torch.zeros(steps, envs, dtype=torch.bool, device=device),
        },
        batch_size=[steps, envs],
        device=device,
    )
    alg.update(data)
    assert alg.mean_symmetry_error > 0.0
    assert (alg.mean_mirror_loss > 0.0) == (mode == "mirror_loss")


# * dof_pos_target clipping (go2sym) ------------------------------------------


def _make_go2(cfg_cls, num_envs=2):
    from gym.envs.go2.go2 import Go2
    from gym.utils.task_registry import select_backend, task_registry

    cfg = cfg_cls()
    cfg.seed = 3
    cfg.env.num_envs = num_envs
    cfg.push_robots.toggle = False
    cfg.init_state.reset_mode = "reset_to_basic"
    cfg.domain_randomization.startup.contact_friction_range = None
    cfg.domain_randomization.startup.link_mass_scale_range = None
    cfg.domain_randomization.episode.scale_ranges = {}
    task_registry.convert_frequencies_to_params(cfg, Go2RunnerCfg())
    backend = select_backend(cfg, "cpu", "mujoco")
    return Go2(cfg, "cpu", True, backend)


def test_dof_pos_target_bounds_are_mirror_symmetric_and_saturating():
    from gym.envs.go2.go2sym_config import Go2SymCfg

    env = _make_go2(Go2SymCfg)
    lower, upper = env.dof_pos_target_bounds()
    m = build_mirror(env, ["dof_pos"])
    # Mirroring maps the box onto itself: M[lower, upper] = [lower, upper].
    mirrored = torch.stack((m(lower[0]), m(upper[0])))
    expected = torch.stack((torch.minimum(*mirrored), torch.maximum(*mirrored)))
    torch.testing.assert_close(expected, torch.stack((lower[0], upper[0])))
    # From both joint limits, the clipped extremes still saturate the torque.
    idx = env.actuated_dof_indices
    default = env.default_dof_pos.index_select(1, idx)[0]
    limits = env.dof_pos_limits.index_select(0, idx)
    tau = env.actuated_torque_limits[0]
    kp = env.p_gains[0]
    assert torch.all(kp * (upper[0] + default - limits[:, 1]) >= tau - 1e-5)
    assert torch.all(kp * (lower[0] + default - limits[:, 0]) <= -tau + 1e-5)


def test_go2sym_clips_huge_targets_but_go2_does_not():
    from gym.envs.go2.go2sym_config import Go2SymCfg

    for cfg_cls, clipped in ((Go2SymCfg, True), (Go2Cfg, False)):
        env = _make_go2(cfg_cls)
        env.dof_pos_target[:] = 300.0
        env.step()
        obs = env.get_states(["dof_pos_target"])
        if clipped:
            _, upper = env.dof_pos_target_bounds()
            torch.testing.assert_close(env.dof_pos_target, upper)
            assert obs.abs().max() < 20
        else:
            assert obs.abs().max() > 100


def test_ppo2sym_skips_non_finite_updates():
    from tensordict import TensorDict

    from learning.algorithms.ppo2_sym import PPO2Sym
    from learning.modules import Actor, Critic

    env = _device_env("cpu")
    am = build_mirror(env, Go2RunnerCfg.actor.obs)
    cm = build_mirror(env, Go2RunnerCfg.critic.obs)
    um = build_mirror(env, Go2RunnerCfg.actor.actions)
    actor = Actor(len(am), len(um), [16], normalize_obs=False)
    critic = Critic(len(cm), [16], normalize_obs=False)
    alg = PPO2Sym(
        actor,
        critic,
        symmetry_mode="mirror_loss",
        mirror_loss_coef=1.0,
        batch_size=32,
        max_gradient_steps=2,
        device="cpu",
    )
    alg.set_mirrors(am, cm, um)
    before = [p.detach().clone() for p in actor.parameters()]
    steps, envs = 4, 8
    critic_obs = torch.randn(steps, envs, len(cm))
    rewards = torch.randn(steps, envs)
    rewards[0, 0] = float("inf")
    data = TensorDict(
        {
            "actor_obs": torch.randn(steps, envs, len(am)),
            "critic_obs": critic_obs,
            "next_critic_obs": critic_obs,
            "actions": torch.randn(steps, envs, len(um)),
            "rewards": rewards,
            "timed_out": torch.zeros(steps, envs, dtype=torch.bool),
            "dones": torch.zeros(steps, envs, dtype=torch.bool),
        },
        batch_size=[steps, envs],
    )
    alg.update(data)
    assert alg.skipped_updates > 0
    for p, q in zip(actor.parameters(), before):
        assert torch.isfinite(p).all()


# * Action latency (go2sym) -----------------------------------------------------


def _latency_env(steps):
    from gym.envs.go2.go2sym_config import Go2SymCfg

    class Cfg(Go2SymCfg):
        class control(Go2SymCfg.control):
            action_latency_steps_range = [steps, steps]

    return _make_go2(Cfg)


def test_latency_applies_old_targets_but_observes_commanded():
    env = _latency_env(2)
    env._reset_idx(torch.ones(env.num_envs, dtype=torch.bool))
    targets = [torch.full_like(env.dof_pos_target, 0.1 * (i + 1)) for i in range(4)]
    applied = []
    for x in targets:
        env.dof_pos_target[:] = x
        env.step()
        applied.append(env._applied_target.clone())
        # The observation is the commanded target, as on the robot.
        torch.testing.assert_close(env.dof_pos_target, x)
    torch.testing.assert_close(applied[0], torch.zeros_like(targets[0]))
    torch.testing.assert_close(applied[1], torch.zeros_like(targets[0]))
    torch.testing.assert_close(applied[2], targets[0])
    torch.testing.assert_close(applied[3], targets[1])


def test_latency_changes_torque_and_zero_latency_does_not():
    env0, env1 = _latency_env(0), _latency_env(1)
    for env in (env0, env1):
        env.dof_pos_target[:] = 0.3
        env.step()
    torch.testing.assert_close(env0._applied_target, env0.dof_pos_target)
    assert not torch.allclose(env0.torques, env1.torques)


def test_latency_reset_clears_history_for_reset_envs_only():
    env = _latency_env(1)
    env.dof_pos_target[:] = 0.2
    env.step()
    mask = torch.tensor([True, False])
    env._reset_idx(mask)
    assert torch.all(env._target_history[:, 0] == 0.0)
    assert torch.any(env._target_history[:, 1] != 0.0)
