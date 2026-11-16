import torch

from gym.envs.go2.go2_se import Go2SE
from gym.envs.go2.go2_se_config import Go2SECfg
from learning.algorithms import StateEstimator
from learning.modules import StateEstimatorNN


def test_foot_states_from_body_positions_and_contact_forces():
    task = Go2SE.__new__(Go2SE)
    task.cfg = Go2SECfg()
    task.num_envs = 1
    task.num_feet = 2
    task.feet_indices = torch.tensor([1, 2])
    task._rigid_body_pos = torch.tensor(
        [[[0.0, 0.0, 0.3], [0.2, 0.1, 0.022], [-0.2, -0.1, 0.122]]]
    )
    task.root_states = torch.zeros(1, 13)
    task.root_states[0, 2] = 0.3
    task.base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    task.contact_forces = torch.zeros(1, 3, 3)
    task.contact_forces[0, 1, 2] = 50.0
    task.foot_height = torch.zeros(1, 2)
    task.foot_contact = torch.zeros(1, 2)
    task.foot_pos_base = torch.zeros(1, 6)

    task._update_foot_states()

    torch.testing.assert_close(task.foot_height, torch.tensor([[0.0, 0.1]]))
    torch.testing.assert_close(task.foot_contact, torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(
        task.foot_pos_base, torch.tensor([[0.2, 0.1, -0.278, -0.2, -0.1, -0.178]])
    )


def test_estimator_outputs_probabilities_only_for_bce_targets():
    torch.manual_seed(0)
    bce_mask = torch.tensor([False, False, True, True])
    se = StateEstimator(StateEstimatorNN(3, 4, hidden_dims=[8]), bce_mask=bce_mask)
    estimate = se.predict(10.0 * torch.randn(64, 3))
    assert ((estimate[:, 2:] > 0.0) & (estimate[:, 2:] < 1.0)).all()
    assert (estimate[:, :2] < 0.0).any()


def test_estimator_update_reduces_loss_on_fixed_data():
    from tensordict import TensorDict

    torch.manual_seed(0)
    obs = torch.randn(8, 16, 3)
    targets = torch.cat((obs[..., :1] * 2.0, (obs[..., 1:2] > 0).float()), dim=-1)
    data = TensorDict({"se_obs": obs, "se_targets": targets}, batch_size=(8, 16))
    se = StateEstimator(
        StateEstimatorNN(3, 2, hidden_dims=[32]),
        learning_rate=1e-2,
        batch_size=32,
        max_gradient_steps=200,
        bce_mask=torch.tensor([False, True]),
    )
    first = sum(se.loss(obs.flatten(0, 1), targets.flatten(0, 1))).item()
    se.update_from_data(data)
    last = sum(se.loss(obs.flatten(0, 1), targets.flatten(0, 1))).item()
    assert last < 0.5 * first


def test_command_curriculum_follows_paper_schedule():
    task = Go2SE.__new__(Go2SE)
    task.cfg = Go2SECfg()
    task.command_ranges = {"lin_vel_x": [0.0, 0.0]}
    task.update_command_curriculum(0)
    lo, hi = task.command_ranges["lin_vel_x"]
    assert 1.0 < hi < 1.4 and lo == -0.5 * hi
    task.update_command_curriculum(600)
    assert abs(task.command_ranges["lin_vel_x"][1] - 2.25) < 1e-6
    task.update_command_curriculum(10_000)
    torch.testing.assert_close(
        torch.tensor(task.command_ranges["lin_vel_x"]), torch.tensor([-1.75, 3.5])
    )


def test_air_time_reward_branches():
    task = Go2SE.__new__(Go2SE)
    task.commands = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    task.feet_stance_time = torch.tensor([[0.5, 0.1], [0.1, 0.0]])
    task.feet_air_time = torch.tensor([[0.0, 0.0], [0.0, 0.3]])
    reward = task._reward_air_time()
    # * standing cmd: clip(T_s - T_a) -> 0.3 + 0.1; moving: 0.1 + 0 (T_max>=0.25)
    torch.testing.assert_close(reward, torch.tensor([0.4, 0.1]))


def test_multiplicative_reward_combination():
    from types import SimpleNamespace

    from tensordict import TensorDict

    from learning.runners import EstimatorRunner

    runner = EstimatorRunner.__new__(EstimatorRunner)
    runner.device = "cpu"
    runner.env = SimpleNamespace(dt=0.01)
    runner.critic_cfg = {
        "reward": {
            "neg_scale": 0.2,
            "weights": {"lin_vel_exp": 3.0},
            "negative_weights": {"joint_pos": -1.0},
            "termination_weight": {"termination": 10.0},
        }
    }
    # * per-step values already multiplied by dt, as in update_rewards_dict
    rd = TensorDict(
        {
            "lin_vel_exp": torch.tensor([[0.03, 0.0]]),
            "joint_pos": torch.tensor([[-0.05, 0.0]]),
            "termination": torch.tensor([[0.0, -0.1]]),
        },
        batch_size=(1, 2),
    )
    total = runner.compute_total_rewards(rd)
    expected = torch.tensor([0.03 * torch.exp(torch.tensor(0.2 * -5.0)).item(), -0.1])
    torch.testing.assert_close(total, expected)
