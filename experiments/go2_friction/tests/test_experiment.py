"""Check sweep commands, actor noise, and evaluation metrics on CPU tensors."""
from contextlib import redirect_stdout
from io import StringIO
import shlex
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from gym.envs.base.task_skeleton import TaskSkeleton
from gym.envs.go2.go2trot_config import Go2TrotCfg, Go2TrotRunnerCfg
from gym.utils.helpers import class_to_dict
from learning.runners.BaseRunner import BaseRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sweep_friction import LOWER_BOUNDS, SEEDS, experiment_name, command_for
from sweep_noise import NOISE_SCALES, main as sweep_noise
from train_friction import configure_observation_noise
from evaluate_friction import aggregate, critical_friction, MANEUVERS, parse_numbers
from analyze_friction import compute_critical


class FrictionProjectTests(unittest.TestCase):
    def test_twenty_one_runs_and_paired_seeds(self):
        self.assertEqual(len(LOWER_BOUNDS) * len(SEEDS), 21)
        names = [experiment_name(mu, seed) for mu in LOWER_BOUNDS for seed in SEEDS]
        self.assertEqual(len(set(names)), 21)
        self.assertEqual(experiment_name(0.6, 42), 'go2friction_lb_0p60_seed_42')

    def test_command_uses_correct_friction_and_seed(self):
        args = SimpleNamespace(upper=1.25, max_iterations=200, num_envs=256,
                               backend='mujoco', device='cpu', wandb=False,
                               keep_other_randomization=False)
        cmd = command_for(Path('/BigRedGym'), 0.6, 42, args)
        self.assertEqual(cmd[cmd.index('--min-friction')+1], '0.6')
        self.assertEqual(cmd[cmd.index('--seed')+1], '42')
        self.assertEqual(cmd[cmd.index('--max_iterations')+1], '200')
        self.assertEqual(float(cmd[cmd.index('--noise-scale') + 1]), 0.0)
        self.assertIn('--disable_wandb', cmd)

    def test_noise_sweep_has_fixed_friction_and_distinct_names(self):
        output = StringIO()
        with redirect_stdout(output):
            sweep_noise(["--dry-run", "--seeds", "42"])
        commands = [
            shlex.split(line[2:]) for line in output.getvalue().splitlines()
            if line.startswith("$ ")
        ]
        self.assertEqual(len(commands), 5)
        names = set()
        for cmd, scale in zip(commands, NOISE_SCALES, strict=True):
            self.assertEqual(float(cmd[cmd.index("--min-friction") + 1]), 1.25)
            self.assertEqual(float(cmd[cmd.index("--max-friction") + 1]), 1.25)
            self.assertEqual(float(cmd[cmd.index("--noise-scale") + 1]), scale)
            names.add(cmd[cmd.index("--experiment_name") + 1])
        self.assertEqual(len(names), 5)
        self.assertNotIn(experiment_name(1.25, 42), names)

    def test_observation_noise_scales_only_imu_and_joint_inputs(self):
        env = TaskSkeleton(num_envs=16)
        env.scales = class_to_dict(Go2TrotCfg().scaling, "cpu")
        obs = Go2TrotRunnerCfg().actor.obs
        sizes = (3, 3, 3, 12, 12, 12, 2, 1)
        for name, size in zip(obs, sizes, strict=True):
            setattr(env, name, torch.zeros(env.num_envs, size))
        runner = BaseRunner.__new__(BaseRunner)
        runner.env = env
        runner.device = "cpu"
        clean = runner.get_obs(obs)
        samples = []
        with torch.random.fork_rng(devices=[]):
            for scale in (0.0, 1.0, 2.0):
                train_cfg = Go2TrotRunnerCfg()
                default_std = configure_observation_noise(train_cfg, scale)
                torch.manual_seed(42)
                samples.append(runner.get_noisy_obs(obs, train_cfg.actor.noise))
        self.assertEqual(default_std, {
            "base_ang_vel": 0.01,
            "projected_gravity": 0.1,
            "dof_pos_obs": 0.01,
            "dof_vel": 0.005,
        })
        zero, one, two = samples
        torch.testing.assert_close(zero, clean, rtol=0, atol=0)
        torch.testing.assert_close(two - clean, 2 * (one - clean))
        deltas = (one - clean).split(sizes, dim=1)
        changed = {
            name for name, delta in zip(obs, deltas, strict=True)
            if torch.count_nonzero(delta).item()
        }
        self.assertEqual(changed, set(default_std))
        torch.testing.assert_close(runner.get_obs(obs), clean, rtol=0, atol=0)

    def test_maneuvers(self):
        self.assertEqual(len(MANEUVERS), 7)
        self.assertEqual(MANEUVERS['stand_2_stop'][1][1], 2.0)
        self.assertEqual(MANEUVERS['forward_backward'][2][1], -1.0)
        self.assertEqual(MANEUVERS['forward_sideways'][2][2], 1.0)

    def test_aggregate_and_critical_friction(self):
        rows=[]
        for mu in [0.2, 0.4]:
            for trial in range(10):
                rows.append(dict(friction=mu, maneuver='stand_2_stop',
                    success=int(mu==0.4 or trial<8),
                    tracking_rmse_mps=0.3, friction_demand_p95=0.2,
                    foot_slip_fraction=0.1, stopping_distance_m=0.4,
                    time_to_speed_s=0.5))
        grouped=aggregate(rows)
        self.assertEqual(critical_friction(grouped), 0.4)
        self.assertEqual(grouped[0]['successes'], 8)
        self.assertEqual(grouped[1]['successes'], 10)
        self.assertIsNone(critical_friction([dict(x, trials=9) for x in grouped]))

    def test_cross_seed_aggregation(self):
        rows=[{'training_lower_bound': 0.6, 'training_seed': 42,
               'maneuver': 'stand_2_stop', 'friction': '0.4',
               'trials': '10', 'successes': '9'}]
        self.assertEqual(compute_critical(rows)[0]['critical_friction'], 0.4)

    def test_friction_parsing(self):
        self.assertEqual(parse_numbers('0.2, 0.5, 1.25'), (0.2,0.5,1.25))
        with self.assertRaises(ValueError):
            parse_numbers('0.1,-0.2')


if __name__ == '__main__':
    unittest.main()
