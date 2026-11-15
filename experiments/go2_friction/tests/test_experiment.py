"""Pure-Python smoke tests; do not need MuJoCo or a Go2."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sweep_friction import LOWER_BOUNDS, SEEDS, experiment_name, command_for
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
        self.assertIn('--disable_wandb', cmd)

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
