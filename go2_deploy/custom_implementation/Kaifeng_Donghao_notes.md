# go2sym deployment notes (Kaifeng Jia kj366, Donghao Hong dh787)

Task: `go2sym` (`gym/envs/go2/go2sym_config.py`). Policies: flat-ground velocity
tracking; the project compares PPO with left-right-symmetric training.

## Checkpoints

| Policy | Path |
|---|---|
| Mirror-loss policy (primary) | `deploy_checkpoints/go2sym/mirror_loss_s2_v3/model_3000.pt` |
| PPO baseline | `deploy_checkpoints/go2sym/none_s0_v3/model_3000.pt` |

`setup_actor()` loads from `logs/<task_name>/<run>/`, so copy a run folder to
`logs/go2sym/<run>/`. The actor takes 45 observations and outputs 12 actions.

## Differences from the go2trot deployment defaults

Only the observation vector and scaling can be set in `Kaifeng_Donghao.py`. Everything
below must be changed in `DeployConfig` / the action path:

| Setting | go2trot deploy | go2sym (trained with) |
|---|---|---|
| `task_name` | `go2trot` | `go2sym` |
| `kp`, `kd` | 30, 2.0 | **20, 0.5** |
| `exp_moving_avg` | True (alpha about 0.1) | **False** |
| `default_dof_pos` (hip, thigh, calf, all legs) | 0.0, 0.96, -1.36 | **0.0, 0.66, -1.36** (`Go2Cfg.init_state`) |
| action scale (`dof_pos_target`) | 0.2, 0.3, 0.3 | 0.5236, 1.2654, 0.4712 (set via `DeployScaling` in `Kaifeng_Donghao.py`) |
| target clipping | none | **clip each action offset (rad) to the range below before sending it and before using it as the `dof_pos_target` observation** |

Clip range of the action offset in radians (offset from `default_dof_pos`),
from `Go2.dof_pos_target_bounds()` with Kp = 20:

| Joint | Front legs | Rear legs |
|---|---|---|
| hip | [-2.232, 2.232] | [-2.232, 2.232] |
| thigh | [-3.416, 4.016] | [-2.369, 5.063] |
| calf | [-3.634, 2.794] | [-3.634, 2.794] |

Commands are the standard `[vx, vy, yaw_rate]`; the deploy command limits
(2.0, 1.0, 3.0) are within the training ranges.

In our simulated stress test (`scripts/evaluate_sim2real.py`), every policy
failed with the go2trot gains (Kp 30, Kd 2) or with the action smoothing, and
both selected policies passed with the settings above plus one control step of
action delay and pushes.
