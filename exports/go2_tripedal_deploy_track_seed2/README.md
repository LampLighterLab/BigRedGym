# Go2 tripedal policy: `even2.5_deploy_track` seed 2

Sim-to-real candidate for the Go2 with the front-right leg locked (tucked).
Decentralized phase oscillators (Zhang et al., ICRA 2024) with an even-load
reward, stronger speed tracking, and the deploy package: pushes, wide
friction range, noisy and delayed foot loads, observation noise, mass
randomization, stop commands, starts from rest, turning while moving, and a
stand-still penalty. Forward commands are trained over 0.25-2.5 m/s.

- `policy.pt`: TorchScript. It takes the 54-dim observation exactly as in
  training and returns absolute position targets [rad] for the 9 free
  joints. The locked leg's outputs are dropped inside the model.
- `deploy.json`: joint order, PD gains, default pose, the locked leg's hold
  command (pose `[0, 1.36, -2.65]` rad, kp 40, kd 1), the observation layout
  and scales, and the control period (0.01 s). The oscillators (`phase_obs`,
  `osc_omega`) must run on the robot from the measured foot loads.

Source: `logs/go2osc_wide_even2.5_deploy_track/Nov16_16-13-14_seed2/model_2000.pt`.
The export checks that it reproduces the actor's free-joint actions (atol
1e-6). Reproduce with

    uv run --frozen scripts/export_tripedal.py --task go2osc_wide_even2.5_deploy_track \
        --load_run Nov16_16-13-14_seed2 --out exports/go2_tripedal_deploy_track_seed2

## Readiness evaluation (simulation)

Speed x frequency grid: 0.5-2.5 m/s x 1.5-6 Hz (35 cells), 64 robots per
cell from random oscillator phases, measured 10 s after 10 s of settling.
Pushes: instant base-velocity changes in 4 directions while walking at 1 and
2.5 m/s. Start/stop: stand 3 s, ramp to v at 2 m/s^2, turn, stop, stand 5 s,
restart, at v = 1 and 2.5 m/s and 3 and 4.5 Hz, 64 robots each.

| Criterion | Threshold | Result | |
|---|---|---|---|
| Least-loaded leg (load share) | >= 25% | FL 35%, RL 27%, RR 38% (worst cell 23%) | pass on average |
| Peak landing force | < 2 x body weight | mean 1.62, max 2.0 | at the limit |
| Fall rate, walking | < 1% | 0.2% | pass |
| Fall rate, 2 m/s pushes | < 5% | 3% (0% at 0.5-1.5 m/s pushes) | pass |
| Peak torque | < 80% of actuator limit | not measured yet | open |
| Cost of transport | < 2.5 | 2.17 (mean joint power 442 W) | pass |
| Forward speed tracking | < 15% error up to 2.5 m/s | 85-105% of command in every cell | pass |
| Starts from rest | moves within 2 s | 0.5 s at 1 m/s; at 2.5 m/s it levels off near 2.2 m/s | pass |
| Stops when commanded | all robots | 100%, no falls | pass |
| Joint motion while stopped | none | 0 steps; RMS joint speed 0.04 rad/s | pass |
| Stopping distance from 2.5 m/s | < 2 m | median 1.5-1.6 m, worst robot 2.4 m | median passes |
| Turning (secondary) | | 0.51 of the commanded yaw rate | weak |

Above 2.5 m/s (outside the trained range) falls rise to 3-9% at 3 m/s and
over 90% at 4.5 m/s. Still to do before hardware: peak joint torque
(including the locked leg's hold torque), the low-friction (0.3) test, and an
ONNX export if the robot side needs it.
