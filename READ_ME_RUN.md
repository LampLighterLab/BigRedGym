# Phase 1 — Go2 Policy Deployment (Linux)

## 1. Setup

From the repository root:

```bash
uv sync --frozen
```

## 2. Run the policy in simulation

```bash
uv run --frozen python scripts/play.py \
  --task go2osc \
  --device cpu \
  --num_envs 1 \
  --experiment_name go2osc \
  --load_run varying_freq_standing_knee_clearance_90_deg_phase_aa3ea0f \
  --checkpoint 1000 \
  --original_cfg
```

## 3. Verify the deployment configuration

```bash
uv run --frozen pytest tests/unit_tests/test_go2osc_deploy.py -q
```

```bash
uv run --frozen python -c "from go2_deploy.deploy_config import DeployConfig; from go2_deploy.utility.import_custom_impl import load_custom_impl; c=DeployConfig(load_custom_impl('go2osc')); print(c.task_name, c.obs_dim())"
```

Expected output: `go2osc 61`. The matching checkpoint is
`logs/go2osc/varying_freq_standing_knee_clearance_90_deg_phase_aa3ea0f/model_1000.pt`.

## 4. Export the policy

```bash
uv run --frozen python \
  scripts/export_go2_policy.py \
  --task go2osc \
  --name go2osc_with_force \
  --run_name varying_freq_standing_knee_clearance_90_deg_phase_aa3ea0f
```

The exported policy is saved in `logs/policies/go2osc_with_force/`. Before
exporting for the robot, set the shared deployment control rate, gains, and
action smoothing to match training: 50 Hz, kp=20, kd=0.5, and no smoothing.

## 5. Deploy on the real robot

Follow `README_DEPLOY.md` for Unitree SDK installation
and Ethernet configuration.

`Go2OscCustom.force_calibration` has an uncalibrated Go2 EDU baseline: zero
offsets, unit gains, identity foot order, and 15 kg mass. Only the approximate
mass comes from the robot specification. Check the `LowState.foot_force` order,
unloaded offsets, and conversion to newtons on this robot before deployment.
The policy expects each foot's upward force divided by body weight.
Also keep movement commands within the training range (forward 0.3–0.7 m/s,
lateral 0, yaw 0) and verify joint limits and the emergency-stop procedure
with course staff. The current deployment controller does not provide a
frequency keybind, so its oscillator remains at 2 Hz unless a control is added.

Once approved, run:

```bash
uv run --frozen --extra unitree_sdk \
  python -m go2_deploy.deploy eth0 \
  --bundle logs/policies/go2osc_with_force
```

Replace `eth0` with the appropriate network interface.

## Policy configuration

- Observation dimension: 61
- Action dimension: 12
- Control frequency: 50 Hz
- Initial gait frequency: 2 Hz
- Phase offsets: 0°, 90°, 180°, 270°
- Checkpoint: 1000 from `varying_freq_standing_knee_clearance_90_deg_phase_aa3ea0f`

Physical deployment remains subject to hardware calibration
and supervised testing.
