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
  --no-keyboard
```

## 3. Verify the deployment configuration

```bash
uv run --frozen pytest tests/unit_tests/test_go2osc_deploy.py -q
```

```bash
uv run --frozen python -c "from go2_deploy.deploy_config import DeployConfig; from go2_deploy.utility.import_custom_impl import load_custom_impl; c=DeployConfig(load_custom_impl('go2osc')); print(c.task_name, c.obs_dim())"
```

Expected output: `go2osc 57`. The existing `model_1000.pt` was trained with
61 inputs and cannot be used with this configuration; train a new policy first.

## 4. Export the policy

```bash
uv run --frozen python \
  scripts/export_go2_policy.py \
  --task go2osc \
  --name go2osc_no_force \
  --run_name YOUR_NEW_RUN_NAME
```

The exported policy is saved in `logs/policies/go2osc_no_force/`.

## 5. Deploy on the real robot

Follow `README_DEPLOY.md` for Unitree SDK installation
and Ethernet configuration.

Before deployment, match the deployment control rate and gains to training,
and verify joint limits and the emergency-stop procedure with course staff.

Once approved, run:

```bash
uv run --frozen --extra unitree_sdk \
  python -m go2_deploy.deploy eth0 \
  --bundle logs/policies/go2osc_no_force
```

Replace `eth0` with the appropriate network interface.

## Policy configuration

- Observation dimension: 57
- Action dimension: 12
- Control frequency: 50 Hz
- Initial gait frequency: 2 Hz
- Phase offsets: 0°, 90°, 180°, 270°
- Checkpoint: from a new 57-input training run

Physical deployment remains subject to hardware calibration
and supervised testing.
