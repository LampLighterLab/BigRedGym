# Phase 1 — Go2 Policy Deployment (Linux)

## 1. Setup

From the repository root:

```bash
git switch dev-phase1
git pull --ff-only origin dev-phase1
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
GO2_PHASE=phase1 uv run --frozen python -c "from go2_deploy.deploy_config import DeployConfig; c=DeployConfig(); print(c.task_name, c.obs_dim(), c.ctrl_freq)"
```

Expected output: `go2osc 61 50`

## 4. Export the policy

```bash
GO2_PHASE=phase1 uv run --frozen python \
  scripts/export_go2_policy.py \
  --name dev_phase1 \
  --experiment_name go2osc \
  --run_name varying_freq_standing_knee_clearance_90_deg_phase_aa3ea0f \
  --checkpoint 1000 \
  --current_cfg
```

The exported policy is saved in `logs/policies/dev_phase1/`.

## 5. Deploy on the real robot

Follow `README_DEPLOY.md` for Unitree SDK installation
and Ethernet configuration.

Before deployment, verify the force sensor calibration,
joint limits and emergency-stop procedure with course staff.

Once approved, run:

```bash
GO2_PHASE=phase1 uv run --frozen --extra unitree_sdk \
  python -m go2_deploy.deploy eth0 \
  --bundle logs/policies/dev_phase1
```

Replace `eth0` with the appropriate network interface.

## Policy configuration

- Observation dimension: 61
- Action dimension: 12
- Control frequency: 50 Hz
- Initial gait frequency: 2 Hz
- Phase offsets: 0°, 90°, 180°, 270°
- Checkpoint: 1000
- Branch: dev-phase1

Physical deployment remains subject to hardware calibration
and supervised testing.