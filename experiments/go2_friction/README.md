# Go2 friction and observation noise experiments

Standalone experiment code for BigRedGym, kept in one directory. Run **all commands from the BigRedGym repository root** (where `pyproject.toml` and `scripts/train.py` live).

Both experiments reuse BigRedGym's training pipeline in `scripts/train.py`. Checkpoints and their `friction_experiment.json` manifest live together under `logs/<experiment_name>/<run>/`. The manifest includes the requested and applied friction ranges, the actor noise settings, and the resolved training and environment configs. Sweep manifests and evaluation outputs go in `experiments/go2_friction/results/`.

## Layout

```
BigRedGym/
├── scripts/train.py                  # existing core training code (unchanged)
├── experiments/go2_friction/
│   ├── train_friction.py
│   ├── sweep_friction.py
│   ├── sweep_noise.py
│   ├── evaluate_friction.py
│   ├── evaluate_all_friction.py
│   ├── analyze_friction.py
│   ├── tests/test_experiment.py
│   ├── results/                    # generated outputs, gitignored
│   └── README.md
└── logs/                            # BigRedGym model checkpoints
```

## Training conditions

- **Experiment A** varies the lower friction bound across `1.25, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2` with upper bound `1.25` and zero observation noise. Each simulated robot keeps its startup friction draw across episode resets. Runs use `go2friction_lb_<bound>_seed_<seed>`.
- **Experiment B** uses fixed friction `1.25` and noise scales `0, 0.5, 1, 2, 4`. Runs use `go2noise_scale_<scale>_seed_<seed>`.

The noise scale multiplies the configured Gaussian standard deviations for base angular velocity, projected gravity (using the `gravity_vec` default), joint position, and joint velocity. The shared runner starts with a clean actor observation and applies noise on subsequent training observations. The critic uses clean simulator state throughout. Scale zero supplies an empty noise map and gives clean observations at every step.

Both sweeps use seeds `42,123,456` by default. Pass a comma-separated `--seeds` list to change the number of runs per condition. `--keep-other-randomization` enables the configured mass and PD-gain randomization. Both sweeps use the same rewards, network, and push settings by default.

## Verify and smoke test

```bash
# Preview the two training matrices
uv run --frozen experiments/go2_friction/sweep_friction.py --dry-run
uv run --frozen experiments/go2_friction/sweep_noise.py --dry-run

# Run smoke tests
uv run --frozen python -m unittest discover -s experiments/go2_friction/tests -p 'test_experiment.py' -v

# Train one short PPO iteration in each smoke case
smoke() {
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen --no-sync \
    experiments/go2_friction/train_friction.py --task go2trot --device cpu \
    --seed 42 --num_envs 8 --max_iterations 1 --rollout-size 2048 \
    --batch_size 256 --headless --disable_wandb --max-friction 1.25 "$@"
}
smoke --min-friction 0.6 --noise-scale 0 \
  --experiment_name go2friction_smoke_lb_0p60_seed_42
smoke --min-friction 1.25 --noise-scale 0 \
  --experiment_name go2noise_smoke_scale_0p00_seed_42
smoke --min-friction 1.25 --noise-scale 2 \
  --experiment_name go2noise_smoke_scale_2p00_seed_42
```

## Full training commands

```bash
# Experiment A trains 21 policies with the default seeds
uv run --frozen experiments/go2_friction/sweep_friction.py \
  --max-iterations 200 --num-envs 256 --device cpu

# Experiment B trains 15 policies with the default seeds
uv run --frozen experiments/go2_friction/sweep_noise.py \
  --max-iterations 200 --num-envs 256 --device cpu
```

## Evaluate and analyze

```bash
# Check one trained policy first
uv run --frozen experiments/go2_friction/evaluate_friction.py \
  --experiment-name go2friction_lb_0p60_seed_42 \
  --frictions 0.6 --maneuvers stand_2_stop --trials 2 --device cpu

# Full evaluation (very many simulation trials)
uv run --frozen experiments/go2_friction/evaluate_all_friction.py --dry-run
uv run --frozen experiments/go2_friction/evaluate_all_friction.py --trials 10 --device cpu
uv run --frozen experiments/go2_friction/analyze_friction.py
```

The friction evaluation metrics use exploratory thresholds. Use the run manifest to confirm that a checkpoint has the intended training noise setting, especially for friction runs that predate `--noise-scale`.
