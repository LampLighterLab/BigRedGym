---
name: q2-train-and-evaluate
description: Run, resume, inspect, play back, benchmark, and evaluate Q2 policies with MuJoCo CPU, MuJoCo Warp, or optional VSim. Use for training commands, CLI overrides, W&B, log and checkpoint discovery, deterministic inference, cross-backend transfer, fidelity probes, hardware-oriented scorecards, campaign evidence, or diagnosing whether two runs are comparable.
---

# Q2 Train and Evaluate

Read `genAI_skills/AGENTS.md`, `scripts/train.py --help`, the selected task config,
and the relevant evaluation script. Use current CLI help rather than a copied
flag list; this workflow changes during active tuning.

## Prepare a controlled run

Record before execution:

- commit and dirty-worktree state;
- task, backend, device, seed, environment count, control/sim frequency;
- `algorithm.rollout_size`, optimizer `batch_size`, gradient steps, and iterations;
- config/CLI overrides and whether W&B is enabled;
- the predicted outcome or acceptance threshold.

Do not compare results with unequal temporal rollout horizons unless rollout
horizon is the independent variable.

## Train

Start with a CPU smoke run:

```bash
uv run --frozen scripts/train.py --task TASK --backend mujoco --device cpu --num_envs 8 --max_iterations 2 --headless --disable_wandb
```

Use MuJoCo Warp only after CPU/unit correctness:

```bash
uv run --frozen scripts/train.py --task TASK --backend mujoco --device cuda:0 --headless
```

The first Warp run may compile kernels. Treat constraint-capacity warnings as
physics failures, not log noise.

Run VSim only through its process environment:

```bash
uv run --frozen --extra vsim --env-file .env.vsim scripts/train.py --task TASK --backend vsim --device cuda:0 --headless
```

Use `--disable_wandb` for local discriminators. Otherwise configure credentials
through local W&B authentication. Set project/entity in `user/wandb_config.json`
using `user/wandb_config_default.json`, or pass `--wandb_project` and
`--wandb_entity`. Do not commit credentials. Ordinary W&B logging remains
supported; orphaned sweep configurations were retired.

## Resume and locate artifacts

- Runs land below `logs/<experiment_name>/<timestamp>_<run_name>/` with
  `model_<iteration>.pt`, source snapshots, and metrics.
- Resolve a checkpoint through the same helpers the script uses; do not assume
  lexicographic order equals the highest iteration.
- Resume with explicit experiment/run/checkpoint inputs when reproducibility
  matters. Confirm whether `--max_iterations` means a total or additional
  budget in the current runner/script before scheduling a long run.
- Loading for inference should restore model and normalization state, switch
  to evaluation mode, and avoid optimizer loading unless resuming training.

## Play and evaluate

- Use `scripts/play.py` for interactive playback. MuJoCo Warp is
  headless; use MuJoCo CPU for its passive viewer. VSim has its own viewer and
  keyboard integration.
- Use `scripts/eval_policy.py` for deterministic, artifact-producing policy
  evaluation. Keep commands, reset distribution, seed, episode duration, and
  environment count fixed across transfer cells.
- Use `scripts/eval_go2_policy.py` for labeled Go2 policy/checkpoint comparisons
  and `notebooks/go2_policy_evaluation.py` for the report.
- Use `tools/benchmark_simulation.py`, `tools/profile_simulation.py`, and
  `tools/compare_policy_observations.py` as documented in `tools/README.md`.
- For a physics question, begin with native contract/physics tests and a focused
  policy-free probe with a predicted invariant. Historical campaign and manual
  fidelity programs are retired; recover them from git only for a specific need.

## Interpret results

- Startup reward means may be NaN until episodes complete; distinguish that
  from tensor/optimizer NaNs.
- Report per-term reward, survival/episode duration, observation/action
  statistics, KL/losses, throughput, and task-specific physical metrics.
- Compare CPU and Warp tightly because they share a MuJoCo model; allow only
  predeclared coarse bounds for cross-formulation VSim contact behavior.
- Treat checkpoint selection as part of the experiment. Do not pick a target
  backend checkpoint after viewing its transfer score unless validation-based
  selection was declared in advance.
- Do not claim correctness from a gait video, aggregate reward, or throughput
  alone. Link claims to tests, saved artifacts, and predicted acceptance
  criteria. Update `genAI_skills/DEVELOPMENT_NOTES.md` when consequential
  evidence or supported limitations change.
