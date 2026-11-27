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
│   ├── evaluation_contacts.py
│   ├── sweep_evaluation.py
│   ├── metrics.py
│   ├── analyze_results.py
│   ├── plot_results.py
│   ├── evaluate_all_friction.py
│   ├── analyze_friction.py
│   ├── tests/test_experiment.py
│   ├── tests/test_evaluation.py
│   ├── tests/test_metrics.py
│   ├── tests/verify_live_evaluation.py
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

## Simulation evaluation

The default maneuver is `stand_2_stop` with 1 s standing, 3 s at 2.0 m/s forward, and 3 s stopping. The steady maneuvers `steady_0p5`, `steady_1p0`, `steady_1p5`, and `steady_2p0` each last 4 s. `forward_backward` and `forward_sideways` start with 1 s standing, then command 3 s forward and 3 s in the new direction at 1.0 m/s. Select a comma-separated list with `--maneuvers`, or use `--maneuvers all`.

The evaluator freezes the checkpoint policy, pins friction through each reset, and disables pushes and mass and gain randomization. It fixes gait frequency at 2 Hz by default and seeds the starting phase and Gaussian noise per trial. The same trial seeds pair the phase and noise draws across conditions. Test noise uses the four sensor channels and default amplitudes from Step 1. It starts at the first inference observation.

`--device` selects the actor device. The physics and contact logging run in MuJoCo CPU. `--num-envs` caps concurrent worlds. Use `--eval-seed` with `--trials` for consecutive seeds, or pass an explicit list with `--eval-seeds`. Use a new output directory for each dataset.

```bash
uv run --frozen --no-sync experiments/go2_friction/evaluate_friction.py \
  --checkpoint-path logs/go2friction_lb_0p40_seed_42/Nov23_20-35-00_/model_2.pt \
  --frictions 0.4,1.25 --noise-scales 0,1 --eval-seeds 42,73 --num-envs 2 \
  --device cpu --output-dir "$(mktemp -d)/evaluation"

# Preview the grid for the latest checkpoints from completed Step 1 runs
uv run --frozen --no-sync experiments/go2_friction/sweep_evaluation.py --dry-run
```

You can also select a checkpoint using `--experiment-name`, `--load-run`, and `--checkpoint`. Pass `--original-cfg` to load a run's saved configuration classes. The sweep accepts repeated `--checkpoint-path` flags to pin an explicit set of policies. Its default test grid is friction `0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.25` and noise scales `0,1`.

### Override a maneuver

Pass `--maneuver-config FILE.json` to either script. Each stage has the form `[seconds, body vx in m/s, body vy in m/s, yaw rate in rad/s]`. Durations are rounded to the nearest control step. Unlisted maneuvers use their defaults.

```json
{
  "stand_2_stop": [[0.5, 0, 0, 0], [2.0, 1.8, 0, 0], [1.5, 0, 0, 0]],
  "forward_sideways": [[2.0, 1.0, 0, 0], [2.0, 0, 0.5, 0]]
}
```

### Dataset format

- `metadata.json` records checkpoint identity and SHA-256, training provenance when available, resolved configs, evaluation settings, frame conventions, and stage schedules.
- `trials.csv` links each trial's friction, noise, seed, and outcome to its trace. `completed` means the trial reached its commanded horizon. A `fall` is Go2Trot's base-contact termination. The terminal sample remains in the trace for falls, timeouts, and numerical failures. An instability warning from MuJoCo marks a numerical failure even if the solver recovers to a finite state. Its timestamp marks the attempted control interval.
- Each `traces/trial_*.npz` contains numeric arrays with time along the first axis. Elapsed time excludes the reset warm-up step and starts at one control interval. `command` and `stage_index` describe the just-executed interval. Base and foot kinematics are sampled at the interval end. Contact states and forces come from the solver's last physics substep in that interval.

Foot order is listed in `metadata.json`. `foot_ground_force_world_N` sums the last physics substep's active ground-plane contacts on each foot. MuJoCo's contact-frame forces are rotated into world XYZ with positive Z up and the sign chosen for force on the foot. `foot_contact` marks active solver contacts. `foot_velocity_world_m_s` is the world-frame velocity of each foot body's origin. These traces are the input for offline metrics.

`evaluate_all_friction.py` delegates to the current evaluator. Use
`metrics.py` and `analyze_results.py` for the resulting telemetry.
`analyze_friction.py` reads archived pre-telemetry summary CSVs.

### Evaluation tests

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen --no-sync python -m pytest \
  -q experiments/go2_friction/tests
```

## Offline performance metrics

`metrics.py` accepts a Step 2 dataset directory or the parent directory of a sweep. It writes to a fresh output directory and reads the raw traces without changing them.

```bash
uv run --frozen --no-sync experiments/go2_friction/metrics.py \
  experiments/go2_friction/results/step2_validation_i3yls91f \
  --output-dir "$(mktemp -d)/metrics" \
  --min-normal-force-n 5 --slip-speed-m-s 0.1
```

- `per_trial.csv` contains one row per scheduled trial with its source status, data issues, and scalar metrics. `derived/` contains time-by-foot NPZ arrays for the GRF ratio, friction-cone utilization, and stance and slip masks. Foot order comes from `metrics_metadata.json`.
- `summary.csv` groups trials by dataset, maneuver, friction, and test noise. Each metric has a mean over available trial values and its contributing count. Diagnostic means include valid samples from failed trials. The success rate is defined when every scheduled trial has an outcome.
- `critical_friction.csv` gives the lowest tested friction with at least nine successes in exactly ten scored `stand_2_stop` trials, separately for each dataset and test noise scale. The status distinguishes insufficient trials from a scored grid where every friction failed the rule.
- `metrics_metadata.json` records source datasets, thresholds, and any supplied hardware-readiness rubric. An optional `--readiness-rubric proposal_thresholds.json` accepts `minimum` and `maximum` maps keyed by `summary.csv` metrics or `critical_friction`. Readiness is `not_assessed` when the rubric or a required metric is unavailable.

### Metric definitions

The command schedule supplies the phase windows. A stage accelerates when the new commanded speed exceeds the old command projected onto the new direction by more than `--moving-command-m-s` (default `0.01`). It brakes when the old commanded speed exceeds the new command projected onto the old direction by that amount. A reversal or lateral transition can qualify for both windows. The final zero-command interval after movement is a braking window. A first-stage steady command has no preceding command transition.

| Metric | Definition |
|---|---|
| Stance-foot GRF ratio | $R = \sqrt{F_x^2 + F_y^2}/F_z$, using finite world-frame foot forces with an active ground contact and $F_z$ strictly above `--min-normal-force-n`. |
| GRF percentiles and peaks | The linearly interpolated 95th percentile and maximum of eligible foot samples over the full trial and in each acceleration and braking window. |
| Friction-cone utilization | $R/\mu$, where $\mu$ is the applied test friction recorded for the trial. Utilization is unavailable for zero or missing $\mu$. Values above one are retained. |
| Foot slip rate | Fraction of loaded stance-foot samples with finite world XY velocity and $\sqrt{v_x^2 + v_y^2}$ strictly above `--slip-speed-m-s`. Only samples with a usable horizontal velocity enter the denominator. |
| Velocity-tracking RMSE | $\sqrt{\mathrm{mean}((v_x-c_x)^2+(v_y-c_y)^2)}$ in body XY. The moving-stage RMSE uses only moving commands. Yaw-rate RMSE is computed separately in rad/s. |
| Acceleration rise time | Seconds from the first acceleration command onset to the first post-step sample where velocity projected onto the command direction reaches `--rise-fraction` of the target speed (default `0.9`). |
| Stopping distance | World XY path length from the sample immediately before the terminal stop command to the first sample in the continuous below-threshold interval that lasts through the final sample. The default `--stop-speed-m-s` is `0.15`. |
| Maximum pitch excursion | $\max_t |\mathrm{pitch}(t)|$ from upright, in radians. |
| Trial success | Survival through the full schedule, moving RMSE at most `--max-moving-rmse-m-s` (default `0.75`), mean signed speed along each moving command at least `--min-stage-speed-fraction` of its target (default `0.5`), and a successful stop when the schedule ends with a stop command. `stand_2_stop` requires a terminal stop. `--max-yaw-rmse-rad-s` can add a yaw-rate limit. |

Missing measurements produce blank scalar fields and NaNs in derived force arrays. A known fall or timeout can contribute diagnostic samples through its terminal frame. A numerical failure contributes only the prefix before its failure frame because MuJoCo can recover to a finite reset state during that frame. Missing traces, missing success inputs, and premature traces with no confirmed outcome remain unscored. They do not qualify a friction for the nine-of-ten rule.

### Offline tests

```bash
uv run --frozen --no-sync python -m pytest -q experiments/go2_friction/tests \
  -k 'not test_cpu_dataset and not test_cpu_autoreset'
```

## Offline analysis and figures

Run these commands from the repository root. Use the directory created by
`metrics.py` as input. Keep the Step 2 dataset directories referenced by
`metrics_metadata.json` accessible for training provenance and time histories.
Use fresh output directories for the analysis and figures.

```bash
mkdir -p experiments/go2_friction/results
step4="$(mktemp -d experiments/go2_friction/results/step4_XXXXXX)"

uv run --frozen --no-sync python experiments/go2_friction/analyze_results.py \
  experiments/go2_friction/results/step3_offline_metrics \
  --output-dir "$step4/analysis" \
  --nominal-friction 1.25 --cost-metric tracking_rmse_m_s --match-tolerance 0.05

uv run --frozen --no-sync python experiments/go2_friction/plot_results.py \
  "$step4/analysis" --output-dir "$step4/figures" \
  --maneuver stand_2_stop --test-noise-scale 0
```

For a new saved evaluation sweep, first run `metrics.py` on its Step 2 dataset
directory as shown above. Both Step 4 scripts also work with incomplete inputs.
The two-iteration smoke-test data have too few trials to establish critical
friction.

### Seed aggregation and matching

- Each training seed contributes at most one value per training configuration,
  maneuver, test noise, evaluation profile, checkpoint iteration, and test
  friction. The evaluation profile records the command schedule, planned friction
  grid, timing, evaluation backend, effective actor-noise levels, and relevant
  simulator settings. Training configurations use the embedded Step 1
  experiment name and parameters. A missing manifest leaves the dataset visible
  in the individual tables with unknown training provenance.
- When multiple evaluations cover the same seed and group, the analysis uses
  the dataset with the most fully scored friction conditions, then the most
  scored trials, then the first path in lexical order. Other datasets remain
  visible with `selected=False`.
- Each aggregate is the arithmetic mean of the available per-seed values. The
  sample SD uses (n-1) and is blank for fewer than two training seeds.
  `n_training_seeds` counts seeds with a row. `n_<metric>` counts seeds with a
  defined metric. Evaluation seeds and trial counts are reported independently.
  A seed-level success rate is available when all its scheduled trials for that
  condition are scored. Condition-level diagnostic means retain usable samples
  from failed trials as in Step 3.
- Nominal metrics use trials that survived through the full horizon at
  `--nominal-friction` (default `1.25`). Nominal planar tracking RMSE also
  requires a tracking sample at every step. The default cost is this RMSE in
  m/s. The trade-off figure uses the same paired training seeds on both axes
  for each training configuration.
- Critical friction uses the Step 3 nine-of-ten rule on the tested grid for
  `stand_2_stop`. The seed table preserves the cutoff status and counts any
  lower scheduled friction conditions with insufficient scored trials.
  Equivalent-noise comparisons use only available cutoffs in the same
  evaluation group. For each friction policy and configuration mean, the
  closest observed noise candidate minimizes the absolute cutoff difference.
  Ties favor the same training seed when comparing policies, then the lower
  tested noise multiplier. Candidate values remain visible when the absolute
  difference exceeds `--match-tolerance`. In that case
  `equivalent_noise_scale` is blank and `status` is `outside_tolerance`.
  Undefined cutoffs and absent candidates have distinct statuses. Differences
  are noise candidate minus friction target.

### Output files

| File | Contents |
|---|---|
| `seed_conditions.csv` | Per-seed condition values, evaluation seeds, scored and unscored trial counts, and contributing trial counts for each metric. |
| `condition_summary.csv` | Across-seed condition means, sample SDs, and sample counts. |
| `seed_nominal.csv` | Per-seed cutoffs and statuses, nominal metric values, and contributing trial counts. |
| `nominal_summary.csv` | Across-seed critical friction and nominal metrics, including paired-seed summaries for the trade-off axes. |
| `equivalent_noise.csv` | Nearest observed noise candidate per friction policy and configuration mean, matching status, and nominal performance and GRF comparisons. Counts refer to trials for policy rows and training seeds for configuration rows. |
| `traces.csv` | Training and evaluation seeds, trial outcome, applied friction, and source and derived NPZ paths. |
| `analysis_metadata.json` | Analysis settings, Step 3 thresholds, evaluation profile definitions, and dataset selection. |

The plotter writes the 13 numbered figure pairs as PNG (300 DPI by default)
and PDF. Each output subdirectory corresponds to one maneuver, test noise
scale, evaluation profile, and checkpoint iteration. Figures 1–5 and 7 show
test-friction and critical-friction results. Figures 9–11 show rise time, stopping
distance, and tracking RMSE at nominal friction. Figure 12 places nominal cost
on the horizontal axis and critical friction on the vertical axis for both
training methods. The faint points in summary figures show individual training
seeds.

Figures 6, 8, and 13 show one recorded trial's phase-specific stance-foot GRF
ratios against applied friction, commanded and measured body velocity, and
signed pitch in degrees. The command is drawn over its executed interval.
The default trace is selected from the retained datasets, using friction closest
to nominal and preferring a completed trial at that friction. Use
`--trace-id trace_000123` with an ID from `traces.csv` to select a specific
trial, or use `--trace-friction 0.8` to restrict the automatic trace selection.
Use `--evaluation-profile` to select a profile from `analysis_metadata.json`.
The figure manifest `figures.json` records the selected trial and any trace
loading issue. Missing metrics leave gaps and unavailable panels. A fallen
trial's history ends at its recorded horizon.

### Reproduce the synthetic validation

These commands generate labeled analytic fixtures and keep their CSVs and
figures under a separate synthetic directory. They require no checkpoint loads
or physics simulations.

```bash
mkdir -p experiments/go2_friction/results
synthetic="$(mktemp -d experiments/go2_friction/results/synthetic_step4_XXXXXX)"

uv run --frozen --no-sync python experiments/go2_friction/tests/synthetic_step4.py \
  "$synthetic/evaluation"
uv run --frozen --no-sync python experiments/go2_friction/metrics.py \
  "$synthetic/evaluation" --output-dir "$synthetic/metrics"
uv run --frozen --no-sync python experiments/go2_friction/analyze_results.py \
  "$synthetic/metrics" --output-dir "$synthetic/analysis"
uv run --frozen --no-sync python experiments/go2_friction/plot_results.py \
  "$synthetic/analysis" --output-dir "$synthetic/figures" \
  --maneuver stand_2_stop --test-noise-scale 0

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen --no-sync python -m pytest \
  -q experiments/go2_friction/tests -k 'not test_cpu_dataset and not test_cpu_autoreset'
```

## End-to-end CPU integration check

These commands use the three saved two-iteration checkpoints. The primary grid
runs 36 trials across the three policies, three frictions, two test noise scales,
and two evaluation seeds. The baseline checkpoint also runs 24 short trials
covering the other maneuver types. The replay checks live actor inputs, frozen
checkpoint tensors, native friction, force, kinematics, and the trial clock.

```bash
exp_dir="$PWD/experiments/go2_friction"
python_bin="$PWD/.venv/bin/python"
baseline="$PWD/logs/go2noise_scale_0p00_seed_42/Nov23_20-32-25_/model_2.pt"
friction="$PWD/logs/go2friction_lb_0p40_seed_42/Nov23_20-35-00_/model_2.pt"
noise="$PWD/logs/go2noise_scale_2p00_seed_42/Nov23_20-37-34_/model_2.pt"
mkdir -p "$exp_dir/results"
step5="$(mktemp -d "$exp_dir/results/step5_XXXXXX")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cat > "$step5/short_maneuvers.json" <<'JSON'
{
  "steady_0p5": [[0.36, 0.5, 0, 0]],
  "steady_1p0": [[0.36, 1.0, 0, 0]],
  "steady_1p5": [[0.36, 1.5, 0, 0]],
  "steady_2p0": [[0.36, 2.0, 0, 0]],
  "forward_backward": [[0.16, 0, 0, 0], [0.24, 1, 0, 0], [0.24, -1, 0, 0]],
  "forward_sideways": [[0.16, 0, 0, 0], [0.24, 1, 0, 0], [0.24, 0, 1, 0]]
}
JSON

"$python_bin" -B -m pytest -p no:cacheprovider -q "$exp_dir/tests"

(
  cd "$step5" || exit
  grid=(--checkpoint-path "$baseline" --checkpoint-path "$friction"
    --checkpoint-path "$noise" --frictions 0.4,1.0,1.25 --noise-scales 0,1
    --trials 2 --eval-seeds 10000,10001 --num-envs 2 --device cpu
    --maneuvers stand_2_stop --output-dir evaluation/primary)
  "$python_bin" -B "$exp_dir/sweep_evaluation.py" "${grid[@]}" --dry-run
  "$python_bin" -B "$exp_dir/sweep_evaluation.py" "${grid[@]}"
  "$python_bin" -B "$exp_dir/evaluate_friction.py" \
    --checkpoint-path "$baseline" --frictions 1.0 --noise-scales 0,1 \
    --trials 2 --eval-seeds 10000,10001 --num-envs 2 --device cpu \
    --maneuvers steady_0p5,steady_1p0,steady_1p5,steady_2p0,forward_backward,forward_sideways \
    --maneuver-config short_maneuvers.json --output-dir evaluation/short

  "$python_bin" -B "$exp_dir/tests/verify_live_evaluation.py" evaluation
  "$python_bin" -B "$exp_dir/metrics.py" evaluation --output-dir metrics
  "$python_bin" -B "$exp_dir/metrics.py" evaluation/primary \
    --output-dir rise_time_probe --rise-fraction 0.02
  "$python_bin" -B "$exp_dir/analyze_results.py" metrics --output-dir analysis
  "$python_bin" -B "$exp_dir/plot_results.py" analysis \
    --output-dir figures --maneuver stand_2_stop
  "$python_bin" -B "$exp_dir/plot_results.py" analysis \
    --output-dir figures_sideways --maneuver forward_sideways --test-noise-scale 1
)
```

The separate rise-time probe checks threshold crossing on live data at 2% of the
2 m/s command. The standard metrics retain the 90% threshold. Two trials per
condition leave critical friction and equivalent noise undefined.

### Recheck reordered and incomplete data

```bash
(
  cd "$step5" || exit
  baseline_data="evaluation/primary/go2noise_scale_0p00_seed_42/Nov23_20-32-25_/model_2"
  "$python_bin" -B "$exp_dir/evaluate_friction.py" \
    --experiment-name go2noise_scale_0p00_seed_42 --load-run Nov23_20-32-25_ \
    --checkpoint 2 --frictions 1.25,1.0,0.4 --noise-scales 0,1 \
    --trials 2 --eval-seeds 10000,10001 --num-envs 2 --device cpu \
    --maneuvers stand_2_stop --output-dir reordered_evaluation
  mkdir duplicate_input
  cp -R "$baseline_data" duplicate_input/original
  cp -R reordered_evaluation duplicate_input/reordered
  "$python_bin" -B "$exp_dir/metrics.py" duplicate_input --output-dir duplicate_metrics
  "$python_bin" -B "$exp_dir/analyze_results.py" duplicate_metrics \
    --output-dir duplicate_analysis

  cp -R "$baseline_data" partial_evaluation
  "$python_bin" -B - <<'PY'
import csv
import json
from pathlib import Path

index = Path("partial_evaluation/trials.csv")
with index.open(newline="") as source:
    rows = list(csv.DictReader(source))
retained = [
    row for row in rows
    if (row["friction"], row["noise_scale"], row["eval_seed"]) != ("1.0", "0.0", "10001")
]
assert len(rows) - len(retained) == 1
with index.open("w", newline="") as target:
    writer = csv.DictWriter(target, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(retained)
manifest = index.with_name("metadata.json")
metadata = json.loads(manifest.read_text())
metadata.update(status="interrupted_or_failed", trials_written=len(retained))
manifest.write_text(json.dumps(metadata, indent=2) + "\n")
PY
  "$python_bin" -B "$exp_dir/metrics.py" partial_evaluation --output-dir partial_metrics
  "$python_bin" -B "$exp_dir/analyze_results.py" partial_metrics --output-dir partial_analysis
)
```

The reordered copy should contribute one training seed per condition.
The incomplete copy should report one unscored trial and a blank success rate
at test friction 1.0 and test noise 0.

The fault test injects a MuJoCo recovery and a timeout in separate worlds. To
inspect those trial traces through Step 3, retain its temporary dataset.

```bash
"$python_bin" -B -m pytest -p no:cacheprovider -q --basetemp "$step5/fault_test" \
  "$exp_dir/tests/test_evaluation.py::test_cpu_autoreset_and_timeout_end_only_the_affected_trials"
"$python_bin" -B "$exp_dir/metrics.py" "$step5/fault_test" \
  --output-dir "$step5/fault_metrics"
```

## Subsequent pilot training

This pilot uses one training seed, 25 iterations, and 32 CPU environments per
policy. It runs the friction-randomized policy and the two fixed-friction noise
settings used in the CPU integration check. Append `--dry-run` to preview them.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen --no-sync python \
  experiments/go2_friction/sweep_friction.py --lower-bounds 0.4 --upper 1.25 \
  --seeds 42 --max-iterations 25 --num-envs 32 --device cpu --backend mujoco

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen --no-sync python \
  experiments/go2_friction/sweep_noise.py --noise-scales 0,2 \
  --seeds 42 --max-iterations 25 --num-envs 32 --device cpu --backend mujoco
```
