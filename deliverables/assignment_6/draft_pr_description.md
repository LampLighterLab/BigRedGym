# Draft PR title

`[DRAFT] Evaluate training-free Go2 walking-to-height composition`

# Project scope and planned assessment

This draft implements the submitted project plan: test whether live action blending of two frozen policies can support a simpler Go2 walking-to-crouching transition without training another controller. Policy A is pretrained Go2Trot walking and policy B is Go2Height height control. Both remain frozen and receive their own observations.

The planned comparison is B alone, immediate A-to-B switching, and 0.25/0.50/1.00 s live target blending. The full simulation assessment will use commands of 0, 0.10, and 0.20 m/s, followed by a stop command and a 0.35-to-0.20 m height change, with at least 50 randomized gait-phase trials per condition and then mass and PD-gain perturbations. Success requires no fall plus height error <=0.025 m, forward speed <=0.10 m/s, and yaw-rate error <=0.15 rad/s within 2 s, sustained for 0.5 s. Reports will include 95% confidence intervals, settling time, peak roll/pitch, peak target step, peak torque fraction, and pre-handoff walking speed.

Sim-to-real readiness requires at least 95% success in 50 held-out randomized-dynamics trials, no falls, maximum tilt <=15 degrees, and no joint-limit requests. Only after that gate would the plan proceed to five supervised, low-speed Go2 transitions.

# Preliminary results (not the full planned assessment)

- A survived 30/30 reduced walking-gate cases but passed 0/30 because of jitter and gait-quality failures.
- In the composition diagnostic, A reached mean pre-transition speeds of 0.1674 and 0.3228 m/s for 0.20 and 0.40 m/s commands.
- Across four deterministic gait phases at each speed, B alone and immediate switching each succeeded in 4/8 trials.
- The 0.25/0.50/1.00 s blends succeeded in 7/8, 6/8, and 3/8 trials, respectively; every composition trial remained upright.
- Steady height RMSE ranged from 0.0053 to 0.0213 m, and maximum tilt across conditions was 12.18 degrees.
- The best 0.25 s blend produced 0.5416-0.5870 rad peak joint-target steps.

These results are preliminary and do not establish hardware readiness. The best result, 7/8 (87.5%), is below the 95% readiness threshold; the planned 50 held-out randomized-dynamics trials, joint-limit evidence, and hardware tests have not been completed.

# Changes

- Register Go2Height and its height command.
- Add reproducible MuJoCo evaluators for A, B, immediate switching, and live blending.
- Add unit coverage for composition and readiness metrics.
- Add a synchronized six-second simulation comparison video and action-output check.

# Validation already completed

- `python -m pytest tests/unit_tests/test_go2_composition.py tests/unit_tests/test_go2height_readiness.py tests/unit_tests/test_go2trot_performance.py gym/envs/go2/test_go2trot.py -q`
- Result: `24 passed`.
- Evaluation data are recorded in `output/data/assignment_6_go2_composition.json`, `output/data/assignment_6_go2height_readiness.json`, and `output/data/assignment_6_go2trot_readiness.json`.

# Demo reproduction

From the repository root, run:

`uv run --frozen --with imageio --with imageio-ffmpeg python -m scripts.render_go2_comparison_video`

This sequentially simulates A walking, immediate A-to-B switching, and 0.25 s live blending under identical conditions, then writes the synchronized six-second stack to `output/video/assignment_6_frozen_policy_comparison_sequential.mp4`. During the blending window, the recorded maximum action difference from B is 0.5607, while immediate switching matches B to numerical precision.

# Known limitations

- MuJoCo CPU only; no hardware or cross-backend evidence.
- The preliminary composition diagnostic used four deterministic phases at two speeds, not the full submitted assessment.
- A survives but fails the strict gait-quality gate.
- The best blend remains below the submitted sim-to-real readiness threshold and has a large peak target step.
