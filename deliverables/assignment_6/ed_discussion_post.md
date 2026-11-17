# Training-free composition of Go2 walking and height control

**Team:** Huaijin Hu (hh833), Zoey Mo (zm73)

## Project description and planned assessment

Our submitted project asks whether short, training-free action blending can produce reliable low-speed Go2 walking-to-crouching transitions without training another controller. Policy A is pretrained Go2Trot walking and policy B is Go2Height height control. Both policies remain frozen and receive their own observations. We compare B alone, immediate A-to-B switching, and live target blending over 0.25, 0.50, and 1.00 s.

The planned simulation assessment uses initial speed commands of 0, 0.10, and 0.20 m/s, followed by a stop command and a 0.35-to-0.20 m height change. We will run at least 50 randomized gait-phase trials per condition, followed by mass and PD-gain perturbations. Success requires no fall plus height error <=0.025 m, forward speed <=0.10 m/s, and yaw-rate error <=0.15 rad/s within 2 s, sustained for 0.5 s. We will report 95% confidence intervals, settling time, peak roll/pitch, peak target step, peak torque fraction, and pre-handoff walking speed.

The sim-to-real gate is at least 95% success over 50 held-out randomized-dynamics trials, no falls, maximum tilt <=15 degrees, and no joint-limit requests. Only after passing that gate would we request five supervised, low-speed Go2 transitions (<=0.10 m/s and <=0.05 m height change), reporting height error, settling time, peak tilt, and safety interventions.

## Preliminary results

These are diagnostic results and are separate from the planned assessment above. Go2Trot survived 30/30 reduced walking-gate cases but passed 0/30 because of jitter and gait-quality failures. At 2 Hz, linear-velocity RMSE was 0.12 m/s for a 0.50 m/s command and 0.30 m/s for a 1.00 m/s command.

The composition diagnostic used four deterministic gait phases at each of two commanded speeds, 0.20 and 0.40 m/s. A achieved mean pre-transition speeds of 0.1674 and 0.3228 m/s. Results were:

- B alone: 4/8 successes.
- Immediate A-to-B switching: 4/8.
- 0.25 s blend: 7/8.
- 0.50 s blend: 6/8.
- 1.00 s blend: 3/8.

Every composition trial remained upright. Steady height RMSE ranged from 0.0053 to 0.0213 m, and maximum tilt across conditions was 12.18 degrees. The best 0.25 s blend still produced 0.5416-0.5870 rad peak joint-target steps. Recorded actions verify that immediate switching applies B directly, while the 0.25 s case interpolates live outputs and differs from B by as much as 0.5607 during the blend window.

## Readiness assessment

This is **not ready for hardware**. The best preliminary result is 7/8 (87.5%), below the submitted 95% gate. We have not completed 50 held-out randomized-dynamics trials, demonstrated the joint-limit requirement, resolved A's gait-quality failures, or reduced the blend's large peak target step. No hardware tests have been run.

## Links

- Draft PR: https://github.com/LampLighterLab/BigRedGym/pull/23
- Synchronized simulation video: https://github.com/CMDHL/BigRedGym/blob/assignment-6-training-free-composition/output/video/assignment_6_frozen_policy_comparison_sequential.mp4

## Feedback requested

1. Is the <=0.10 m/s stopping threshold appropriate for the first supervised hardware test?
2. Should peak joint-target step or peak torque fraction be the primary smoothness/safety gate?
3. Is it reasonable to treat A's gait jitter as a separate limitation while evaluating transition composition?

Related work: Christmann, Luo, and Chen, “Expert Composer Policy: Scalable Skill Repertoire for Quadruped Robots,” ICRA 2024: https://arxiv.org/abs/2403.11412
