---
title: "Training-Free Composition of Go2 Walking and Height Control"
author: "Huaijin Hu (hh833), Zoey Mo (zm73)"
date: ""
documentclass: article
fontsize: 11pt
papersize: letter
geometry: margin=0.85in
pagestyle: empty
---

## Context

Expert Composer Policy trains a reusable transition policy between frozen quadruped skills and reports 99.99% mean transition success in simulation and 97.22% on hardware (Christmann, Luo, and Chen, 2024). We ask a narrower question: when walking and height-control policies share the same 12 joint-position-target action space, can direct switching or live output blending provide useful walk-to-crouch transitions without training another policy?

## Contribution

We will characterize the speed and height-change region in which training-free composition works, and identify why it fails outside that region. Policy A is a frozen Go2Trot walking policy; policy B is the frozen Go2Height policy. Each receives its own observation vector, then their live joint targets are either switched or blended. No new policy is trained. A learned policy C is reserved for future work only if the training-free baseline is inadequate.

Preliminary MuJoCo evidence now includes both frozen policies. A survived all 30 fixed-command cases in a reduced performance gate, but passed 0/30 because of joint jitter and gait-quality failures; at 2 Hz, its linear-velocity RMSE was 0.12 m/s for a 0.50 m/s command and 0.30 m/s for 1.0 m/s. In the composition test, A reached 0.17 and 0.32 m/s before transitions commanded from 0.20 and 0.40 m/s. Across four gait phases at each speed, B alone and direct switching each succeeded in 4/8 trials, while 0.25, 0.50, and 1.00 s blends succeeded in 7/8, 6/8, and 3/8. All methods had 0.01-0.02 m steady height RMSE and at most 12.2 degrees tilt. The best 0.25 s blend still produced 0.54-0.59 rad peak step-to-step joint-target changes, so it is not yet hardware-ready.

## Experiment and quantitative assessment

A and B use the same joint order, action scaling, limits, and control rate while retaining separate observation vectors. Compare B alone, immediate A-to-B switching, and linear live-action blends of 0.25, 0.50, and 1.00 s. From 0, 0.10, and 0.20 m/s, command a stop and a 0.35-to-0.20 m height change. Run at least 50 trials per condition with randomized gait phase; then repeat the selected method with domain-randomized mass and PD gains.

Simulation success requires no fall and simultaneous height error <=0.025 m, forward speed <=0.10 m/s, and yaw-rate error <=0.15 rad/s within 2.0 s, sustained for 0.5 s. Report success rate with a 95% confidence interval, settling time, maximum roll/pitch, peak step-to-step joint-target change, and peak torque fraction. Comparisons to B alone distinguish transition quality from B's locomotion limitation.

Sim-to-real readiness requires at least 95% success in 50 held-out randomized trials, zero falls, maximum tilt <=15 degrees, and no joint-limit request. The best current method is 7/8 in a small deterministic test, while A fails the strict gait-quality gate, so the system is not hardware-ready. Once simulation passes, request five supervised hardware trials at <=0.10 m/s and <=0.05 m height change; require height error <=0.03 m and speed <=0.10 m/s within 3.0 s, while reporting peak tilt and every safety intervention.

Reference: G. Christmann, Y.-S. Luo, and W.-C. Chen, “Expert Composer Policy: Scalable Skill Repertoire for Quadruped Robots,” ICRA, 2024. https://arxiv.org/abs/2403.11412
