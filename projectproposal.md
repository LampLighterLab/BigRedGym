---
title: "Adaptive Quadruped Locomotion Using MPC-Tuned Oscillators and RL"
author: "Utku Melemetci (um44), Sidharth Rao (smr353), Aloyse Maille (aam299)"
date: "11/12"
documentclass: article
fontsize: 11pt
papersize: letter
geometry: margin=0.85in
pagestyle: empty
header-includes:
  - |
    ```{=latex}
    \usepackage{graphicx}
    \usepackage{titlesec}
    \titleformat{\section}{\normalfont\large\bfseries}{}{0pt}{}
    \titlespacing*{\section}{0pt}{1em}{0.35em}
    \setlength{\parindent}{0pt}
    \setlength{\parskip}{0.5em}
    \raggedbottom
    \makeatletter
    \renewcommand{\maketitle}{%
      \begingroup\centering
      {\Large\bfseries\@title\par}
      \vspace{0.35em}
      {\normalsize\@author\par}
      \endgroup\vspace{0.6em}}
    \makeatother
    % Start each example on a new PDF page without raw LaTeX in the Markdown body.
    \newif\ifoutlinefirstsection
    \outlinefirstsectiontrue
    \let\outlinesection\section
    \renewcommand{\section}{%
      \ifoutlinefirstsection\outlinefirstsectionfalse\else\clearpage\fi
      \outlinesection}
    ```
---

<!--
Copy this template for each proposal and replace the title, names, and prompts.
The headings are a starting point; adapt them if another structure communicates
your idea more clearly. This is an exploration, so say what you are still unsure about.
Include related work where it helps explain your idea.

Build from the folder containing your copy:
  pandoc project_outline.md --from=gfm+raw_attribute+smart --pdf-engine=pdflatex -o project_outline.pdf

The matching LaTeX template is generated with:
  pandoc project_outline.md --from=gfm+raw_attribute+smart --standalone -o project_outline.tex
-->

# *Performance-driven idea*

## What do you want to achieve?

The goal is to develop an adaptive gait controtroller for the Go2 robot that can maintain a commanded gait speed while adapting to inclined and uneven terrain.

The controller will combine decentralized oscillators (CPGs), Model Predictive Control (MPC) and Reinforcement Learning (RL).

The CPGs will generate periodic motion, with each oscillator corresponding to one leg. The MPC will adjust oscillator parameters, such as frequency and amplitude, based on the desired velocity and the current state of the robot. The RL controller will use the resulting gait references and sensor feedback to generate joint-level actions and maintain stability.

The main objective is to determine whether dynamically adjusting the oscillators through MPC improves the robot's ability to maintain its commanded velocity and remain stable on a sloped and then rough terrain, compared with using fixed oscillator parameters. In summary, we are comparing system behavior with and without the MPC control. 

We will first develop and evaluate the controller on flat ground before moving on the sloped ground. A stretch goal would be to extend the evaluation to uneven terrain with small obstacles and irregular floor heights.

## How will you achieve it?

- CPGs: Each leg will have an oscillator that determines the timing of its periodic motion. The oscillators will be coordinated to produce a gait.

The oscillator parameters will determine properties such as stepping frequency, step amplitude and relative phase between legs. 

- MPC: It will predict the robot's motion over a shot time horizon and optimize selected oscillator parameters to minimize the difference between the commanded and predicted velocities.

For example, as the robot evolves on a sloped floor, the MPC will adjust stepping frequency or step amplitude to compensate.

- RL: It will receive the desired gait reference from the CPGs, as well as robot state information (orientation, joint positions, velocities).

It will learn to generate joint-level commands that follow the desired gait while maintaining balance and adapting to changes in ground contact.

## How will you show it?