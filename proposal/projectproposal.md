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

# *Performance Driven Idea*

## Context

The paper *"Learning Emergent Gaits with Decentralized Phase Oscillators: on the role of Observations, Rewards, and Feedback"* explores how gaits can naturally emerge in a quadruped robot using four decentralized oscillators, one for each leg. In this configuration, the oscillators are indirectly coupled via ground-reaction forces. During training, the RL policy receives a phase-based reward that encourages leg-to-ground contact during the stance phase and penalizes it during the swing phase:
$$\begin{equation}
r_{\text{gait}}(x)=-F_i^{\text{GRF}}sin(\phi_i)
\end{equation}$$

At runtime, the oscillator phases are accessible by the RL policy. Testing showed that this approach allowed the robot to naturally develop coordinated gaits with high disturbance rejection.

The oscillator dynamics introduced in the paper are given by:
$$\begin{equation}
\dot{\phi}_i = 2\pi\left(\omega - \sigma F_i^{\mathrm{GRF}}\left(\cos(\phi_i) + \xi\right)\right)
\end{equation}$$

where $\omega$ is the nominal frequency, $\sigma$ the feedback gain and $\xi$ an offset. The ground reaction force allows each oscillator to naturally speed up or slow down depending on the interaction of the respective leg with the ground. Coordination between the legs can therefore emerge without explicitly prescribing a gait.

## Contribution

While this approach shows promising results for locomotion on flat ground, we are interested in investigating its capability to navigate more complex terrain such as stairs. Climbing a staircase involves discrete, leg-specific movements, during which each leg must periodically generate sufficient lifting force to raise the robot onto the next step. However, despite their increased complexity, these tasks still involve cyclic and symmetric movements across the legs, suggesting that decentralized oscillators could remain beneficial.

We propose replacing the original force-feedback law with Model Predictive Control (MPC) adjusting the frequency of each oscillator based on the predicted motion of the robot. The objective is to allow the robot to naturally adapt its gait to changes in terrain without prior knowledge of its geometry.

The proposed oscillator dynamics are given by:
$$\begin{equation}
\dot{\phi}_i = 2\pi\left(\omega + \text{MPC}(\phi_i,x_i))\right)
\end{equation}$$

We still have no explicit coupling between the oscillators. We also propose defining a desired ground reaction force profile $\mathbf F_i^*(\phi_i)$ for each oscillator.
Rather than only rewarding contact during stance, the RL policy would be rewarded for reproducing the desired force at each phase:
$$\begin{equation}
r_{\text{MPC}}(x)=F_i^*(\phi_i)-F_i^{\text{GRF}}
\end{equation}$$

The desired force would be zero during swing and follow a chosen force profile during ground contact. By training the RL policy to reproduce these forces, the MPC controller can adjust the oscillator rates and predict the resulting motion based on the expected ground reaction forces  $F_i^*(\phi_i)$. The idea is that the RL policy will allow us to reduce the complex quadruped dynamics to a simpler model ot the forces exerted by each leg, making it easier for the MPC controller to handle more complicated locomotion tasks.

![System Diagram](diagram.png)

## Proposed Experimental Steps and Success Criterias

We will evaluate the quadruped's ability to track a commanded forward velocity while adapting its gait to unknown terrain. We will implement our control architecture progressively, starting with decentralized oscillator only on flat ground, then integrating MPC before introducing uneven terrain/stairs.

> **Phase 1: Locomotion without MPC**
>
> First, we will test the robot on flat ground using decentralized oscillators with a fixed frequency and ground reaction force feedback, as described in equation (2). This will allow us to evaluate the RL policy's ability to reproduce the desired force profiles defined in Equation (4) while maintaining stability.
> 
> We will measure velocity fluctuations and center-of-mass motion, and observe whether coordinated gaits naturally emerge.

> **Phase 2: MPC control on flat ground**
>We will then integrate the MPC controller into the oscillator dynamics, as described in equation (3), while keeping the same RL policy. The MPC will guide the robot between predefined waypoints. 
>
>We will measure the optimality of the trajectory, the accuracy of ground reaction force tracking by the RL policy, the stability of the robot throughout its motion, as well as the emergence and evolution of different gaits.

> **Phase 3: MPC control on complex terrain**
>
>Finally, we will test the controller on more challenging tasks, particularly stairs, without providing information about terrain geometry. We will use the same performance metrics as in phase 2, and add the robot's ability to successfully climb stairs of increasing height and observing how its gait adapts to changes in ground contact.


