---
title: 'On RL Objectives and Credit Assignment for LLMs'
date: 2026-10-21
permalink: /posts/2026-10-21-on-rl-objectives-and-credit-assignment-for-llms/
---
Maximum Entropy RL :

$$
J(\theta)
=
\sum_t
\mathbb{E}_{\tau \sim \pi_\theta(\tau)}
\left[
r(s_t, a_t)
+
\beta H\!\left(\pi_\theta(\cdot \mid s_t)\right)
\right]
$$
