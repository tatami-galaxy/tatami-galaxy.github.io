# TODO: RL Objectives and Credit Assignment for LLMs

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



- Lean use within COT, NLAC, SFT first from stronger model

- Objective for 'good' COT

- Edit context during rollout

- AR vs diffusion, critic vs grpo -> linguistics

