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


- Action space -> multiple tokens

Reinforcement learning and espeically policy gradient based methods are widely used for LLM post training. The standard way to model LLM generation is to treat the current prefix as the state and each new token as a new action. The generated token is then appended to the prefix resulting in a new state. State transition dynamics are therefore trivially known. Treating each new token as an action is a simple and natural thing to do but is it the right thing to do? In other words is it a good model of what an LLM does? There's a growing body of literature that suggests that LLM representations encode information about future tokens and not just the next token (latent planning). This might suggest that LLMs would 'like to' output multiple tokens corresponding to its plan at a particular state, but are constrained by the next token prediction objective to only output one token at a time. (Does emitting a token change the plan?). If this is indeed the case, perhaps it would make sense to treat this plan of multiple tokens as an action. Plan length -> Spec decoding -> action -> PPO/SDPO etc.


- Objective on the final response, iteratively refined during COT




- Lean use within COT, NLAC, SFT first from stronger model

- Objective for 'good' COT

- Edit context during rollout
