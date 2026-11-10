---
title: 'Changing the action space in LLM RL'
date: 2026-11-08
permalink: /posts/2026-11-08-changing-the-action-space-in-llm-rl/
---

- Noisy advantage in LLM RL

Each new token is treated as a new action in LLM + critic based RL like PPO, Self Distillation etc. The critic is asked to estimate the value for each token and as a result its forced to compute noisy advantages : 

  ![Illustrative token-level advantages: each token has a synthetic advantage, with coral indicating negative values and green indicating positive values.]({{ '/images/llm-rl/token-advantages.svg' | relative_url }})

- Internal Planning

There's growing evidence that suggests that LLM representations encode information about future tokens and not just the next token (latent planning). This might suggest that LLMs would 'like to' output multiple tokens corresponding to its plan at a particular state, but are constrained by the next token prediction objective to only output one token at a time. If this is indeed the case, perhaps it would make sense to treat this plan of multiple tokens as an action : 

  ![Prefix activations and five later states are compressed into discrete codes. The first three future tokens are highlighted as an illustrative selection based on high mutual information, estimated across many rollouts.]({{ '/images/llm-rl/planning-horizon.svg' | relative_url }})

- Speculative decoding

We could also use a diffusion drafter and its accepted draft length as a cheap proxy for internal plan length : 

  ![A diffusion drafter conditioned on target-model context features proposes five tokens in parallel. Target verification accepts the first three, rejects the fourth, and discards the fifth. The accepted draft length h is an illustrative proxy for plan length.]({{ '/images/llm-rl/speculative-plan-length.svg' | relative_url }})

- Role of action space in discrete RL settings

There's a lot of evidence in RL that different choices of the action space for the same RL objective and algorithm can have very different consequences : 

  ![Changing available controls has task-dependent effects: removing backward movement sped up learning in Obstacle Tower, while more movement options improved performance in harder Doom scenarios.]({{ '/images/llm-rl/action-space-shaping.svg' | relative_url }})

  The available controls can change how easily an RL agent learns. In [Action Space Shaping in Deep Reinforcement Learning](https://arxiv.org/html/2004.00980v2), allowing backward movement slowed learning in Obstacle Tower, while increasing the available controls improved performance in harder Doom scenarios.

  ![The same 18 Atari actions represented as flat composite choices or as horizontal, vertical, and firing factors. Up-right-fire is highlighted in both. FARA3C outperformed A3C on nine of fourteen tasks.]({{ '/images/llm-rl/factored-action-representations.svg' | relative_url }})

  Even when the available actions stay the same, their representation can affect learning. [Learning to Factor Policies and Action-Value Functions](https://arxiv.org/html/1705.07269v1) represents Atari actions through horizontal movement, vertical movement, and firing factors. Shared factor outputs couple the estimates for related actions; its factored actor-critic variant, FARA3C, outperformed A3C on 9 of 14 tasks, with A3C scoring higher on the remaining five.

- Dynamic action space for LLM RL

  ![The original response grouped into variable-length internal planning windows. Each window is one action with a shared advantage color. Boundaries and advantages are illustrative, not learned.]({{ '/images/llm-rl/planning-window-advantages.svg' | relative_url }})

  Each action now spans a variable-length internal planning window, with one advantage assigned to the whole window.
