---
title: 'Changing the action space in LLM RL'
date: 2026-11-08
permalink: /posts/2026-11-08-changing-the-action-space-in-llm-rl/
---

- Noisy advantage in LLM RL

  ![Illustrative token-level advantages: each token has a synthetic advantage, with coral indicating negative values and green indicating positive values.]({{ '/images/llm-rl/token-advantages.svg' | relative_url }})

- Plan

  ![Prefix activations and five later states are compressed into discrete codes. The first three future tokens are highlighted as an illustrative selection based on high mutual information, estimated across many rollouts.]({{ '/images/llm-rl/planning-horizon.svg' | relative_url }})

- Spec decoding

  ![A diffusion drafter conditioned on target-model context features proposes five tokens in parallel. Target verification accepts the first three, rejects the fourth, and discards the fifth. The accepted draft length h is an illustrative proxy for plan length.]({{ '/images/llm-rl/speculative-plan-length.svg' | relative_url }})

- Role of action space in discrete RL settings

  ![Changing available controls has task-dependent effects: removing backward movement sped up learning in Obstacle Tower, while more movement options improved performance in harder Doom scenarios.]({{ '/images/llm-rl/action-space-shaping.svg' | relative_url }})

  The available controls can change how easily an RL agent learns. In [Action Space Shaping in Deep Reinforcement Learning](https://arxiv.org/html/2004.00980v2), allowing backward movement slowed learning in Obstacle Tower, while increasing the available controls improved performance in harder Doom scenarios. A smaller action set can simplify exploration, but it can also remove useful behavior.

  ![The same 18 Atari actions represented as flat composite choices or as horizontal, vertical, and firing factors. Up-right-fire is highlighted in both. FARA3C outperformed A3C on nine of fourteen tasks.]({{ '/images/llm-rl/factored-action-representations.svg' | relative_url }})

  Even when the available actions stay the same, their representation can affect learning. [Learning to Factor Policies and Action-Value Functions](https://arxiv.org/html/1705.07269v1) represents Atari actions through horizontal movement, vertical movement, and firing factors. Shared factor outputs couple the estimates for related actions; its factored actor-critic variant, FARA3C, outperformed A3C on 9 of 14 tasks, with A3C scoring higher on the remaining five.

  These results motivate examining action design for LLMs. Grouping successive tokens introduces temporal abstraction, however, which is a different change from selecting controls or factoring simultaneous actions; the benefits for LLM credit assignment remain to be tested.
