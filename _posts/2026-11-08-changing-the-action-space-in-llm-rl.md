---
title: 'Changing the action space in LLM RL'
date: 2026-11-08
permalink: /posts/2026-11-08-changing-the-action-space-in-llm-rl/
---

- Noisy advantage in LLM RL

  ![Illustrative token-level advantages: each token has a synthetic advantage, with coral indicating negative values and green indicating positive values.]({{ '/images/llm-rl/token-advantages.svg' | relative_url }})

- Plan

  ![Prefix activations and five later states are compressed into discrete codes. The first three future tokens are highlighted as an illustrative selection based on high mutual information, estimated across many rollouts.]({{ '/images/llm-rl/planning-horizon.svg' | relative_url }})

- Role of action space in discrete RL settings

- Spec decoding
