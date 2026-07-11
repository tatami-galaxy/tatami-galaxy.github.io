---
title: 'Self Distillation'
date: 2026-07-01
permalink: /posts/2026-07-01-self-distillation-post-1./
---

Self Distillation has recently come up as a promising direction for language model post training. It targets two major shortcomings of dominant RL based post training algorithms like GRPO. GRPO requires a verifier signal (ex : answer correctness) which might be hard to obtain for tasks where there isnt a clear notion of correctness. GRPO also has a credit assignment problem : for positive advantages, every token in the rollout is equally upweighted. The algorithm cannot assign granular credit to parts of the rollout, unlike its predecessor PPO, which suffers from its own inefficiencies and instabilities. 

Self Distillation instead distills from the self-teacher; a privileged information (PI) conditioned student model where the PI can be the correct answer, a demonstration, environment feedback etc. This has the advantage of granular feedback : the logit level KL (or some other f-divergence) at every token position is non-uniform. Ideally (we hope) it upweights correct tokens while downweighting incorrect ones. It also does not necessarily require a verifier : the PI can be constructed from any "helpful" information. The key assumption here is that the model's own in-context learning (ICL) abilities will allow it accurately leverage PI to not only solve the problem (it can do so trivially when the PI is a demonstration), but to also accurately critique[^critique] its own rollouts. 

While this works well for a range of domains, it can lead to training collapse in thinking models for reasoning tasks like math and code. It seems to stem from the fact that while the PI conditioned model can reach the correct answer, the PI in its context makes its behavior different from a strong teacher model with no PI. Concretely, the PI makes the self-teacher increasingly confident which causes it to penalizing uncertainty verbalization and exploration, essential for thinking models [cite].

In order to somewhat quantify this, lets try to analyze the behavior of the self-teacher under different PIs. Assuming our data has expert demonstrations (from a larger LLM), we will analyze three simple choices of PI : full solution, final answer, and a self generated hint from the full solution. We will also compare this with the teacher behavior in on-policy distillation from a stronger teacher. What are some useful behaviors we can analyze? 

- Number of generated tokens
- Uncertainty verbalization
- Credit assignment
- Backtracking, verification, backward-chaining, subgoal setting
- Underthinking


Now lets step back and think about what we want self-distillation to do. Here we will only think about intuitions and leave rigorous arguments for later. We want several things : 

- Granular credit assignment
- Low bias
- On-policy training

We can always ensure on-policyness by only training on student generated trajectories. Its the other two properties that are harder to satisfy *together*. To ensure low bias we want to rely as much as possible on verifier signal. If we restrict ourselves to verifiable domains (math, code) for now, we have access to this in the form of the final answer or compiler output. GRPO fully relies on the verifier and foregoes credit assignment altogether which leads to the sample inefficiency of GRPO, especially for long horizon tasks. Intuitively, **we want to leverage the model's ICL to distribute the verifier signal at the end, across the trajectory**. This is what naive self-distillation attempts to do but simply adding the signal to the self-teacher context as PI. 

Given only the model and the environment, we only have the environment signal and the model's ICL. There's no other source of information we have access to. As long as we rely on ICL, we will be adding *some* bias since we are relying on the model's own interpretation of the verifier signal. But perhaps we can control that in a principled way with our formulation. 




[^critique]: The self-teacher is trying to solve the problem from each student rollout token position. Given this objective, its next-token distribution over that token position is the "critique" of that token.

