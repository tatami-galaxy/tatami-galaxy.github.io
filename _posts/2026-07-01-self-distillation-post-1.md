---
title: 'Self Distillation'
date: 2026-07-01
permalink: /posts/2026-07-01-self-distillation-post-1./
---

Policy gradient based post-training has become an important driver of rapidly improving LLM capabilities, particularly in verifiable domains such as mathematics and code [[1]](#ref-1). InstructGPT [[2]](#ref-2) was among the first major success stories which used the PPO algorithm [[3]](#ref-3) with a learned reward model to train a language model towards human preferences. While powerful under carefully tuned hyperparameters, PPO comes with its own challenges. PPO's actor-critic formulation requires training a value function along with the model, which is usually LLM initialized from a reward model. This adds significant computational overhead and training instability at frontier scale. Initializing the value function from a reward model has its own problems [[5]](#ref-5). Shao et al. introduced GRPO to tackle the memory and training instabilities of PPO at scale.[[4]](#ref-4). GRPO eliminates the value function entirely. To compute the advantage, GRPO instead samples a group of $G$ responses to the same prompt and uses their relative rewards as a baseline. A simplified form of its clipped surrogate objective is

$$
J_{\mathrm{GRPO}}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_i}\sum_{t=1}^{T_i}
\left(\min\!\left(\rho_{i,t}(\theta)\hat{A}_i,
\operatorname{clip}\!\left(\rho_{i,t}(\theta),1-\epsilon,1+\epsilon\right)\hat{A}_i\right)
- \beta D_{\mathrm{KL}}\!\left(\pi_\theta\,\|\,\pi_{\mathrm{ref}}\right)\right)\right],
$$

where $$\rho_{i,t}(\theta)=\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})}$$ is the policy probability ratio. With outcome-level rewards, the group-relative advantage is

$$
\hat{A}_i = \frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}{\operatorname{std}(r_1,\ldots,r_G)},
$$

Treating the sampled advantages as constants during the update, differentiating the surrogate objective gives the GRPO policy gradient

$$
\nabla_\theta J_{\mathrm{GRPO}}(\theta)
= \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_i}\sum_{t=1}^{T_i}
\left(
\nabla_\theta \min\!\left(\rho_{i,t}(\theta)\hat{A}_i,
\operatorname{clip}\!\left(\rho_{i,t}(\theta),1-\epsilon,1+\epsilon\right)\hat{A}_i\right)
- \beta\nabla_\theta D_{\mathrm{KL}}\!\left(\pi_\theta\,\|\,\pi_{\mathrm{ref}}\right)
\right)\right].
$$

For a token whose probability ratio is not clipped, the policy-gradient term reduces to the familiar score-function form

$$
\nabla_\theta J_{\mathrm{GRPO}}(\theta)
\approx \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_i}\sum_{t=1}^{T_i}
\rho_{i,t}(\theta)\hat{A}_i\nabla_\theta
\log \pi_\theta(o_{i,t}\mid q,o_{i,<t})
- \beta\nabla_\theta D_{\mathrm{KL}}\!\left(\pi_\theta\,\|\,\pi_{\mathrm{ref}}\right)\right].
$$

The same $\hat{A}_i$ is assigned to every token in response $i$. This removes the value model, but it also means that outcome-level GRPO does not distinguish between the helpful and unhelpful tokens within a successful response [[4]](#ref-4).

Self Distillation has recently come up as a promising direction for language model post training [cite]. It targets two major shortcomings of dominant RL based post training algorithms like GRPO [cite]. GRPO requires a verifier signal (ex : answer correctness) which might be hard to obtain for tasks where there isnt a clear notion of correctness. GRPO also has a credit assignment problem : for positive advantages, every token in the rollout is equally upweighted. The algorithm cannot assign granular credit to parts of the rollout, unlike its predecessor PPO, which suffers from its own inefficiencies and instabilities. 

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

## References

1. <a id="ref-1"></a>DeepSeek-AI et al. (2025). [*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*](https://arxiv.org/abs/2501.12948). arXiv:2501.12948.
2. <a id="ref-2"></a>Ouyang, L., Wu, J., Jiang, X., et al. (2022). [*Training Language Models to Follow Instructions with Human Feedback*](https://arxiv.org/abs/2203.02155). Advances in Neural Information Processing Systems, 35.
3. <a id="ref-3"></a>Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347). arXiv:1707.06347.
4. <a id="ref-4"></a>Shao, Z., Wang, P., Zhu, Q., et al. (2024). [*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*](https://arxiv.org/abs/2402.03300). arXiv:2402.03300.
5. <a id="ref-5"></a>Yuan, Y., Yue, Y., Zhu, R., Fan, T., & Yan, L. (2025). [*What's Behind PPO's Collapse in Long-CoT? Value Optimization Holds the Secret*](https://arxiv.org/abs/2503.01491). arXiv:2503.01491.
