---
title: 'Self Distillation'
date: 2026-07-01
permalink: /posts/2026-07-01-self-distillation-post-1./
---

## Introduction

Policy gradient based post-training has become an important driver of rapidly improving LLM capabilities, particularly in verifiable domains such as math and code [[1]](#ref-1). InstructGPT [[2]](#ref-2) was among the first major success stories which used the PPO algorithm [[3]](#ref-3) with a learned reward model to train a language model towards human preferences. While powerful under carefully tuned hyperparameters, PPO comes with its own challenges. PPO's actor-critic formulation requires training a value function along with the model, which is usually another LLM initialized from a reward model. This adds significant computational overhead and training instability at frontier scale. Initializing the value function from a reward model has its own problems [[5]](#ref-5). Shao et al. introduced GRPO to tackle the memory and training instabilities of PPO at scale.[[4]](#ref-4). GRPO eliminates the value function entirely. To compute the advantage, GRPO instead samples a group of $$G$$ responses to the same prompt and uses their relative rewards as a baseline. Assuming outcome rewards $$r_i$$ the advantage becomes : 

$$
\hat{A}_i = \frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}{\operatorname{std}(r_1,\ldots,r_G)},
\tag{1}
$$

GRPO uses a PPO style clipped objective function with a KL anchor on the reference policy. More recent GRPO variants have done away with the KL term [[6]](#ref-6). Ignoring the clipping and KL penalty, the policy-gradient term has the same well known policy gradient form : 

$$
\nabla_\theta J_{\mathrm{GRPO}}(\theta)
\approx \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_i}\sum_{t=1}^{T_i}
\rho_{i,t}(\theta)\hat{A}_i\nabla_\theta
\log \pi_\theta(o_{i,t}\mid q,o_{i,<t})\right].
\tag{2}
$$

where $$\rho_{i,t}(\theta)=\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})}$$ is the importance sampling ratio (useful even in a purely on-policy setting).

GRPO has proved to be quite successful for post-training LLMs in verifiable domains [[1]](#ref-1) [[4]](#ref-4). Over the last couple of years it has spawned numerous variants that tackle its different amendable shortcomings [[6]](#ref-6) [[7]](#ref-7) [[8]](#ref-8). *Halving* the memory overhead and alleviating training instabilities of PPO, that become more and more acute at frontier scale, is naturally quite appealing. GRPO however is fundamentally incapable of a particularly desirable attribute of PPO : granular credit assignment per rollout.

<figure>
  <img src="/images/ppo-grpo-credit.svg" alt="Comparison of PPO token-level credit assignment using a value function and GAE with GRPO's uniform group-relative credit assignment." style="width: auto; height: auto; margin-left: auto; margin-right: auto;">
  <figcaption style="width: 100%;"><strong>Figure 1.</strong> Difference in credit assignment in PPO and GRPO : token level vs uniform.</figcaption>
</figure>

Most state-of-the-art LLMs are now thinking models. They generate reasoning traces that can be thousands or even tens of thousands of tokens long before producing a user-facing answer or solution [[1]](#ref-1) [[9]](#ref-9). GRPO's group level advantage estimation assigns uniform credit to all such tokens. With enough rollouts and training steps (i.e. enough compute), models are able to successfully localize credit which explains the success of GRPO. One can also argue that if compute is not particularly a constraint, sampling task advantage directly is more desirable over PPO's biased value function. 

Compute however is a constraint outside of frontier labs, and perhaps it should be. We would like to be able to train reasoning models in as compute-efficient a manner as possible. Algorithms that are able to perform *accurate* granular credit assignment at a similar rollout budget should intuitively be much more sample and compute efficient than GRPO. If such an algorithm scales well, then that would also be superior to group level advantage estimation at frontier scale. Granular credit assignment is also something humans naturally attempt to do, often very successfully. Therefore we should aspire to mitigate the significant shortcomings of PPO like policy gradient methods or attempt to assign token level credit in other ways.

**On-policy distillation** (OPD) offers a convenient way of assigning granular credit to rollouts *when we have a access to a much stronger model* [[10]](#ref-10) [[11]](#ref-11) [[12]](#ref-12). As the name suggests, it computes the teacher's next-token distribution at every prefix of the *student's* rollouts. An *f*-divergence between the student and teacher distributions then becomes the token level supervision or 'credit'. This formulation also has the advantage of not needing a verifier or a reward model and can be readily applied to non-verifiable domains, assuming the teacher has enough domain expertise. This has proven to be extremely useful for distilling a large post-trained model into smaller model of the same family [[10]](#ref-10) [[11]](#ref-11) [[15]](#ref-15). The goal is obviously to do such credit assignment without first training a large model with group level advantage estimation. However its still worth looking into OPD carefully as it offers *a* surrogate to study and analyze how our desirable algorithm should behave. 

## What is self-distillation

On-policy self-distillation (OPSD) builds upon OPD by attempting to construct the teacher from the student model itself [[16]](#ref-16) [[17]](#ref-17) [[20]](#ref-20) [[21]](#ref-21). The key idea is that a student might be able to critique[^critique] its own rollouts if given access to privileged information (PI) in its context; PI that enables the self-teacher to exceed the performance of the student. In addition to token level feedback, OPSD also preserves OPD's desirable property of not needing an explicit verifier. The objective can be written as :

$$
\mathcal{L}(\theta)=D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot\mid q)\,\|\,\pi(\cdot\mid q,c)\right)
=\mathbb{E}_{o\sim\pi_\theta(\cdot\mid q)}\!\left[\log\frac{\pi_\theta(o\mid q)}{\pi(o\mid q,c)}\right],
\tag{3}
$$

where $$c$$ is the PI. The teacher can be frozen or an exponential moving average of the student. Differentiating this w.r.t $$\theta$$ again results in the familiar policy gradient expression : 

$$
\nabla_\theta \mathcal{L}(\theta)=\mathbb{E}_{o\sim\pi_\theta}\!\left[\sum_{t=1}^{T}
\log\frac{\pi_\theta(o_t\mid q,o_{<t})}{\pi(o_t\mid q,c,o_{<t})}\,
\nabla_\theta\log\pi_\theta(o_t\mid q,o_{<t})\right].
\tag{4}
$$

Comparing equations (2) and (4) allows us to write out the OPSD advantage : 

$$
\hat{A}_t = \log\frac{\pi(o_t\mid q,c,o_{<t})}{\pi_\theta(o_t\mid q,o_{<t})},
\tag{5}
$$

The OPSD objective is minimized whereas the GRPO objective is maximized (expected sum of rewards). Therefore the advantage sign needs to be flipped which is the same as flipping the ratio. Since the advantage is *per token* it also has a $$t$$ subscript, unlike the GRPO advantage.


Why should some choice of PI enable this self-teacher construction? This is because the student in this case is an instruction tuned LLM with in-context learning abilities (ICL) [[18]](#ref-18) [[19]](#ref-19). Having the PI in context, demonstrably increases task performance of the self-teacher beyond the student [[21]](#ref-21). Its easy to be convinced by this. For example, an instruction tuned LLM should be able to trivially solve a problem when the entire solution (PI) is present in its context. It only needs to have the minimal intelligence to recognize the solution and copy it. This also leads to the interesting question of how should one choose the PI. The choice of PI is crucial as we'll see but the formulation does not impose strong constraints on what the PI should be.

## Analyzing self-distillation

Does OPSD work? It seems to work well for a range of tasks and domains [[16]](#ref-16) [[20]](#ref-20) [[21]](#ref-21) [[22]](#ref-22). However it seems to have a particularly important and interesting failure mode which will be the subject of our discussion here. OPSD can lead to training collapse in thinking models for reasoning tasks like math [[23]](#ref-23) [[24]](#ref-24) [[25]](#ref-25). It seems that while the self-teacher can produce the correct solution (trivially, if the PI *is* the solution), the PI makes the self-teacher increasingly confident, causing it to penalize the student's exploration, essential for thinking models [[23]](#ref-23) [[24]](#ref-24) [[25]](#ref-25). 

Its hard to precisely define exploration of an LLM's chain-of-thought (COT) but in our context it loosely refers to the model exploring different strategies or related concepts, expressing uncertainty, backtracking and correcting itself before committing to a final solution. Kim et. al [[25]](#ref-25) report that the self-teacher generates fewer expressions of uncertainty ('wait', 'hmm', 'perhaps', etc) as the PI becomes more informative. To replicate this finding we take the Qwen3-1.7B and the Qwen3-4B models [[27]](#ref-27) and evaluate their uncertainty verbalization and pass@k metrics under different PIs, on the Deepmath [[26]](#ref-26) training dataset [^why-training] : 

<figure>
  <img src="/images/pi_uncertainty_tradeoff_linear_inverted.png" alt="Plot of pass@8 accuracy against mean chain-of-thought epistemic marker count for Qwen3-1.7B and Qwen3-4B on DeepMath, under five privileged information conditions: none, hint, answer, rollout and full." style="max-width: 100%; height: auto; margin-left: auto; margin-right: auto;">
  <figcaption style="width: 100%;" markdown="span"><strong>Figure 2.</strong> Verbalized uncertainty against pass@8 accuracy for Qwen3-1.7B and Qwen3-4B on 128 DeepMath problems[^dropped-problem] under different PIs -> none: no PI, hint: a self generated hint from a correct demonstration, answer: the correct final answer, rollout: a self generated rollout which may be correct or incorrect, full: a full demonstration. </figcaption>
</figure>

Both thinking models exhibit above 80% pass@8 accuracy with over 70 epistemic tokens on average in the think traces. As we make the PI progressively more informative about the problem, uncertainty verbalization decreases with overall increase in pass@8 accuracy. 




[^critique]: The self-teacher is trying to solve the problem from each student prefix. Given this objective, its next-token distribution over that token position is the 'critique' of that token.

[^why-training]: We primarily care about how the self-teacher behaves on training data. The self-teacher is absent at test-time.

[^dropped-problem]: 1 problem is dropped for Qwen3-4B because the full PI exceeds 8192 max tokens.





## References

1. <a id="ref-1"></a>DeepSeek-AI et al. (2025). [*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*](https://arxiv.org/abs/2501.12948). arXiv:2501.12948.
2. <a id="ref-2"></a>Ouyang, L., Wu, J., Jiang, X., et al. (2022). [*Training Language Models to Follow Instructions with Human Feedback*](https://arxiv.org/abs/2203.02155). Advances in Neural Information Processing Systems, 35.
3. <a id="ref-3"></a>Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347). arXiv:1707.06347.
4. <a id="ref-4"></a>Shao, Z., Wang, P., Zhu, Q., et al. (2024). [*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*](https://arxiv.org/abs/2402.03300). arXiv:2402.03300.
5. <a id="ref-5"></a>Yuan, Y., Yue, Y., Zhu, R., Fan, T., & Yan, L. (2025). [*What's Behind PPO's Collapse in Long-CoT? Value Optimization Holds the Secret*](https://arxiv.org/abs/2503.01491). arXiv:2503.01491.
6. <a id="ref-6"></a>Yu, Q., Zhang, Z., Zhu, R., et al. (2025). [*DAPO: An Open-Source LLM Reinforcement Learning System at Scale*](https://arxiv.org/abs/2503.14476). arXiv:2503.14476.
7. <a id="ref-7"></a>Liu, Z., Chen, C., Li, W., et al. (2025). [*Understanding R1-Zero-Like Training: A Critical Perspective*](https://arxiv.org/abs/2503.20783). arXiv:2503.20783.
8. <a id="ref-8"></a>Zheng, C., Liu, S., Li, M., et al. (2025). [*Group Sequence Policy Optimization*](https://arxiv.org/abs/2507.18071). arXiv:2507.18071.
9. <a id="ref-9"></a>Kimi Team et al. (2025). [*Kimi k1.5: Scaling Reinforcement Learning with LLMs*](https://arxiv.org/abs/2501.12599). arXiv:2501.12599.
10. <a id="ref-10"></a>Gu, Y., Dong, L., Wei, F., & Huang, M. (2024). [*MiniLLM: On-Policy Distillation of Large Language Models*](https://arxiv.org/abs/2306.08543). International Conference on Learning Representations.
11. <a id="ref-11"></a>Agarwal, R., Vieillard, N., Zhou, Y., et al. (2024). [*On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*](https://arxiv.org/abs/2306.13649). International Conference on Learning Representations.
12. <a id="ref-12"></a>Ko, J., Kim, S., Chen, T., & Yun, S. (2024). [*DistiLLM: Towards Streamlined Distillation for Large Language Models*](https://arxiv.org/abs/2402.03898). International Conference on Machine Learning.
13. <a id="ref-13"></a>Ye, T., Dong, L., Chi, Z., Wu, X., Huang, S., & Wei, F. (2025). [*Black-Box On-Policy Distillation of Large Language Models*](https://arxiv.org/abs/2511.10643). arXiv:2511.10643.
14. <a id="ref-14"></a>Ye, T., Dong, L., Wu, X., Huang, S., & Wei, F. (2026). [*On-Policy Context Distillation for Language Models*](https://arxiv.org/abs/2602.12275). arXiv:2602.12275.
15. <a id="ref-15"></a>Yang, W., Liu, W., Xie, R., Yang, K., Yang, S., & Lin, Y. (2026). [*Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation*](https://arxiv.org/abs/2602.12125). arXiv:2602.12125.
16. <a id="ref-16"></a>Zhao, S., Xie, Z., Liu, M., et al. (2026). [*Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models*](https://arxiv.org/abs/2601.18734). arXiv:2601.18734.
17. <a id="ref-17"></a>Penaloza, E., Vattikonda, D., Gontier, N., et al. (2026). [*Privileged Information Distillation for Language Models*](https://arxiv.org/abs/2602.04942). arXiv:2602.04942.
18. <a id="ref-18"></a>Brown, T. B., Mann, B., Ryder, N., et al. (2020). [*Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165). Advances in Neural Information Processing Systems, 33.
19. <a id="ref-19"></a>Min, S., Lyu, X., Holtzman, A., et al. (2022). [*Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?*](https://arxiv.org/abs/2202.12837). Conference on Empirical Methods in Natural Language Processing.
20. <a id="ref-20"></a>Hübotter, J., Lübeck, F., Behric, L., et al. (2026). [*Reinforcement Learning via Self-Distillation*](https://arxiv.org/abs/2601.20802). arXiv:2601.20802.
21. <a id="ref-21"></a>Shenfeld, I., Damani, M., Hübotter, J., & Agrawal, P. (2026). [*Self-Distillation Enables Continual Learning*](https://arxiv.org/abs/2601.19897). arXiv:2601.19897.
22. <a id="ref-22"></a>Rezaei, M., Mahmoud, A., Wang, Z., et al. (2026). [*Rubric-Guided Self-Distillation: Post-Training Without Rubric Verifiers*](https://arxiv.org/abs/2606.12507). arXiv:2606.12507.
23. <a id="ref-23"></a>Kaur, S., Ri, N., He, Y., Fowl, L., & Arora, S. (2026). [*Rethinking On-Policy Self-Distillation for Thinking Models*](https://arxiv.org/abs/2607.05184). arXiv:2607.05184.
24. <a id="ref-24"></a>Peng, K., Li, C., Ouyang, Y., Yuan, Y., & Ding, L. (2026). [*Diagnosing and Mitigating Thinking Collapse in On-Policy Self-Distillation*](https://arxiv.org/abs/2607.10805). arXiv:2607.10805.
25. <a id="ref-25"></a>Kim, J., Luo, X., Kim, M., et al. (2026). [*Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?*](https://arxiv.org/abs/2603.24472). arXiv:2603.24472.
26. <a id="ref-26"></a>He, Z., Liang, T., Xu, J., et al. (2025). [*DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset for Advancing Reasoning*](https://arxiv.org/abs/2504.11456). arXiv:2504.11456.
27. <a id="ref-27"></a>Yang, A., Li, A., Yang, B., et al. (2025). [*Qwen3 Technical Report*](https://arxiv.org/abs/2505.09388). arXiv:2505.09388.
