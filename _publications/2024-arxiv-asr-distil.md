---
title: "AdaKD: Dynamic Knowledge Distillation of ASR models using Adaptive Loss Weighting"
collection: publications
category: preprints
permalink: /publication/2024-arxiv-asr-distil
excerpt: 'A curriculum learning based distillation method for transformer based speech recognition models.'
date: 2024-05-01
venue: 'arxiv'
paperurl: 'https://arxiv.org/abs/2405.08019'
---

Knowledge distillation, a widely used model compression technique, works on the basis of transferring knowledge from a cumbersome teacher model to a lightweight student model. The technique involves jointly optimizing the task specific and knowledge distillation losses with a weight assigned to them. Despite these weights playing a crucial role in the performance of the distillation process, current methods provide equal weight to both losses, leading to suboptimal performance. In this paper, we propose Adaptive Knowledge Distillation, a novel technique inspired by curriculum learning to adaptively weigh the losses at instance level. This technique goes by the notion that sample difficulty increases with teacher loss. Our method follows a plug-and-play paradigm that can be applied on top of any task-specific and distillation objectives. Experiments show that our method performs better than conventional knowledge distillation method and existing instance-level loss functions. 