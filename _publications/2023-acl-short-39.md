---
title: "Zero-shot Cross-lingual Transfer With Learned Projections Using Unlabeled Target-Language Data"
collection: publications
category: conferences
permalink: /publication/2023-acl-short-39
excerpt: 'An adapter based framework to improve cross lingual transfer when target languages are known at training time.'
date: 2023-07-01
venue: 'Association for Computational Linguistics'
paperurl: 'https://aclanthology.org/2023.acl-short.39/'
---

Adapters have emerged as a parameter-efficient Transformer-based framework for cross-lingual transfer by inserting lightweight language-specific modules (language adapters) and task-specific modules (task adapters) within pretrained multilingual models. Zero-shot transfer is enabled by pairing the language adapter in the target language with an appropriate task adapter in a source language. If our target languages are known apriori, we explore how zero-shot transfer can be further improved within the adapter framework by utilizing unlabeled text during task-specific finetuning. We construct language-specific subspaces using standard linear algebra constructs and selectively project source-language representations into the target language subspace during task-specific finetuning using two schemes. Our experiments on three cross-lingual tasks, Named Entity Recognition (NER), Question Answering (QA) and Natural Language Inference (NLI) yield consistent benefits compared to adapter baselines over a wide variety of target languages with up to 11% relative improvement in NER, 2% relative improvement in QA and 5% relative improvement in NLI.