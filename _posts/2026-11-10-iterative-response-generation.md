---
title: 'Iterative response generation'
date: 2026-11-10
permalink: /posts/2026-11-10-iterative-response-generation/
---

<figure>
  <video autoplay controls loop muted playsinline preload="metadata"
         poster="{{ '/assets/videos/iterative-response-generation.png' | relative_url }}"
         aria-label="Animation comparing standard and iterative response generation"
         width="1600" height="1000" style="width: 100%; height: auto;">
    <source src="{{ '/assets/videos/iterative-response-generation.mp4' | relative_url }}" type="video/mp4">
    <a href="{{ '/assets/videos/iterative-response-generation.mp4' | relative_url }}">Watch the response generation animation.</a>
  </video>
</figure>

Recently [context language models](https://arxiv.org/abs/2609.37725) proposed giving models a text file to freely edit which is synced to be its conversation history at each tool call. This allows the model to edit and preserve its context history as it chooses when it moves from one call to the next. 

Can we generalize this for response generation itself? We want the final response to be a text file which the model can iteratively build and edit as it reasons via its chain of thought. Intuitively, this allows the model to refine its final response freely without needing to commit to a fully structured final response at the end of its COT budget. 