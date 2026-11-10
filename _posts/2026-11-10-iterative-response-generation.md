---
title: 'Iterative response generation'
date: 2026-11-10
permalink: /posts/2026-11-10-iterative-response-generation/
---

<figure>
  <video autoplay controls loop muted playsinline preload="metadata"
         poster="{{ '/assets/videos/iterative-response-generation.png' | relative_url }}"
         aria-label="Animation comparing standard and iterative response generation"
         aria-describedby="response-generation-caption"
         width="1600" height="1000" style="width: 100%; height: auto;">
    <source src="{{ '/assets/videos/iterative-response-generation.mp4' | relative_url }}" type="video/mp4">
    <a href="{{ '/assets/videos/iterative-response-generation.mp4' | relative_url }}">Watch the response generation animation.</a>
  </video>
  <figcaption id="response-generation-caption">
    Standard generation writes the answer in one forward pass after reasoning.
    Iterative generation adds and revises text during reasoning, syncing the current
    document back into context. The final document is the response.
    Text and timing are illustrative.
  </figcaption>
</figure>
