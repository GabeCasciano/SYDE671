---
title: "Assignment 1: Cameras and the Prokudin-Gorskii Collection"
date: 2026-09-28
excerpt: "Perspective, focal length, and the dolly zoom. Aligning and colorizing Prokudin-Gorskii glass plate images."
---

<!--
Plain Markdown only (no Liquid), so this file also renders outside the site.
Media goes in website/images/assignment-1/ and is referenced as ../images/assignment-1/<file>.

Image:
![Close-up selfie](../images/assignment-1/selfie-close.jpg)

Image with caption:
<figure>
  <img src="../images/assignment-1/selfie-far.jpg" alt="Zoomed selfie">
  <figcaption>Taken from about 3 m with zoom.</figcaption>
</figure>

Side by side (columns adapt to the number of figures):
<div class="media-grid">
  <figure><img src="../images/assignment-1/selfie-close.jpg" alt=""><figcaption>Close, wide</figcaption></figure>
  <figure><img src="../images/assignment-1/selfie-far.jpg" alt=""><figcaption>Far, zoomed</figcaption></figure>
</div>

GIF: same as an image. Video (the poster frame is what prints to PDF):
<video src="../images/assignment-1/dolly.mp4" poster="../images/assignment-1/dolly.jpg" controls loop muted playsinline></video>

Math: $$ \text{NCC} = \frac{a}{\|a\|} \cdot \frac{b}{\|b\|} $$
-->

## Part 1: Becoming Friends with Your Camera

### Selfie: The Wrong Way vs. The Right Way

<!-- Close-up selfie vs. stepped back and zoomed in, face the same size. Explain why the second looks better. -->

### Architectural Perspective Compression

<!-- Zoomed-in street view vs. walked closer without zoom. Explain the flattening. -->

### The Dolly Zoom

<!-- 4-8+ stills moving back while zooming in, combined into a GIF. -->

## Part 2: Colorizing the Prokudin-Gorskii Photo Collection

### Approach

<!-- Split the plate into B, G, R thirds and align G and R to B. -->

### Single-Scale Alignment

<!-- Exhaustive search over a displacement window, metric used (L2 / NCC). -->

### Image Pyramid

<!-- Coarse-to-fine search for the full-size plates. -->

### Results

| Image | G offset (x, y) | R offset (x, y) |
|-------|-----------------|-----------------|
|       |                 |                 |

### Bells & Whistles

<!-- Any extensions: cropping, contrast, white balance, better features, etc. -->
