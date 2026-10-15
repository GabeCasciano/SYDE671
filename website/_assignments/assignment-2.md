---
title: "Assignment 2: Fun with Filters and Frequencies"
date: 2026-10-14
excerpt: "Convolution, edge detection, unsharp masking, hybrid images, and multiresolution blending."
---

<!--
Plain Markdown only (no Liquid), so this file also renders outside the site.
Media goes in website/images/assignment-2/ and is referenced as ../images/assignment-2/<file>.
See assignment-1.md for figure, grid, video, and math snippets.
-->

## Part 1: Fun with Filters

### 1.1 Convolutions from Scratch

Both implementations flip the kernel and zero pad the image so the output is the same size as the input, matching `scipy.signal.convolve2d(img, kernel, mode="same", boundary="fill")`. Code: `Assignment 2/src/conv.py`.

```python
def pad(img, kernel):
    kh, kw = kernel.shape
    return np.pad(img, ((kh // 2, (kh - 1) // 2), (kw // 2, (kw - 1) // 2)))


def conv4(img, kernel):
    k = kernel[::-1, ::-1]
    kh, kw = k.shape
    p = pad(img, k)
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            total = 0.0
            for u in range(kh):
                for v in range(kw):
                    total += p[i + u, j + v] * k[u, v]
            out[i, j] = total
    return out


def conv2(img, kernel):
    k = kernel[::-1, ::-1]
    kh, kw = k.shape
    p = pad(img, k)
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.sum(p[i:i + kh, j:j + kw] * k)
    return out
```

#### 9x9 box filter

The box filter is `np.ones((9, 9)) / 81`, applied to a 635 x 900 grayscale image with each method.

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/1.1/gray.jpg" alt="Grayscale input"><figcaption>Input (grayscale)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.1/box_four_loops.jpg" alt="Box, four loops"><figcaption>Four loops</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.1/box_two_loops.jpg" alt="Box, two loops"><figcaption>Two loops</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.1/box_scipy.jpg" alt="Box, scipy"><figcaption>scipy convolve2d</figcaption></figure>
</div>

| Method | Runtime (s) | Max abs difference from scipy |
|--------|-------------|-------------------------------|
| Four loops | 21.19 | 2.98e-08 |
| Two loops | 2.98 | 2.98e-08 |
| `scipy.signal.convolve2d` | 0.10 | - |

<!-- Analysis (yours): Why is each method faster than the last (Python loop overhead vs vectorized numpy vs compiled C)? How do they handle boundaries: what does zero padding do to the edges of the box-filtered image, and what other boundary options does scipy offer? Why does the difference come out around 3e-8 rather than exactly 0? -->

#### Finite difference operators

$$ D_x = \begin{bmatrix} 1 & -1 \end{bmatrix}, \quad D_y = \begin{bmatrix} 1 \\ -1 \end{bmatrix} $$

Computed with the two-loop convolution. Mid grey is zero.

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/1.1/dx.jpg" alt="Dx"><figcaption>Image * Dx</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.1/dy.jpg" alt="Dy"><figcaption>Image * Dy</figcaption></figure>
</div>

<!-- Analysis (yours): Which edges show up in Dx and which in Dy, and why? What does the sign (light vs dark side) tell you about the edge? -->

### 1.2 Finite Difference Operator

### 1.3 Derivative of Gaussian (DoG) Filter

### Bells & Whistles: Gradient Orientation

## Part 2: Fun with Frequencies

### 2.1 Image Sharpening

### 2.2 Hybrid Images

### Bells & Whistles: Hybrid Images

### 2.3 Gaussian and Laplacian Stacks

### 2.4 Multiresolution Blending

### Bells & Whistles: Blending
