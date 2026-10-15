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

Every entry of the 9x9 box filter is 1/81, so each output pixel is the mean of the 9x9 neighbourhood around it:

$$ B = \frac{1}{81} \begin{bmatrix} 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \end{bmatrix} $$

The filter is applied to a 635 x 900 grayscale picture of me with each of the three methods:

```python
img = load_image(DATA_DIR / "pasta-boy.jpg", gray=True)
box = np.ones((9, 9)) / 81

out_four = conv4(img, box)
out_two = conv2(img, box)
out_scipy = convolve2d(img, box, mode="same", boundary="fill")
```

<div class="media-grid media-grid--2x2">
  <figure><img loading="lazy" src="../images/assignment-2/1.1/gray.jpg" alt="Grayscale input"><figcaption>Input (grayscale)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.1/box_four_loops.jpg" alt="Box, four loops"><figcaption>Four loops</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.1/box_two_loops.jpg" alt="Box, two loops"><figcaption>Two loops</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.1/box_scipy.jpg" alt="Box, scipy"><figcaption>scipy convolve2d</figcaption></figure>
</div>

| Method | Runtime (s) | Max abs difference from scipy |
|--------|-------------|-------------------------------|
| Four loops | 20.70 | 2.98e-08 |
| Two loops | 2.98 | 2.98e-08 |
| `scipy.signal.convolve2d` | 0.10 | - |

<!-- Analysis (yours): Why is each method faster than the last (Python loop overhead vs vectorized numpy vs compiled C)? How do they handle boundaries: what does zero padding do to the edges of the box-filtered image, and what other boundary options does scipy offer? Why does the difference come out around 3e-8 rather than exactly 0? -->

#### Finite difference operators

$$ D_x = \begin{bmatrix} 1 & 0 & -1 \end{bmatrix}, \quad D_y = \begin{bmatrix} 1 \\ 0 \\ -1 \end{bmatrix} $$

The same central difference along the two diagonals:

$$ D_{45} = \begin{bmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ -1 & 0 & 0 \end{bmatrix}, \quad D_{135} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & -1 \end{bmatrix} $$

Computed with the two-loop convolution. Mid grey is zero.

<div class="media-grid media-grid--2x2">
  <figure><img loading="lazy" src="../images/assignment-2/1.1/dx.jpg" alt="Dx"><figcaption>Image * Dx</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.1/dy.jpg" alt="Dy"><figcaption>Image * Dy</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.1/d45.jpg" alt="D45"><figcaption>Image * D45</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.1/d135.jpg" alt="D135"><figcaption>Image * D135</figcaption></figure>
</div>

<!-- Analysis (yours): Which edges show up in Dx and which in Dy, and why? Which edges do the diagonal operators pick up that Dx and Dy miss or weaken? What does the sign (light vs dark side) tell you about the edge? -->

### 1.2 Finite Difference Operator

The cameraman image is convolved with $$ D_x $$ and $$ D_y $$ using `scipy.signal.convolve2d`, and the gradient magnitude is computed from the two partial derivatives. The image borders are reflected (`boundary="symm"`) rather than zero padded. The edge image is the gradient magnitude binarized at a threshold. Code: `Assignment 2/src/finite_diff.py`.

$$ \lVert \nabla f \rVert = \sqrt{\left(\frac{\partial f}{\partial x}\right)^2 + \left(\frac{\partial f}{\partial y}\right)^2} $$

```python
dx = convolve2d(img, DX, mode="same", boundary="symm")
dy = convolve2d(img, DY, mode="same", boundary="symm")
mag = np.sqrt(dx ** 2 + dy ** 2)
edges = mag > threshold
```

<div class="media-grid media-grid--2x2">
  <figure><img loading="lazy" src="../images/assignment-2/1.2/gray.jpg" alt="Cameraman"><figcaption>Input</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.2/dx.png" alt="df/dx"><figcaption>df/dx</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.2/dy.png" alt="df/dy"><figcaption>df/dy</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.2/mag.png" alt="Gradient magnitude"><figcaption>Gradient magnitude</figcaption></figure>
</div>

#### Threshold progression

The gradient magnitude ranges from 0 to 0.86. The median is 0.012 and the 99th percentile is 0.39.

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/1.2/edges_0.05.png" alt="t = 0.05"><figcaption>t = 0.05</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.2/edges_0.1.png" alt="t = 0.10"><figcaption>t = 0.10</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.2/edges_0.15.png" alt="t = 0.15"><figcaption>t = 0.15</figcaption></figure>
</div>
<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/1.2/edges_0.2.png" alt="t = 0.20"><figcaption>t = 0.20</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.2/edges_0.25.png" alt="t = 0.25"><figcaption>t = 0.25</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.2/edges_0.3.png" alt="t = 0.30"><figcaption>t = 0.30</figcaption></figure>
</div>

#### Chosen threshold: t = 0.20

<figure>
  <img loading="lazy" src="../images/assignment-2/1.2/edges_0.2.png" alt="Edges at t = 0.20">
  <figcaption>Binarized gradient magnitude, t = 0.20</figcaption>
</figure>

<!-- Analysis (yours): Where does the noise come from at low thresholds (grass, sky)? Which real edges disappear first as the threshold rises (buildings in the background, tripod legs)? Which threshold is the best tradeoff, and why? -->

### 1.3 Derivative of Gaussian (DoG) Filter

The 2D Gaussian is the outer product of a 1D kernel from `cv2.getGaussianKernel`, with $$ \sigma = 2 $$ and a kernel size of $$ 6\sigma + 1 = 13 $$. Code: `Assignment 2/src/dog.py`.

```python
def gaussian_2d(sigma):
    g = cv2.getGaussianKernel(int(6 * sigma) | 1, sigma)
    return g @ g.T
```

#### Blur, then finite differences

The cameraman image is blurred with the Gaussian, and then the 1.2 steps are repeated on the blurred image.

```python
blurred = convolve2d(img, G, mode="same", boundary="symm")
dx = convolve2d(blurred, DX, mode="same", boundary="symm")
dy = convolve2d(blurred, DY, mode="same", boundary="symm")
mag = np.sqrt(dx ** 2 + dy ** 2)
```

<div class="media-grid media-grid--2x2">
  <figure><img loading="lazy" src="../images/assignment-2/1.3/blurred.jpg" alt="Blurred"><figcaption>Blurred (sigma = 2)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.3/dx.png" alt="Blurred df/dx"><figcaption>df/dx of blurred</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.3/dy.png" alt="Blurred df/dy"><figcaption>df/dy of blurred</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.3/mag.png" alt="Blurred gradient magnitude"><figcaption>Gradient magnitude of blurred</figcaption></figure>
</div>

The gradient magnitude of the blurred image goes up to 0.32 (it was 0.86 without blur). The median is 0.005 and the 99th percentile is 0.20, so the thresholds below are lower than in 1.2.

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/1.3/edges_0.02.png" alt="t = 0.02"><figcaption>t = 0.02</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.3/edges_0.04.png" alt="t = 0.04"><figcaption>t = 0.04</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.3/edges_0.06.png" alt="t = 0.06"><figcaption>t = 0.06</figcaption></figure>
</div>
<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/1.3/edges_0.08.png" alt="t = 0.08"><figcaption>t = 0.08</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.3/edges_0.1.png" alt="t = 0.10"><figcaption>t = 0.10</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.3/edges_0.12.png" alt="t = 0.12"><figcaption>t = 0.12</figcaption></figure>
</div>

#### Comparison with the finite difference method

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/1.2/edges_0.2.png" alt="Finite difference, t = 0.20"><figcaption>Finite difference only (1.2), t = 0.20</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.3/edges_0.06.png" alt="Gaussian then finite difference, t = 0.06"><figcaption>Gaussian, then finite difference, t = 0.06</figcaption></figure>
</div>

<!-- Analysis (yours): What differences do you see? Look at the grass, edge thickness, and how continuous the tripod legs and skyline are. Why does blurring first suppress the noise, and what does it cost (localization, fine detail)? Why does the magnitude range shrink? -->

#### DoG filters

Convolving the Gaussian with $$ D_x $$ and $$ D_y $$ gives the derivative of Gaussian filters, so the blur and the derivative become one convolution:

$$ (I * G) * D_x = I * (G * D_x) $$

```python
dog_x = convolve2d(G, DX)
dog_y = convolve2d(G, DY)
dx = convolve2d(img, dog_x, mode="same", boundary="symm")
dy = convolve2d(img, dog_y, mode="same", boundary="symm")
```

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/1.3/gaussian.png" alt="Gaussian kernel"><figcaption>Gaussian G (13 x 13)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.3/dog_x.png" alt="DoG x"><figcaption>G * Dx (13 x 15)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.3/dog_y.png" alt="DoG y"><figcaption>G * Dy (15 x 13)</figcaption></figure>
</div>

Applying the DoG filters directly to the image:

<div class="media-grid media-grid--2x2">
  <figure><img loading="lazy" src="../images/assignment-2/1.3/dog_dx.png" alt="DoG x response"><figcaption>Image * (G * Dx)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.3/dog_dy.png" alt="DoG y response"><figcaption>Image * (G * Dy)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.3/dog_mag.png" alt="DoG gradient magnitude"><figcaption>Gradient magnitude (DoG)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/1.3/diff.png" alt="Difference"><figcaption>|two-step - DoG| on the magnitude scale</figcaption></figure>
</div>

The largest difference between the two-step and DoG gradient magnitudes is 1.65e-15.

<!-- Analysis (yours): Do you get the same result as before? Why is the difference ~1e-15 and not exactly 0? Why is one DoG convolution preferable to two convolutions? -->

### Bells & Whistles: Gradient Orientation

The orientation comes from the DoG gradients in 1.3 ($$ \sigma = 2 $$). $$ g_y $$ is negated so that positive y points up, which makes angles count counter-clockwise from the +x axis as on a unit circle. No built-in angle function (`np.arctan2`, `np.angle`) is used. The angle is the arctangent of the slope $$ g_y / g_x $$, which only covers $$ (-\pi/2, \pi/2) $$, so $$ \pi $$ is added wherever $$ g_x < 0 $$ to recover the full circle. Code: `Assignment 2/src/orientation.py`.

$$ \theta = \begin{cases} \arctan(g_y / g_x) & g_x > 0 \\ \arctan(g_y / g_x) + \pi & g_x < 0 \\ \pm \pi / 2 & g_x = 0 \end{cases} \pmod{2\pi} $$

```python
def orientation(gx, gy):
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.arctan(gy / gx)
    theta = np.where(gx < 0, theta + np.pi, theta)
    theta = np.nan_to_num(theta)
    return np.mod(theta, 2 * np.pi)
```

The orientation sets the hue and the gradient magnitude sets the brightness, so flat regions go dark. Hue is periodic, which means 0 and $$ 2\pi $$ get the same colour. The magnitude is scaled by its 99th percentile and clipped to 1. In the "edges only" images, the brightness is the binary edge map from 1.3 (t = 0.06), so every edge pixel is shown at full brightness.

```python
hsv = np.dstack([theta / (2 * np.pi), np.ones_like(theta), value])
rgb = hsv_to_rgb(hsv)                                        # HSV colour space
rgb = colormaps["twilight"](theta / (2 * np.pi))[..., :3] * value[..., None]  # cyclic colormap
```

Each wheel is the legend: angle is the gradient direction, and the distance from the centre is the magnitude.

#### HSV

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/bw1/hsv.png" alt="HSV orientation"><figcaption>Brightness = magnitude</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/bw1/hsv_edges.png" alt="HSV orientation, edges"><figcaption>Edges only</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/bw1/wheel_hsv.png" alt="HSV wheel"><figcaption>Legend</figcaption></figure>
</div>

#### Matplotlib `twilight` (cyclic)

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/bw1/twilight.png" alt="Twilight orientation"><figcaption>Brightness = magnitude</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/bw1/twilight_edges.png" alt="Twilight orientation, edges"><figcaption>Edges only</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/bw1/wheel_twilight.png" alt="Twilight wheel"><figcaption>Legend</figcaption></figure>
</div>

<!-- Analysis (yours): Explain the process in your own words. Pick an edge, e.g. the left side of the coat or a tripod leg: does its colour match the direction from dark to bright on the wheel? Why do the two sides of a thin line (tripod leg) get opposite colours? Why does the colormap need to be cyclic? Which visualization reads better, HSV or twilight, and why? -->

## Part 2: Fun with Frequencies

### 2.1 Image Sharpening

### 2.2 Hybrid Images

### Bells & Whistles: Hybrid Images

### 2.3 Gaussian and Laplacian Stacks

### 2.4 Multiresolution Blending

### Bells & Whistles: Blending
