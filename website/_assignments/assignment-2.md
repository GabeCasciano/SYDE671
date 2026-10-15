---
title: "Assignment 2: Fun with Filters and Frequencies"
date: 2026-10-14
excerpt: "Convolution, edge detection, unsharp masking, hybrid images, and multiresolution blending."
---

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

The difference image is shown on the same scale as the gradient magnitude images and comes out completely black. The largest difference between the two-step and DoG magnitudes is 1.65e-15, about 5e-15 of the brightest magnitude. That is far below one grey level (1/255), so every pixel rounds to 0.

### Bells & Whistles: Gradient Orientation

The orientation comes from the DoG gradients in 1.3 ($$ \sigma = 2 $$). $$ g_y $$ is negated so that positive y points up, which makes angles count counter-clockwise from the +x axis as on a unit circle. The angle is the arctangent of the slope $$ g_y / g_x $$, which only covers $$ (-\pi/2, \pi/2) $$, so $$ \pi $$ is added wherever $$ g_x < 0 $$ to recover the full circle. Code: `Assignment 2/src/orientation.py`.

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

## Part 2: Fun with Frequencies

### 2.1 Image Sharpening

The Gaussian $$ G $$ (from 1.3, $$ \sigma = 2 $$) is a low-pass filter, so subtracting the blurred image from the original leaves the high frequencies. Adding a scaled amount $$ \alpha $$ of those back to the original gives the sharpened image, and the whole operation collapses into a single kernel, the unsharp mask filter:

$$ f + \alpha (f - f * G) = f * \big( (1 + \alpha)\, e - \alpha G \big) $$

where $$ e $$ is the unit impulse (1 at the centre, 0 elsewhere). Colour images are filtered one channel at a time. Code: `Assignment 2/src/unsharp.py`.

```python
def unsharp_kernel(sigma, a):
    g = gaussian_2d(sigma)
    impulse = np.zeros_like(g)
    impulse[g.shape[0] // 2, g.shape[1] // 2] = 1
    return (1 + a) * impulse - a * g


sharpened = conv(img, unsharp_kernel(sigma, a))
```

#### Taj Mahal

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/2.1/taj.jpg" alt="Taj original"><figcaption>Original</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.1/taj_blurred.jpg" alt="Taj blurred"><figcaption>Blurred, f * G</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.1/taj_high.jpg" alt="Taj high frequencies"><figcaption>High frequencies, f - f * G</figcaption></figure>
</div>

<div class="media-grid media-grid--2x2">
  <figure><img loading="lazy" src="../images/assignment-2/2.1/taj_sharp_0.5.jpg" alt="alpha 0.5"><figcaption>Sharpened, alpha = 0.5</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.1/taj_sharp_1.jpg" alt="alpha 1"><figcaption>Sharpened, alpha = 1</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.1/taj_sharp_2.jpg" alt="alpha 2"><figcaption>Sharpened, alpha = 2</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.1/taj_sharp_4.jpg" alt="alpha 4"><figcaption>Sharpened, alpha = 4</figcaption></figure>
</div>

#### Crazy cat

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/2.1/crazy-cat.jpg" alt="Cat original"><figcaption>Original</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.1/crazy-cat_blurred.jpg" alt="Cat blurred"><figcaption>Blurred, f * G</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.1/crazy-cat_high.jpg" alt="Cat high frequencies"><figcaption>High frequencies, f - f * G</figcaption></figure>
</div>

<div class="media-grid media-grid--2x2">
  <figure><img loading="lazy" src="../images/assignment-2/2.1/crazy-cat_sharp_0.5.jpg" alt="alpha 0.5"><figcaption>Sharpened, alpha = 0.5</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.1/crazy-cat_sharp_1.jpg" alt="alpha 1"><figcaption>Sharpened, alpha = 1</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.1/crazy-cat_sharp_2.jpg" alt="alpha 2"><figcaption>Sharpened, alpha = 2</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.1/crazy-cat_sharp_4.jpg" alt="alpha 4"><figcaption>Sharpened, alpha = 4</figcaption></figure>
</div>

#### Blur, then re-sharpen

A sharp image is blurred with the same Gaussian and then sharpened again with the unsharp mask filter.

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/2.1/pasta-boy.jpg" alt="Original"><figcaption>Original (sharp)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.1/pasta-boy_blurred.jpg" alt="Blurred"><figcaption>Blurred, sigma = 2</figcaption></figure>
</div>

<div class="media-grid media-grid--2x2">
  <figure><img loading="lazy" src="../images/assignment-2/2.1/pasta-boy_resharp_0.5.jpg" alt="alpha 0.5"><figcaption>Re-sharpened, alpha = 0.5</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.1/pasta-boy_resharp_1.jpg" alt="alpha 1"><figcaption>Re-sharpened, alpha = 1</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.1/pasta-boy_resharp_2.jpg" alt="alpha 2"><figcaption>Re-sharpened, alpha = 2</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.1/pasta-boy_resharp_4.jpg" alt="alpha 4"><figcaption>Re-sharpened, alpha = 4</figcaption></figure>
</div>

The table gives the root mean square error between each image and the original (pixel values in [0, 1]):

| Image | RMSE vs original |
|-------|------------------|
| Blurred | 0.0334 |
| Re-sharpened, alpha = 0.5 | 0.0292 |
| Re-sharpened, alpha = 1 | 0.0260 |
| Re-sharpened, alpha = 2 | 0.0235 |
| Re-sharpened, alpha = 4 | 0.0336 |

### 2.2 Hybrid Images

A hybrid image adds the low frequencies of one image to the high frequencies of another (Oliva, Torralba, and Schyns, 2006). Up close the high frequencies dominate, and from far away only the low frequencies are visible. The code is a Python port of the MATLAB starter (`hybrid_image_starter.m` and `align_images.m`). Code: `Assignment 2/src/hybrid.py`.

```mermaid
flowchart TD
  A[Image 1<br/>Derek] --> C[Align<br/>2 points per image:<br/>translate, scale, rotate, crop]
  B[Image 2<br/>Nutmeg] --> C
  C --> D[Grayscale]
  D --> E[Low-pass image 1<br/>Gaussian, sigma_low]
  D --> F[High-pass image 2<br/>image - Gaussian, sigma_high]
  E --> G[Add]
  F --> G
  G --> H[Hybrid image]
  E --> I[Log magnitude FFT<br/>of inputs, filtered images, hybrid]
  F --> I
  H --> I
```

#### Alignment

Two points are clicked on each image (the eyes). `align_images` follows the MATLAB version step by step:

1. Pad each image so the midpoint of its two points is at the image centre.
2. Downscale the larger image so the distance between the points matches.
3. Rotate one image so the angle of the line between the points matches.
4. Crop both images to the same size.

The starter also has an optional manual crop at the end. Here it is done automatically instead: the borders are trimmed until neither image has any of the black fill left by the padding and rotation. The clicked points are saved, so the alignment is repeatable.

```python
def align_images(im1, im2, pts1, pts2):
    (x1, y1), (x2, y2) = pts1.T, pts2.T
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    im1 = pad_y(pad_x(im1, round((w1 / 2 - x1.mean()) * 2)), round((h1 / 2 - y1.mean()) * 2))
    im2 = pad_y(pad_x(im2, round((w2 / 2 - x2.mean()) * 2)), round((h2 / 2 - y2.mean()) * 2))

    len1 = np.hypot(x1[1] - x1[0], y1[1] - y1[0])
    len2 = np.hypot(x2[1] - x2[0], y2[1] - y2[0])
    dscale = len2 / len1
    if dscale < 1:
        im1 = resize(im1, dscale)
    else:
        im2 = resize(im2, 1 / dscale)

    theta1 = np.arctan2(-(y1[1] - y1[0]), x1[1] - x1[0])
    theta2 = np.arctan2(-(y2[1] - y2[0]), x2[1] - x2[0])
    im1 = rotate(im1, np.degrees(theta2 - theta1), reshape=True, order=1)
    ...  # crop both to the same height and width
```

<div class="media-grid media-grid--2x2">
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_im1.jpg" alt="Derek original"><figcaption>Derek (original)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_im2.jpg" alt="Nutmeg original"><figcaption>Nutmeg (original)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_im1_aligned.jpg" alt="Derek aligned"><figcaption>Derek (aligned, grayscale)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_im2_aligned.jpg" alt="Nutmeg aligned"><figcaption>Nutmeg (aligned, grayscale)</figcaption></figure>
</div>

#### Filtering

The low-pass filter is a Gaussian, and the high-pass filter is the impulse minus a Gaussian, computed by subtracting the blurred image from the original:

$$ H = I_1 * G_{\sigma_{low}} + \left( I_2 - I_2 * G_{\sigma_{high}} \right) $$

```python
def hybrid_image(im1, im2, sigma_low, sigma_high):
    low = gaussian_blur(im1, sigma_low)
    high = im2 - gaussian_blur(im2, sigma_high)
    return low, high, low + high
```

The cutoff of each filter is set by its $$ \sigma $$. The Fourier transform of a Gaussian is also a Gaussian, with a standard deviation of $$ 1 / (2\pi\sigma) $$ cycles per pixel. Its response falls to one half at

$$ f_c = \frac{\sqrt{2 \ln 2}}{2 \pi \sigma} \approx \frac{0.187}{\sigma} \text{ cycles per pixel} $$

The aligned images are 659 x 701, so the frequencies below are given in cycles per image (multiplied by 701).

#### Cutoff frequency

Rows change $$ \sigma_{low} $$ (Derek) and columns change $$ \sigma_{high} $$ (Nutmeg).

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_sweep_low4_high2.jpg" alt=""><figcaption>sigma_low 4 (33), sigma_high 2 (66)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_sweep_low4_high4.jpg" alt=""><figcaption>sigma_low 4 (33), sigma_high 4 (33)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_sweep_low4_high8.jpg" alt=""><figcaption>sigma_low 4 (33), sigma_high 8 (16)</figcaption></figure>
</div>
<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_sweep_low8_high2.jpg" alt=""><figcaption>sigma_low 8 (16), sigma_high 2 (66)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_sweep_low8_high4.jpg" alt=""><figcaption>sigma_low 8 (16), sigma_high 4 (33)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_sweep_low8_high8.jpg" alt=""><figcaption>sigma_low 8 (16), sigma_high 8 (16)</figcaption></figure>
</div>
<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_sweep_low16_high2.jpg" alt=""><figcaption>sigma_low 16 (8), sigma_high 2 (66)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_sweep_low16_high4.jpg" alt=""><figcaption>sigma_low 16 (8), sigma_high 4 (33)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_sweep_low16_high8.jpg" alt=""><figcaption>sigma_low 16 (8), sigma_high 8 (16)</figcaption></figure>
</div>

Cutoffs in brackets are in cycles per image.

#### Chosen result: sigma_low = 8, sigma_high = 4

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_low.jpg" alt="Low-pass Derek"><figcaption>Low-pass Derek</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_high.jpg" alt="High-pass Nutmeg"><figcaption>High-pass Nutmeg (mid grey is zero)</figcaption></figure>
</div>

<figure>
  <img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_hybrid.jpg" alt="Hybrid">
  <figcaption>Hybrid</figcaption>
</figure>

The same hybrid at 1/2, 1/4, and 1/8 size, which is like viewing it from further away:

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_hybrid_0.5.jpg" alt="1/2"><figcaption>1/2</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_hybrid_0.25.jpg" alt="1/4"><figcaption>1/4</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_hybrid_0.125.jpg" alt="1/8"><figcaption>1/8</figcaption></figure>
</div>

#### Frequency analysis

Log magnitude of the 2D Fourier transform, computed as `np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray_image))))`. All five spectra use the same grey scale (set from the two inputs), so brightness can be compared between them.

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_fft_im1.png" alt="FFT Derek"><figcaption>Derek</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_fft_im2.png" alt="FFT Nutmeg"><figcaption>Nutmeg</figcaption></figure>
</div>
<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_fft_low.png" alt="FFT low-pass"><figcaption>Low-pass Derek</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_fft_high.png" alt="FFT high-pass"><figcaption>High-pass Nutmeg</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_fft_hybrid.png" alt="FFT hybrid"><figcaption>Hybrid</figcaption></figure>
</div>

#### Cats

Two different cats, aligned on the eyes. Each photo was cropped around the cat's face before alignment, to a box 3.5 eye distances wide centred on the eyes, so the faces fill the frame and the two backgrounds take up less of it. Both arrangements are shown: each cat is used once as the low-frequency image and once as the high-frequency image. Both use $$ \sigma_{low} = 8 $$ and $$ \sigma_{high} = 4 $$.

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/2.2/bigcat_littlecat_im1.jpg" alt="Big cat"><figcaption>Big cat (low frequencies)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/bigcat_littlecat_im2.jpg" alt="Little cat"><figcaption>Little cat (high frequencies)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/bigcat_littlecat_hybrid.jpg" alt="Hybrid"><figcaption>Hybrid</figcaption></figure>
</div>

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/2.2/littlecat_bigcat_im1.jpg" alt="Little cat"><figcaption>Little cat (low frequencies)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/littlecat_bigcat_im2.jpg" alt="Big cat"><figcaption>Big cat (high frequencies)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/littlecat_bigcat_hybrid.jpg" alt="Hybrid"><figcaption>Hybrid</figcaption></figure>
</div>

Some background clutter is left in both hybrids. The high-pass filter keeps every fine detail in its image, not only the cat. The patterned pillow and the printed blanket are full of high-frequency texture, so they come through as faint patterns around the cat. The low-pass image adds its own background as large blurred patches of colour and brightness. The two backgrounds don't match, so they don't cancel out the way the aligned faces do. Only the eyes are aligned, so anything that differs between the photos also shows up as faint double outlines, such as the ear shapes, the head tilt, and the body pose. Cropping to the faces reduced how much background is in the frame, but it can't remove the background inside the crop.

#### Three-band hybrid: father and sisters

This extends the hybrid to three images. A third image adds a middle band of frequencies, taken with a band-pass filter: the difference of two Gaussians. Up close the high-pass image dominates, at a middle distance the band-pass image does, and from far away the low-pass image does.

$$ H = I_1 * G_{\sigma_{low}} + \left( I_2 * G_{\sigma_a} - I_2 * G_{\sigma_b} \right) + \left( I_3 - I_3 * G_{\sigma_{high}} \right) $$

```python
def three_band(im1, im_mid, im2, sigma_low, sigma_mid, sigma_high):
    low = gaussian_blur(im1, sigma_low)
    mid = gaussian_blur(im_mid, sigma_mid[0]) - gaussian_blur(im_mid, sigma_mid[1])
    high = im2 - gaussian_blur(im2, sigma_high)
    return low, mid, high, low + mid + high
```

The middle and high images are aligned to the low-frequency image using the same two-point translation, scale, and rotation as `align_images`, applied as a single warp. Both results use $$ \sigma_{low} = 14 $$, $$ \sigma_a = 4 $$, $$ \sigma_b = 10 $$, and $$ \sigma_{high} = 2 $$. For a 1350 px tall image this keeps the three bands apart:

| Band | Cycles per image |
|------|------------------|
| Low | below 18 |
| Middle | 25 to 63 |
| High | above 126 |

##### Father (low), sister 1 (middle), sister 2 (high)

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/2.2/father_sister1_sister2_im1.jpg" alt="Father"><figcaption>Father (low)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/father_sister1_sister2_im_mid.jpg" alt="Sister 1"><figcaption>Sister 1 (middle)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/father_sister1_sister2_im2.jpg" alt="Sister 2"><figcaption>Sister 2 (high)</figcaption></figure>
</div>

<figure>
  <img loading="lazy" src="../images/assignment-2/2.2/father_sister1_sister2_hybrid.jpg" alt="Three-band hybrid">
  <figcaption>Three-band hybrid</figcaption>
</figure>

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/2.2/father_sister1_sister2_hybrid_0.5.jpg" alt="1/2"><figcaption>1/2</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/father_sister1_sister2_hybrid_0.25.jpg" alt="1/4"><figcaption>1/4</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/father_sister1_sister2_hybrid_0.125.jpg" alt="1/8"><figcaption>1/8</figcaption></figure>
</div>

##### Sister 1 (low), father (middle), sister 2 (high)

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/2.2/sister1_father_sister2_im1.jpg" alt="Sister 1"><figcaption>Sister 1 (low)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/sister1_father_sister2_im_mid.jpg" alt="Father"><figcaption>Father (middle)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/sister1_father_sister2_im2.jpg" alt="Sister 2"><figcaption>Sister 2 (high)</figcaption></figure>
</div>

<figure>
  <img loading="lazy" src="../images/assignment-2/2.2/sister1_father_sister2_hybrid.jpg" alt="Three-band hybrid">
  <figcaption>Three-band hybrid</figcaption>
</figure>

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/2.2/sister1_father_sister2_hybrid_0.5.jpg" alt="1/2"><figcaption>1/2</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/sister1_father_sister2_hybrid_0.25.jpg" alt="1/4"><figcaption>1/4</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/sister1_father_sister2_hybrid_0.125.jpg" alt="1/8"><figcaption>1/8</figcaption></figure>
</div>

### Bells & Whistles: Hybrid Images

The same hybrid ($$ \sigma_{low} = 8 $$, $$ \sigma_{high} = 4 $$), with colour kept in the low-frequency image, the high-frequency image, both, or neither. Colour images are filtered one channel at a time.

<div class="media-grid media-grid--2x2">
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_colour_none.jpg" alt="Grayscale"><figcaption>Grayscale</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_colour_low.jpg" alt="Colour low"><figcaption>Colour low frequencies (Derek)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_colour_high.jpg" alt="Colour high"><figcaption>Colour high frequencies (Nutmeg)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/derek_nutmeg_colour_both.jpg" alt="Colour both"><figcaption>Colour both</figcaption></figure>
</div>

#### Cats

Same settings as above ($$ \sigma_{low} = 8 $$, $$ \sigma_{high} = 4 $$, cropped to the faces).

##### Big cat (low), little cat (high)

<div class="media-grid media-grid--2x2">
  <figure><img loading="lazy" src="../images/assignment-2/2.2/bigcat_littlecat_colour_none.jpg" alt="Grayscale"><figcaption>Grayscale</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/bigcat_littlecat_colour_low.jpg" alt="Colour low"><figcaption>Colour low frequencies (big cat)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/bigcat_littlecat_colour_high.jpg" alt="Colour high"><figcaption>Colour high frequencies (little cat)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/bigcat_littlecat_colour_both.jpg" alt="Colour both"><figcaption>Colour both</figcaption></figure>
</div>

##### Little cat (low), big cat (high)

<div class="media-grid media-grid--2x2">
  <figure><img loading="lazy" src="../images/assignment-2/2.2/littlecat_bigcat_colour_none.jpg" alt="Grayscale"><figcaption>Grayscale</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/littlecat_bigcat_colour_low.jpg" alt="Colour low"><figcaption>Colour low frequencies (little cat)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/littlecat_bigcat_colour_high.jpg" alt="Colour high"><figcaption>Colour high frequencies (big cat)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/littlecat_bigcat_colour_both.jpg" alt="Colour both"><figcaption>Colour both</figcaption></figure>
</div>

#### Three-band hybrids

With three bands, colour can be kept in the low, middle, or high band, or in all three. Same settings as above.

##### Father (low), sister 1 (middle), sister 2 (high)

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/2.2/father_sister1_sister2_colour_none.jpg" alt="Grayscale"><figcaption>Grayscale</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/father_sister1_sister2_colour_low.jpg" alt="Colour low"><figcaption>Colour low (father)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/father_sister1_sister2_colour_mid.jpg" alt="Colour middle"><figcaption>Colour middle (sister 1)</figcaption></figure>
</div>
<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/2.2/father_sister1_sister2_colour_high.jpg" alt="Colour high"><figcaption>Colour high (sister 2)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/father_sister1_sister2_colour_all.jpg" alt="Colour all"><figcaption>Colour all three</figcaption></figure>
</div>

##### Sister 1 (low), father (middle), sister 2 (high)

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/2.2/sister1_father_sister2_colour_none.jpg" alt="Grayscale"><figcaption>Grayscale</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/sister1_father_sister2_colour_low.jpg" alt="Colour low"><figcaption>Colour low (sister 1)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/sister1_father_sister2_colour_mid.jpg" alt="Colour middle"><figcaption>Colour middle (father)</figcaption></figure>
</div>
<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-2/2.2/sister1_father_sister2_colour_high.jpg" alt="Colour high"><figcaption>Colour high (sister 2)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-2/2.2/sister1_father_sister2_colour_all.jpg" alt="Colour all"><figcaption>Colour all three</figcaption></figure>
</div>

### 2.3 Gaussian and Laplacian Stacks

### 2.4 Multiresolution Blending

### Bells & Whistles: Blending
