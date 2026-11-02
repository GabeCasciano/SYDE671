---
title: "Assignment 3: [Auto]Stitching Photo Mosaics"
date: 2026-11-01
excerpt: "Homographies, image warping, and mosaics, stitched by hand and then automatically with Harris corners, feature matching, and RANSAC."
---

## Procedure: Capturing the Source Images

The source images come from videos taken on an iPhone at the Grand Canyon. The frames were pulled out of the videos afterwards, which gives many closely spaced views of each scene to choose from.

<div class="media-grid" style="grid-template-columns: 1.78fr 0.56fr">
  <figure>
    <video src="../images/assignment-3/videos/IMG_0852.mp4" poster="../images/assignment-3/videos/IMG_0852_poster.jpg" controls loop muted playsinline></video>
    <figcaption>IMG_0852, landscape, 34 s</figcaption>
  </figure>
  <figure>
    <video src="../images/assignment-3/videos/IMG_0861.mp4" poster="../images/assignment-3/videos/IMG_0861_poster.jpg" controls loop muted playsinline></video>
    <figcaption>IMG_0861, portrait, 87 s</figcaption>
  </figure>
</div>

### Extracting frames

`Assignment 3/src/extract_frames.py` turns each video into a folder of still frames:

1. ffmpeg decodes the video (iPhone HEVC) and samples it at a fixed rate, 2 frames per second by default. ffmpeg also applies the rotation stored in the file, so portrait videos come out upright.
2. iPhones can record video in HDR (HLG). When a file is tagged as HDR, the frames are tone mapped to standard range so they do not look washed out. Both videos here are standard range, so this step was skipped.
3. Each frame is compared to the last frame that was kept, using a 128-pixel-wide blurred grayscale copy of both. If the mean absolute difference is below a threshold (0.1 on a 0 to 1 scale), the frame is too similar and is dropped.

```python
TONEMAP = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
           "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")

vf = f"fps={fps}" + ("," + TONEMAP if is_hdr(video) else "")
subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", vf,
                "-q:v", "2", f"{tmp}/%05d.jpg"], check=True)

last, kept = None, 0
for f in frames:
    t = thumb(f)
    if last is None or np.abs(t - last).mean() > diff:
        kept += 1
        shutil.move(f, out_dir / f"{video.stem}_{kept:04d}.jpg")
        last = t
```

| Video | Sampled frames | Kept frames |
|---|---|---|
| IMG_0852 | 68 | 13 |
| IMG_0861 | 174 | 81 |

Because the camera pans continuously in both videos, almost every pair of frames half a second apart differed by more than 0.04, so a low threshold kept nearly everything. Raising it to 0.1 keeps a frame only after the view has moved a fair amount, which left neighbouring kept frames overlapping by roughly half.

## Part A: Image Warping and Mosaicing

### A.1 Source Images

Each mosaic is three frames from the videos above. The same mosaics are used in Part B.

| Mosaic | Frames |
|---|---|
| canyon_slope | IMG_0852 frames 9, 10, 11 |
| canyon_ridge | IMG_0861 frames 40, 44, 48 |

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-3/a1/canyon_slope_1.jpg" alt=""><figcaption>canyon_slope, frame 9</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/a1/canyon_slope_2.jpg" alt=""><figcaption>canyon_slope, frame 10</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/a1/canyon_slope_3.jpg" alt=""><figcaption>canyon_slope, frame 11</figcaption></figure>
</div>
<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-3/a1/canyon_ridge_1.jpg" alt=""><figcaption>canyon_ridge, frame 40</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/a1/canyon_ridge_2.jpg" alt=""><figcaption>canyon_ridge, frame 44</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/a1/canyon_ridge_3.jpg" alt=""><figcaption>canyon_ridge, frame 48</figcaption></figure>
</div>

### A.2 Recovering Homographies

Twelve correspondences were clicked by hand for each neighbouring pair of frames (`Assignment 3/src/click_points.py`).

<div class="media-grid media-grid--2x2">
  <figure><img loading="lazy" src="../images/assignment-3/a2/canyon_slope_1_points.jpg" alt=""><figcaption>canyon_slope, frames 9 and 10</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/a2/canyon_slope_2_points.jpg" alt=""><figcaption>canyon_slope, frames 10 and 11</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/a2/canyon_ridge_1_points.jpg" alt=""><figcaption>canyon_ridge, frames 40 and 44</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/a2/canyon_ridge_2_points.jpg" alt=""><figcaption>canyon_ridge, frames 44 and 48</figcaption></figure>
</div>

A homography maps a point $$ (x, y) $$ in the first image to $$ (u, v) $$ in the second, up to scale. Fixing $$ h_{33} = 1 $$ leaves 8 unknowns:

$$ u = \frac{h_{11} x + h_{12} y + h_{13}}{h_{31} x + h_{32} y + 1}, \qquad v = \frac{h_{21} x + h_{22} y + h_{23}}{h_{31} x + h_{32} y + 1} $$

Multiplying out the denominators gives two equations per correspondence that are linear in the unknowns. Stacking all $$ n $$ correspondences gives a $$ 2n \times 8 $$ system $$ A h = b $$:

$$ \begin{bmatrix} x_1 & y_1 & 1 & 0 & 0 & 0 & -x_1 u_1 & -y_1 u_1 \\ 0 & 0 & 0 & x_1 & y_1 & 1 & -x_1 v_1 & -y_1 v_1 \\ \vdots & & & & & & & \vdots \\ x_n & y_n & 1 & 0 & 0 & 0 & -x_n u_n & -y_n u_n \\ 0 & 0 & 0 & x_n & y_n & 1 & -x_n v_n & -y_n v_n \end{bmatrix} \begin{bmatrix} h_{11} \\ h_{12} \\ h_{13} \\ h_{21} \\ h_{22} \\ h_{23} \\ h_{31} \\ h_{32} \end{bmatrix} = \begin{bmatrix} u_1 \\ v_1 \\ \vdots \\ u_n \\ v_n \end{bmatrix} $$

With 12 correspondences there are 24 equations for 8 unknowns, so the system is solved with least squares.

```python
def system(im1_pts, im2_pts):
    A, b = [], []
    for (x, y), (u, v) in zip(im1_pts, im2_pts):
        A.append([x, y, 1, 0, 0, 0, -x * u, -y * u])
        A.append([0, 0, 0, x, y, 1, -x * v, -y * v])
        b += [u, v]
    return np.array(A, dtype=float), np.array(b, dtype=float)


def computeH(im1_pts, im2_pts):
    A, b = system(im1_pts, im2_pts)
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.append(h, 1).reshape(3, 3)
```

The full system for canyon_slope, frames 9 and 10 ($$ A $$, then $$ b $$ after the bar):

```
  332.75   601.41  1.00     0.00     0.00  0.00   -206349.58  -372957.07 |  620.14
    0.00     0.00  0.00   332.75   601.41  1.00   -221482.21  -400307.86 |  665.61
  166.12   726.76  1.00     0.00     0.00  0.00    -76101.29  -332928.36 |  458.10
    0.00     0.00  0.00   166.12   726.76  1.00   -129366.84  -565954.83 |  778.74
  536.06   824.60  1.00     0.00     0.00  0.00   -427487.77  -657583.43 |  797.46
    0.00     0.00  0.00   536.06   824.60  1.00   -481366.97  -740463.16 |  897.97
 1147.53   569.31  1.00     0.00     0.00  0.00  -1630812.78  -809076.81 | 1421.16
    0.00     0.00  0.00  1147.53   569.31  1.00   -777843.20  -385902.60 |  677.84
 1356.95   462.30  1.00     0.00     0.00  0.00  -2237514.83  -762303.04 | 1648.93
    0.00     0.00  0.00  1356.95   462.30  1.00   -789119.58  -268846.60 |  581.54
 1321.79   237.59  1.00     0.00     0.00  0.00  -2139128.32  -384503.27 | 1618.35
    0.00     0.00  0.00  1321.79   237.59  1.00   -453463.40   -81508.98 |  343.07
 1447.14   269.69  1.00     0.00     0.00  0.00  -2523389.32  -470261.37 | 1743.70
    0.00     0.00  0.00  1447.14   269.69  1.00   -551771.78  -102828.74 |  381.28
 1416.57    15.93  1.00     0.00     0.00  0.00  -2441927.69   -27466.01 | 1723.83
    0.00     0.00  0.00  1416.57    15.93  1.00   -169821.49    -1910.10 |  119.88
 1448.67   489.82  1.00     0.00     0.00  0.00  -2539342.04  -858590.94 | 1752.88
    0.00     0.00  0.00  1448.67   489.82  1.00   -886748.72  -299823.50 |  612.11
  850.96    81.67  1.00     0.00     0.00  0.00   -966097.28   -92714.65 | 1135.30
    0.00     0.00  0.00   850.96    81.67  1.00   -154049.05   -14783.82 |  181.03
  286.89   434.79  1.00     0.00     0.00  0.00   -166946.29  -253011.20 |  581.92
    0.00     0.00  0.00   286.89   434.79  1.00   -144908.86  -219612.94 |  505.11
 1468.54    26.63  1.00     0.00     0.00  0.00  -2619074.44   -47499.95 | 1783.45
    0.00     0.00  0.00  1468.54    26.63  1.00   -191766.67    -3477.91 |  130.58
```

Recovered homographies (first frame mapped into the second), with the RMS distance between the mapped clicked points and their clicked partners:

canyon_slope, 9 to 10 (RMS error 3.24 px):

$$ H = \begin{bmatrix} 0.834967 & -0.0652111 & 352.782 \\ -0.00206532 & 0.905444 & 95.082 \\ -7.72146 \times 10^{-5} & -2.70661 \times 10^{-5} & 1 \end{bmatrix} $$

canyon_slope, 10 to 11 (RMS error 2.67 px):

$$ H = \begin{bmatrix} 0.901914 & -0.039196 & 199.319 \\ -0.00801003 & 0.932446 & 131.516 \\ -4.58217 \times 10^{-5} & -2.54684 \times 10^{-5} & 1 \end{bmatrix} $$

canyon_ridge, 40 to 44 (RMS error 3.78 px):

$$ H = \begin{bmatrix} 0.801391 & -0.0582999 & 491.428 \\ -0.124356 & 0.910297 & 132.916 \\ -1.70132 \times 10^{-4} & -1.67338 \times 10^{-5} & 1 \end{bmatrix} $$

canyon_ridge, 44 to 48 (RMS error 3.65 px):

$$ H = \begin{bmatrix} 0.841398 & -0.0425021 & 642.385 \\ -0.0626364 & 1.09375 & -242.963 \\ -2.10217 \times 10^{-4} & 8.11999 \times 10^{-5} & 1 \end{bmatrix} $$

### A.3 Warping and Rectification

Both warps use inverse warping. The four corners of the input are pushed through $$ H $$ to find the output bounding box, then every output pixel is mapped back through $$ H^{-1} $$ and sampled from the input. Output pixels that land outside the input get alpha 0. Nearest neighbour rounds the source position to the closest pixel; bilinear weights the four surrounding pixels by their distance to it. Code: `Assignment 3/src/homography.py` and `Assignment 3/src/rectify.py`.

```python
def sample_nn(im, xs, ys):
    h, w = im.shape[:2]
    xi, yi = np.floor(xs + 0.5).astype(int), np.floor(ys + 0.5).astype(int)
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = im[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)]
    return out, valid


def sample_bilinear(im, xs, ys):
    h, w = im.shape[:2]
    valid = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 2)
    fx, fy = xs - x0, ys - y0
    if im.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    out = ((1 - fx) * (1 - fy) * im[y0, x0] + fx * (1 - fy) * im[y0, x0 + 1]
           + (1 - fx) * fy * im[y0 + 1, x0] + fx * fy * im[y0 + 1, x0 + 1])
    return out, valid


def warp(im, H, box=None, sampler=sample_bilinear):
    x0, y0, x1, y1 = box or warp_box(H, im.shape)
    xs, ys = np.meshgrid(np.arange(x0, x1, dtype=float), np.arange(y0, y1, dtype=float))
    src = apply_h(np.linalg.inv(H), np.c_[xs.ravel(), ys.ravel()])
    out, valid = sampler(im, src[:, 0].reshape(xs.shape), src[:, 1].reshape(xs.shape))
    if out.ndim == 2:
        out = out[..., None]
    out = out * valid[..., None]
    return np.dstack([out, valid]).astype(np.float32)
```

For rectification, the four corners of each book cover are clicked and mapped to a rectangle with the cover's measured aspect ratio, so $$ H $$ comes from `computeH` with exactly 4 correspondences. The output is limited to one cover width and height around the cover, since the table far from the book stretches out without bound.

#### Feedback and Control Systems (15.5 x 23.5 cm)

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-3/a3/books_1_corners.jpg" alt=""><figcaption>Input with clicked corners</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/a3/books_1_nn.jpg" alt=""><figcaption>Nearest neighbour</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/a3/books_1_bilinear.jpg" alt=""><figcaption>Bilinear</figcaption></figure>
</div>
<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-3/a3/books_1_nn_zoom.jpg" alt=""><figcaption>Nearest neighbour, centre of the cover at 4x</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/a3/books_1_bilinear_zoom.jpg" alt=""><figcaption>Bilinear, centre of the cover at 4x</figcaption></figure>
</div>

#### Transducer Interfacing Handbook (21.5 x 15.5 cm)

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-3/a3/books_2_corners.jpg" alt=""><figcaption>Input with clicked corners</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/a3/books_2_nn.jpg" alt=""><figcaption>Nearest neighbour</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/a3/books_2_bilinear.jpg" alt=""><figcaption>Bilinear</figcaption></figure>
</div>
<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-3/a3/books_2_nn_zoom.jpg" alt=""><figcaption>Nearest neighbour, centre of the cover at 4x</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/a3/books_2_bilinear_zoom.jpg" alt=""><figcaption>Bilinear, centre of the cover at 4x</figcaption></figure>
</div>

| Image | Output size | Nearest neighbour | Bilinear |
|---|---|---|---|
| Feedback and Control Systems | 2327 x 2261 | 0.48 s | 1.25 s |
| Transducer Interfacing Handbook | 2134 x 1941 | 0.38 s | 0.96 s |

### A.4 Blending into a Mosaic

The middle frame of each set is the reference. The pairwise homographies are chained so every frame maps into it: the first frame through $$ H_{1 \to 2} $$ and the third through $$ H_{2 \to 3}^{-1} $$. Each frame is inverse warped (bilinear) onto a shared canvas sized to fit all of them, and carries a weight that is 1 at its centre and falls linearly to 0 at its nearest edge. Code: `Assignment 3/src/blend.py`.

Three ways of combining the warped frames are compared:

- **Overwrite**: each frame is pasted over the previous one.
- **Feather**: every pixel is the weighted average of the frames covering it.
- **Two-band**: each frame is split into a low band (Gaussian blur, $$ \sigma = 6 $$) and a high band (the frame minus its low band). The low bands are feathered as above, and the high band at each pixel is taken from the one frame with the largest weight there.

```python
def chain_to_ref(pair_Hs):
    n, m = len(pair_Hs) + 1, (len(pair_Hs) + 1) // 2
    Hs = [np.eye(3) for _ in range(n)]
    for i in range(m - 1, -1, -1):
        Hs[i] = Hs[i + 1] @ pair_Hs[i]
    for i in range(m + 1, n):
        Hs[i] = Hs[i - 1] @ np.linalg.inv(pair_Hs[i - 1])
    return [H / H[2, 2] for H in Hs]


def edge_weight(shape):
    h, w = shape[:2]
    y, x = np.mgrid[:h, :w]
    d = np.minimum.reduce([x + 1, w - x, y + 1, h - y]).astype(np.float32)
    return d / d.max()
```

```python
    # Two-band: feather the low band, take the high band from the highest-weight image
    lows = [masked_blur(rgb, a, sigma) for rgb, _, a in layers]
    low = sum(lo * w[..., None] for lo, (_, w, _) in zip(lows, layers)) / total
    best = np.argmax(np.stack([w for _, w, _ in layers]), axis=0)
    high = np.zeros_like(low)
    for i, (lo, (rgb, _, a)) in enumerate(zip(lows, layers)):
        pick = (best == i) & (a > 0)
        high[pick] = (rgb - lo)[pick]
    return crop(np.clip(low + high, 0, 1) * covered[..., None], covered)
```

#### canyon_slope

<figure>
  <img loading="lazy" src="../images/assignment-3/a4/canyon_slope_overwrite.jpg" alt="">
  <figcaption>Overwrite</figcaption>
</figure>
<figure>
  <img loading="lazy" src="../images/assignment-3/a4/canyon_slope_feather.jpg" alt="">
  <figcaption>Feather</figcaption>
</figure>
<figure>
  <img loading="lazy" src="../images/assignment-3/mosaics/canyon_slope_manual.jpg" alt="">
  <figcaption>Two-band</figcaption>
</figure>

#### canyon_ridge

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-3/a4/canyon_ridge_overwrite.jpg" alt=""><figcaption>Overwrite</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/a4/canyon_ridge_feather.jpg" alt=""><figcaption>Feather</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/mosaics/canyon_ridge_manual.jpg" alt=""><figcaption>Two-band</figcaption></figure>
</div>

## Part B: Feature Matching for Autostitching

Part B follows Brown, Szeliski and Winder, "Multi-Image Matching using Multi-Scale Oriented Patches" (MOPS), single scale and axis aligned unless noted. Code: `Assignment 3/src/features.py` (detection, descriptors, matching, RANSAC) and `Assignment 3/src/part_b.py` (figures and mosaics).

The mosaics are the same as in Part A.

### B.1 Harris Corners and Adaptive Non-Maximal Suppression

Corners come from the provided `harris.py`: the Harris corner strength (harmonic mean of the eigenvalues of the structure tensor, $$ \det H / \operatorname{tr} H $$), every local maximum kept, and corners within 20 pixels of the border discarded. On a 1920x1080 frame this gives over 50,000 corners.

Adaptive non-maximal suppression gives each corner a suppression radius: the distance to the nearest corner that is sufficiently stronger.

$$ r_i = \min_j \lVert x_i - x_j \rVert \quad \text{s.t.} \quad f(x_i) < c_{\text{robust}} \, f(x_j), \qquad c_{\text{robust}} = 0.9 $$

The 500 corners with the largest radii are kept. Sorting the corners by strength means the corners that can suppress corner $$ i $$ are always a prefix of the sorted list, so the radii are computed in chunks with one distance matrix per chunk. Only the 10,000 strongest corners are considered, which keeps that matrix a manageable size.

```python
def anms(pts, f, n=500, c=0.9, max_pts=10000, chunk=2000):
    order = np.argsort(-f)[:max_pts]
    pts, f = pts[order], f[order]
    r = np.full(len(pts), np.inf)
    stronger = np.searchsorted(-c * f, -f, side="left")
    for s in range(0, len(pts), chunk):
        e = min(s + chunk, len(pts))
        k = stronger[s:e].max()
        if k == 0:
            continue
        d = dist2(pts[s:e], pts[:k])
        d[np.arange(k)[None, :] >= stronger[s:e, None]] = np.inf
        r[s:e] = d.min(1)
    keep = np.argsort(-r)[:n]
    return pts[keep], np.sqrt(r[keep])
```

#### canyon_slope, first frame

<figure>
  <img loading="lazy" src="../images/assignment-3/b1/canyon_slope_harris_all.jpg" alt="">
  <figcaption>All Harris corners (57,108)</figcaption>
</figure>
<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-3/b1/canyon_slope_strongest_500.jpg" alt=""><figcaption>Strongest 500 corners</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/b1/canyon_slope_anms_500.jpg" alt=""><figcaption>ANMS 500 corners, r = 41.3</figcaption></figure>
</div>

#### canyon_ridge, first frame

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-3/b1/canyon_ridge_harris_all.jpg" alt=""><figcaption>All Harris corners (53,523)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/b1/canyon_ridge_strongest_500.jpg" alt=""><figcaption>Strongest 500</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/b1/canyon_ridge_anms_500.jpg" alt=""><figcaption>ANMS 500, r = 33.1</figcaption></figure>
</div>

### B.2 Feature Descriptors

Each descriptor is an 8x8 grid of samples spaced 5 pixels apart, so it covers a 40x40 window around the corner. The image is first blurred with a Gaussian ($$ \sigma = 2.5 $$, half the sample spacing) so each sample summarizes its neighbourhood instead of aliasing, and the samples are read with bilinear interpolation. Each descriptor is then bias/gain normalized to zero mean and unit standard deviation.

```python
def descriptors(g, pts, angles=None, size=8, spacing=5):
    blurred = cv2.GaussianBlur(g, (0, 0), spacing / 2)
    k = (np.arange(size) - (size - 1) / 2) * spacing
    ox, oy = [a.ravel() for a in np.meshgrid(k, k)]
    if angles is None:
        angles = np.zeros(len(pts))
    c, s = np.cos(angles)[:, None], np.sin(angles)[:, None]
    xs = pts[:, :1] + c * ox - s * oy
    ys = pts[:, 1:] + s * ox + c * oy
    d, _ = sample_bilinear(blurred, xs, ys)
    return (d - d.mean(1, keepdims=True)) / (d.std(1, keepdims=True) + 1e-8)
```

<figure>
  <img loading="lazy" src="../images/assignment-3/b2/canyon_slope_locations.jpg" alt="">
  <figcaption>Twelve features and their 40x40 windows</figcaption>
</figure>
<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-3/b2/canyon_slope_windows.jpg" alt=""><figcaption>40x40 windows (grayscale)</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/b2/canyon_slope_descriptors.jpg" alt=""><figcaption>Normalized 8x8 descriptors</figcaption></figure>
</div>

### B.3 Feature Matching

Every descriptor in one image is compared with every descriptor in the other using squared Euclidean distance. A match is kept only when the ratio of the nearest to the second-nearest squared distance (1-NN/2-NN, Figure 6b of the paper) is below 0.5.

```python
def match(d1, d2, ratio=0.5):
    d = dist2(d1, d2)
    nn = np.argsort(d, axis=1)[:, :2]
    rows = np.arange(len(d1))
    e1, e2 = d[rows, nn[:, 0]], d[rows, nn[:, 1]]
    r = e1 / np.maximum(e2, 1e-12)
    keep = r < ratio
    return np.c_[rows[keep], nn[keep, 0]], r[keep]
```

<div class="media-grid media-grid--2x2">
  <figure><img loading="lazy" src="../images/assignment-3/b3/canyon_slope_1_matches.jpg" alt=""><figcaption>canyon_slope, frames 9 and 10: 146 matches</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/b3/canyon_slope_2_matches.jpg" alt=""><figcaption>canyon_slope, frames 10 and 11: 194 matches</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/b3/canyon_ridge_1_matches.jpg" alt=""><figcaption>canyon_ridge, frames 40 and 44: 161 matches</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/b3/canyon_ridge_2_matches.jpg" alt=""><figcaption>canyon_ridge, frames 44 and 48: 93 matches</figcaption></figure>
</div>

### B.4 RANSAC and Automatic Mosaics

Each RANSAC iteration picks 4 random matches, solves the exact homography through them with `computeH` from Part A, and counts the matches that land within 3 pixels of where that homography predicts. After 2000 iterations the largest inlier set is kept and the homography is refit to all of its inliers with least squares. The mosaics are then built with the same warping and blending code as Part A.

```python
def ransac(p1, p2, iters=2000, eps=3.0, seed=0):
    rng = np.random.default_rng(seed)
    best = np.zeros(len(p1), bool)
    for _ in range(iters):
        idx = rng.choice(len(p1), 4, replace=False)
        try:
            H = computeH(p1[idx], p2[idx])
        except np.linalg.LinAlgError:
            continue
        err = np.linalg.norm(apply_h(H, p1) - p2, axis=1)
        inliers = err < eps
        if inliers.sum() > best.sum():
            best = inliers
    return computeH(p1[best], p2[best]), best
```

| Pair | Matches | RANSAC inliers |
|---|---|---|
| canyon_slope, frames 9 and 10 | 146 | 112 |
| canyon_slope, frames 10 and 11 | 194 | 179 |
| canyon_ridge, frames 40 and 44 | 161 | 129 |
| canyon_ridge, frames 44 and 48 | 93 | 60 |

<div class="media-grid media-grid--2x2">
  <figure><img loading="lazy" src="../images/assignment-3/b4/canyon_slope_1_ransac.jpg" alt=""><figcaption>canyon_slope, frames 9 and 10</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/b4/canyon_slope_2_ransac.jpg" alt=""><figcaption>canyon_slope, frames 10 and 11</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/b4/canyon_ridge_1_ransac.jpg" alt=""><figcaption>canyon_ridge, frames 40 and 44</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/b4/canyon_ridge_2_ransac.jpg" alt=""><figcaption>canyon_ridge, frames 44 and 48</figcaption></figure>
</div>

Inliers are green and outliers red.

Manual (Part A, clicked points) and automatic mosaics, both with two-band blending:

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-3/mosaics/canyon_slope_manual.jpg" alt=""><figcaption>canyon_slope, manual</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/mosaics/canyon_slope_auto.jpg" alt=""><figcaption>canyon_slope, automatic</figcaption></figure>
</div>
<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-3/mosaics/canyon_ridge_manual.jpg" alt=""><figcaption>canyon_ridge, manual</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/mosaics/canyon_ridge_auto.jpg" alt=""><figcaption>canyon_ridge, automatic</figcaption></figure>
</div>

### B.5 Bells & Whistles

Code: `Assignment 3/src/bells.py`. All three options were implemented.

#### Rotation invariance

Each feature gets an orientation from the image gradient after a wide Gaussian blur ($$ \sigma = 4.5 $$, as in the paper), and the 8x8 sampling grid is rotated to that angle before sampling. To test it, the second canyon_slope frame was rotated 45 degrees with the Part A warp and matched against the first frame, with and without orientation.

| Descriptors | Matches | RANSAC inliers |
|---|---|---|
| Axis aligned | 0 | 0 |
| Oriented | 104 | 87 |

<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-3/b5/rotation_input.jpg" alt=""><figcaption>Second frame rotated 45 degrees</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/b5/rotation_oriented_patches.jpg" alt=""><figcaption>Oriented 40x40 windows</figcaption></figure>
</div>
<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-3/b5/rotation_axis_ransac.jpg" alt=""><figcaption>Axis aligned</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/b5/rotation_oriented_ransac.jpg" alt=""><figcaption>Oriented</figcaption></figure>
</div>
<figure>
  <img loading="lazy" src="../images/assignment-3/b5/rotation_mosaic.jpg" alt="">
  <figcaption>Mosaic of the first frame and the rotated second frame</figcaption>
</figure>

#### Multiscale processing

A Gaussian pyramid ($$ \sigma = 1 $$ blur, then subsample by 2) is built with 3 levels. Harris, ANMS, and descriptors run on every level, so a descriptor from level $$ l $$ covers a $$ 40 \cdot 2^l $$ pixel window in the original image. Points are mapped back to full resolution, and descriptors from all levels are matched together. To test it, the second canyon_ridge frame was shrunk to half size and matched against the first frame.

| Detection | Matches | RANSAC inliers |
|---|---|---|
| Single scale | 0 | 0 |
| 3-level pyramid | 351 | 263 |

<figure>
  <img loading="lazy" src="../images/assignment-3/b5/multiscale_corners.jpg" alt="">
  <figcaption>Features from the pyramid; window size and colour show the level</figcaption>
</figure>
<div class="media-grid">
  <figure><img loading="lazy" src="../images/assignment-3/b5/multiscale_single_ransac.jpg" alt=""><figcaption>Single scale</figcaption></figure>
  <figure><img loading="lazy" src="../images/assignment-3/b5/multiscale_pyramid_ransac.jpg" alt=""><figcaption>3-level pyramid</figcaption></figure>
</div>
<figure>
  <img loading="lazy" src="../images/assignment-3/b5/multiscale_mosaic.jpg" alt="">
  <figcaption>Mosaic of the full-size first frame and the half-size second frame</figcaption>
</figure>

#### Panorama recognition

Thirteen frames, six from IMG_0852, five from IMG_0861, and two unrelated frames, are shuffled and given to the program with no ordering. Every pair is matched and run through RANSAC. Following Brown and Lowe, a pair is accepted as overlapping when the inlier count exceeds $$ 8 + 0.3 \times $$ the number of matches. The connected groups of accepted pairs are the panoramas. In each group, the image with the most total inliers becomes the reference, and the other images are linked to it along the strongest accepted pairs (a maximum spanning tree), multiplying homographies along the way.

<figure>
  <img loading="lazy" src="../images/assignment-3/b5/recognition_inputs.jpg" alt="">
  <figcaption>The shuffled input set</figcaption>
</figure>
<figure>
  <img loading="lazy" src="../images/assignment-3/b5/recognition_graph.jpg" alt="">
  <figcaption>RANSAC inliers for every pair; * marks accepted pairs</figcaption>
</figure>

| Group | Images | Source |
|---|---|---|
| 1 | 1, 8, 9, 10, 13 | IMG_0861 frames 36 to 52 |
| 2 | 3, 5, 6, 7, 11, 12 | IMG_0852 frames 7 to 12 |
| none | 2 | IMG_0861 frame 1 |
| none | 4 | IMG_0852 frame 3 |

<figure>
  <img loading="lazy" src="../images/assignment-3/b5/recognition_panorama_1.jpg" alt="">
  <figcaption>Recognized panorama 1</figcaption>
</figure>
<figure>
  <img loading="lazy" src="../images/assignment-3/b5/recognition_panorama_2.jpg" alt="">
  <figcaption>Recognized panorama 2</figcaption>
</figure>
