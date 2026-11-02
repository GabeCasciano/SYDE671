"""Mosaic assembly and blending (A.4)."""

import cv2
import numpy as np

from homography import warp_box, warpImageBilinear


def chain_to_ref(pair_Hs):
    """pair_Hs[i] maps image i to image i + 1; returns H for every image into the middle one."""
    n, m = len(pair_Hs) + 1, (len(pair_Hs) + 1) // 2
    Hs = [np.eye(3) for _ in range(n)]
    for i in range(m - 1, -1, -1):
        Hs[i] = Hs[i + 1] @ pair_Hs[i]
    for i in range(m + 1, n):
        Hs[i] = Hs[i - 1] @ np.linalg.inv(pair_Hs[i - 1])
    return [H / H[2, 2] for H in Hs]


def edge_weight(shape):
    """1 at the centre, falling linearly to 0 at the nearest edge."""
    h, w = shape[:2]
    y, x = np.mgrid[:h, :w]
    d = np.minimum.reduce([x + 1, w - x, y + 1, h - y]).astype(np.float32)
    return d / d.max()


def masked_blur(img, alpha, sigma):
    a = cv2.GaussianBlur(alpha, (0, 0), sigma)
    return cv2.GaussianBlur(img * alpha[..., None], (0, 0), sigma) / np.maximum(a, 1e-6)[..., None]


def warp_all(images, Hs):
    """Warp each image (with its edge weight) into a shared canvas. RGBA images use alpha as a mask."""
    boxes = np.array([warp_box(H, im.shape) for im, H in zip(images, Hs)])
    box = (boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max())
    layers = []
    for im, H in zip(images, Hs):
        w = edge_weight(im.shape) * (im[..., 3] if im.shape[2] == 4 else 1)
        layers.append(warpImageBilinear(np.dstack([im[..., :3], w]), H, box))
    # Channels: RGB, weight, alpha
    return [(l[..., :3], l[..., 3] * l[..., 4], l[..., 4]) for l in layers], box


def crop(img, covered):
    ys, xs = np.nonzero(covered)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def mosaic(images, Hs, mode="twoband", sigma=6.0):
    layers, _ = warp_all(images, Hs)
    # Masked-out pixels (weight 0) do not count as covered
    layers = [(rgb, w, a * (w > 0)) for rgb, w, a in layers]
    covered = sum(a for _, _, a in layers) > 0
    if mode == "overwrite":
        out = np.zeros_like(layers[0][0])
        for rgb, _, a in layers:
            out[a > 0] = rgb[a > 0]
        return crop(out, covered)

    total = np.maximum(sum(w for _, w, _ in layers), 1e-6)[..., None]
    if mode == "feather":
        return crop(sum(rgb * w[..., None] for rgb, w, _ in layers) / total, covered)

    # Two-band: feather the low band, take the high band from the highest-weight image
    lows = [masked_blur(rgb, a, sigma) for rgb, _, a in layers]
    low = sum(lo * w[..., None] for lo, (_, w, _) in zip(lows, layers)) / total
    best = np.argmax(np.stack([w for _, w, _ in layers]), axis=0)
    high = np.zeros_like(low)
    for i, (lo, (rgb, _, a)) in enumerate(zip(lows, layers)):
        pick = (best == i) & (a > 0)
        high[pick] = (rgb - lo)[pick]
    return crop(np.clip(low + high, 0, 1) * covered[..., None], covered)
