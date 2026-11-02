"""Homography estimation and inverse warping (A.2, A.3).

Points are (x, y) with (0, 0) at the centre of the top-left pixel.
"""

import numpy as np


def system(im1_pts, im2_pts):
    """The 2n x 8 system A h = b for p2 ~ H p1, with H[2, 2] = 1."""
    A, b = [], []
    for (x, y), (u, v) in zip(im1_pts, im2_pts):
        A.append([x, y, 1, 0, 0, 0, -x * u, -y * u])
        A.append([0, 0, 0, x, y, 1, -x * v, -y * v])
        b += [u, v]
    return np.array(A, dtype=float), np.array(b, dtype=float)


def computeH(im1_pts, im2_pts):
    """Least-squares homography mapping im1_pts to im2_pts (n x 2 each, n >= 4)."""
    A, b = system(im1_pts, im2_pts)
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.append(h, 1).reshape(3, 3)


def apply_h(H, pts):
    """Map n x 2 points through H."""
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:]


def corners(shape):
    h, w = shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=float)


def warp_box(H, shape):
    """Integer output box (x0, y0, x1, y1) holding the warped image."""
    c = apply_h(H, corners(shape))
    return (int(np.floor(c[:, 0].min())), int(np.floor(c[:, 1].min())),
            int(np.ceil(c[:, 0].max())) + 1, int(np.ceil(c[:, 1].max())) + 1)


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
    """Inverse warp im by H into box; returns the image with an alpha channel appended."""
    x0, y0, x1, y1 = box or warp_box(H, im.shape)
    xs, ys = np.meshgrid(np.arange(x0, x1, dtype=float), np.arange(y0, y1, dtype=float))
    src = apply_h(np.linalg.inv(H), np.c_[xs.ravel(), ys.ravel()])
    out, valid = sampler(im, src[:, 0].reshape(xs.shape), src[:, 1].reshape(xs.shape))
    if out.ndim == 2:
        out = out[..., None]
    out = out * valid[..., None]
    return np.dstack([out, valid]).astype(np.float32)


def warpImageNearestNeighbor(im, H, box=None):
    return warp(im, H, box, sample_nn)


def warpImageBilinear(im, H, box=None):
    return warp(im, H, box, sample_bilinear)
