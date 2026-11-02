"""Feature detection, description, matching, and RANSAC (B.1 to B.4, B.5).

Follows Brown, Szeliski and Winder, "Multi-Image Matching using Multi-Scale Oriented Patches".
"""

import cv2
import numpy as np

import common  # noqa: F401  (adds starter/ to the path)
from harris import dist2, get_harris_corners
from homography import apply_h, computeH, sample_bilinear


def harris(g, edge=20):
    """All Harris corners as (x, y) points and their corner strengths."""
    h, coords = get_harris_corners(g, edge)
    ys, xs = coords
    return np.c_[xs, ys].astype(float), h[ys, xs]


def anms(pts, f, n=500, c=0.9, max_pts=10000, chunk=2000):
    """Adaptive non-maximal suppression: keep the n points with the largest suppression radius.

    Only the max_pts strongest corners are considered, to keep the pairwise distances small.
    """
    order = np.argsort(-f)[:max_pts]
    pts, f = pts[order], f[order]
    r = np.full(len(pts), np.inf)
    # Points are sorted by strength, so the neighbours that can suppress point i are a prefix
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


def orientations(g, pts, sigma=4.5):
    """Angle of the blurred image gradient at each point."""
    gy, gx = np.gradient(cv2.GaussianBlur(g, (0, 0), sigma))
    dx, _ = sample_bilinear(gx, pts[:, 0], pts[:, 1])
    dy, _ = sample_bilinear(gy, pts[:, 0], pts[:, 1])
    return np.arctan2(dy, dx)


def descriptors(g, pts, angles=None, size=8, spacing=5):
    """size x size patches sampled every `spacing` pixels from a blurred window, bias/gain normalized."""
    blurred = cv2.GaussianBlur(g, (0, 0), spacing / 2)
    k = (np.arange(size) - (size - 1) / 2) * spacing
    ox, oy = [a.ravel() for a in np.meshgrid(k, k)]
    if angles is None:
        angles = np.zeros(len(pts))
    c, s = np.cos(angles)[:, None], np.sin(angles)[:, None]
    xs = pts[:, :1] + c * ox - s * oy
    ys = pts[:, 1:] + s * ox + c * oy
    d, _ = sample_bilinear(blurred, xs, ys)
    d = (d - d.mean(1, keepdims=True)) / (d.std(1, keepdims=True) + 1e-8)
    return d


def pyramid(g, levels):
    out = [g]
    for _ in range(levels - 1):
        out.append(cv2.GaussianBlur(out[-1], (0, 0), 1.0)[::2, ::2])
    return out


def detect(g, n=500, oriented=False, levels=1):
    """Harris + ANMS + descriptors on each pyramid level; points are returned in level-0 pixels."""
    all_pts, all_desc, all_ang, all_lvl = [], [], [], []
    for l, p in enumerate(pyramid(g, levels)):
        pts, f = harris(p)
        pts, _ = anms(pts, f, n)
        ang = orientations(p, pts) if oriented else np.zeros(len(pts))
        all_desc.append(descriptors(p, pts, ang))
        all_pts.append(pts * 2 ** l)
        all_ang.append(ang)
        all_lvl.append(np.full(len(pts), l))
    return (np.vstack(all_pts), np.vstack(all_desc),
            np.concatenate(all_ang), np.concatenate(all_lvl))


def match(d1, d2, ratio=0.5):
    """Lowe ratio test on squared distances: index pairs (i in d1, j in d2) and their ratios."""
    d = dist2(d1, d2)
    nn = np.argsort(d, axis=1)[:, :2]
    rows = np.arange(len(d1))
    e1, e2 = d[rows, nn[:, 0]], d[rows, nn[:, 1]]
    r = e1 / np.maximum(e2, 1e-12)
    keep = r < ratio
    return np.c_[rows[keep], nn[keep, 0]], r[keep]


def ransac(p1, p2, iters=2000, eps=3.0, seed=0):
    """4-point RANSAC for p2 ~ H p1; returns H refit on the largest inlier set and the inlier mask."""
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


def auto_h(g1, g2, n=500, ratio=0.5, oriented=False, levels=1, eps=3.0, min_inliers=12):
    """Full pipeline from two grayscale images to H (image 1 to image 2) and the intermediates."""
    f1, f2 = detect(g1, n, oriented, levels), detect(g2, n, oriented, levels)
    m, _ = match(f1[1], f2[1], ratio)
    p1, p2 = f1[0][m[:, 0]], f2[0][m[:, 1]]
    if len(m) < 8:
        return None, dict(f1=f1, f2=f2, p1=p1, p2=p2, inliers=np.zeros(len(m), bool))
    H, inliers = ransac(p1, p2, eps=eps)
    if inliers.sum() < min_inliers:
        H = None
    return H, dict(f1=f1, f2=f2, p1=p1, p2=p2, inliers=inliers)
