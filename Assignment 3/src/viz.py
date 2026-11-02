"""Figure helpers shared by the Assignment 3 scripts."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import save_image  # noqa: E402


def side_by_side(a, b, gap=20):
    h = max(a.shape[0], b.shape[0])
    out = np.ones((h, a.shape[1] + gap + b.shape[1], 3), np.float32)
    out[:a.shape[0], :a.shape[1]] = a
    out[:b.shape[0], a.shape[1] + gap:] = b
    return out, a.shape[1] + gap


def figure(img, width=12):
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(width, width * h / w), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.clip(img, 0, 1))
    ax.set_axis_off()
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    return fig, ax


def save(fig, path):
    fig.savefig(path, dpi=fig.dpi, pil_kwargs={"quality": 90})
    plt.close(fig)


def points(img, pts, path, color="#ff3030", size=6, labels=False):
    fig, ax = figure(img)
    ax.scatter(pts[:, 0], pts[:, 1], s=size, c=color, linewidths=0)
    if labels:
        for i, (x, y) in enumerate(pts):
            ax.text(x + 12, y - 12, str(i + 1), color=color, fontsize=11, fontweight="bold")
    save(fig, path)


def correspondences(a, b, pa, pb, path):
    """Numbered clicked points on both images of a pair."""
    img, off = side_by_side(a, b)
    fig, ax = figure(img)
    colors = plt.cm.tab10.colors
    for i, (p, q) in enumerate(zip(pa, pb)):
        c = colors[i % 10]
        for x, y in (p, (q[0] + off, q[1])):
            ax.plot(x, y, "+", color=c, ms=12, mew=2)
            ax.text(x + 14, y - 14, str(i + 1), color=c, fontsize=11, fontweight="bold")
    save(fig, path)


def matches(a, b, p1, p2, path, inliers=None):
    """Lines between matched points; inliers green and outliers red when given."""
    img, off = side_by_side(a, b)
    fig, ax = figure(img)
    if inliers is None:
        inliers = np.ones(len(p1), bool)
        colors = plt.cm.hsv(np.linspace(0, 1, len(p1), endpoint=False))
    else:
        colors = np.where(inliers[:, None], [[0.1, 0.9, 0.2, 1]], [[1, 0.15, 0.15, 1]])
    for (x1, y1), (x2, y2), c in zip(p1, p2, colors):
        ax.plot([x1, x2 + off], [y1, y2], "-", color=c, lw=0.8)
        ax.plot([x1, x2 + off], [y1, y2], "o", color=c, ms=3)
    save(fig, path)


def patch_grid(patches, path, titles=None, cols=8):
    """Grid of small grayscale patches, each scaled to its own range."""
    rows = int(np.ceil(len(patches) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.55), dpi=150)
    for ax in np.ravel(axes):
        ax.set_axis_off()
    for i, (ax, p) in enumerate(zip(np.ravel(axes), patches)):
        ax.imshow(p, cmap="gray", interpolation="nearest")
        if titles is not None:
            ax.set_title(titles[i], fontsize=8)
    fig.tight_layout()
    save(fig, path)


def save_scaled(path, img, max_side=2400):
    """Save, downscaling so the longest side is at most max_side."""
    s = max_side / max(img.shape[:2])
    if s < 1:
        import cv2
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    save_image(path, img)
