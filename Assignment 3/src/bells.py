"""B.5 Bells and whistles: rotation invariance, multiscale processing, panorama recognition.

Usage: python bells.py [rotation] [multiscale] [recognition]
"""

import argparse
import json

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import viz
from blend import mosaic
from common import DATA_DIR, MOSAICS, gray, load_image, save_image, web_dir
from features import detect, match, ransac
from homography import warpImageBilinear

RECOGNITION = (
    [f"frames/IMG_0852/IMG_0852_{i:04d}.jpg" for i in (7, 8, 9, 10, 11, 12)]
    + [f"frames/IMG_0861/IMG_0861_{i:04d}.jpg" for i in (36, 40, 44, 48, 52)]
    + ["frames/IMG_0852/IMG_0852_0003.jpg", "frames/IMG_0861/IMG_0861_0001.jpg"]
)


def rotate(img, degrees):
    """Rotate about the centre with our own warp; returns the image and its valid mask."""
    h, w = img.shape[:2]
    t = np.deg2rad(degrees)
    c, s = np.cos(t), np.sin(t)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    H = np.array([[c, -s, cx - c * cx + s * cy], [s, c, cy - s * cx - c * cy], [0, 0, 1]])
    out = warpImageBilinear(img, H)
    return out[..., :3], out[..., 3]


def inside(pts, mask, margin=30):
    """Drop points within margin pixels of the mask edge (the black border of a rotated image)."""
    m = cv2.erode(mask.astype(np.uint8), np.ones((2 * margin + 1,) * 2, np.uint8))
    return m[pts[:, 1].astype(int), pts[:, 0].astype(int)] > 0


def run_pair(a, b, oriented, levels, keep_b=None):
    fa, fb = detect(gray(a), oriented=oriented, levels=levels), detect(gray(b), oriented=oriented, levels=levels)
    if keep_b is not None:
        k = keep_b(fb[0])
        fb = tuple(x[k] for x in fb)
    m, _ = match(fa[1], fb[1])
    p1, p2 = fa[0][m[:, 0]], fb[0][m[:, 1]]
    if len(m) < 4:
        return None, p1, p2, np.zeros(len(m), bool), fa, fb
    H, inl = ransac(p1, p2)
    return H, p1, p2, inl, fa, fb


def oriented_boxes(img, f, path, count=40, levels=False):
    """Draw each feature's 40x40 sampling window, rotated by its orientation (like MOPS figure 1)."""
    pts, _, ang, lvl = f
    fig, ax = viz.figure(img)
    colors = plt.cm.autumn(np.linspace(0, 1, max(lvl.max() + 1, 2)))
    step = max(1, len(pts) // count)
    for (x, y), t, l in list(zip(pts, ang, lvl))[::step]:
        half = 20 * 2 ** l
        c, s = np.cos(t), np.sin(t)
        corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
        rot = corners @ np.array([[c, s], [-s, c]]) + [x, y]
        col = colors[l] if levels else "yellow"
        ax.add_patch(mpatches.Polygon(rot, fill=False, ec=col, lw=1.5))
        ax.plot([x, x + half * c], [y, y + half * s], "-", color=col, lw=1.5)
    viz.save(fig, path)


def rotation(results):
    out = web_dir("b5")
    a, b = load_image(MOSAICS["canyon_slope"][0]), load_image(MOSAICS["canyon_slope"][1])
    b_rot, mask = rotate(b, 45)
    save_image(out / "rotation_input.jpg", b_rot)
    r = {}
    for label, oriented in (("axis", False), ("oriented", True)):
        H, p1, p2, inl, fa, fb = run_pair(a, b_rot, oriented, 1, lambda p: inside(p, mask))
        viz.matches(a, b_rot, p1, p2, out / f"rotation_{label}_ransac.jpg", inl)
        r[label] = dict(matches=len(p1), inliers=int(inl.sum()))
        if oriented:
            oriented_boxes(b_rot, fb, out / "rotation_oriented_patches.jpg")
            Hinv = np.linalg.inv(H)
            save_image(out / "rotation_mosaic.jpg", mosaic([a, np.dstack([b_rot, mask])], [np.eye(3), Hinv]))
    results["rotation"] = r
    print("rotation", r)


def multiscale(results):
    out = web_dir("b5")
    a, b = load_image(MOSAICS["canyon_ridge"][0]), load_image(MOSAICS["canyon_ridge"][1])
    b_small = cv2.resize(b, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    save_image(out / "multiscale_input.jpg", b_small)
    r = {}
    for label, levels in (("single", 1), ("pyramid", 3)):
        H, p1, p2, inl, fa, fb = run_pair(a, b_small, False, levels)
        viz.matches(a, b_small, p1, p2, out / f"multiscale_{label}_ransac.jpg", inl)
        r[label] = dict(matches=len(p1), inliers=int(inl.sum()))
        if levels > 1:
            oriented_boxes(a, fa, out / "multiscale_corners.jpg", count=60, levels=True)
            save_image(out / "multiscale_mosaic.jpg", mosaic([a, b_small], [np.eye(3), np.linalg.inv(H)]))
    results["multiscale"] = r
    print("multiscale", r)


def components(n, edges):
    seen, groups = set(), []
    for s in range(n):
        if s in seen:
            continue
        stack, group = [s], []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            group.append(u)
            stack += [v for (x, v) in edges if x == u] + [x for (x, v) in edges if v == u]
        groups.append(sorted(group))
    return groups


def to_reference(group, edges):
    """Homographies into the most connected image, grown along the strongest edges (max spanning tree)."""
    def w(u, v):
        return edges.get((u, v), edges.get((v, u), (0, None)))[0]

    def H(u, v):
        """Maps image u into image v."""
        return edges[(u, v)][1] if (u, v) in edges else np.linalg.inv(edges[(v, u)][1])

    ref = max(group, key=lambda u: sum(w(u, v) for v in group))
    Hs, done = {ref: np.eye(3)}, {ref}
    while len(done) < len(group):
        u, v = max(((u, v) for u in done for v in group if v not in done and w(u, v) > 0),
                   key=lambda e: w(*e))
        Hs[v] = Hs[u] @ H(v, u)
        done.add(v)
    return ref, Hs


def recognition(results):
    out = web_dir("b5")
    order = np.random.default_rng(3).permutation(len(RECOGNITION))
    files = [RECOGNITION[i] for i in order]
    ims = [load_image(f) for f in files]

    thumbs = [cv2.resize(im, (480, round(480 * im.shape[0] / im.shape[1])) if im.shape[1] > im.shape[0]
                         else (round(270 * im.shape[1] / im.shape[0]), 270)) for im in ims]
    fig, axes = plt.subplots(3, 5, figsize=(15, 7), dpi=120)
    for ax in axes.ravel():
        ax.set_axis_off()
    for i, (ax, t) in enumerate(zip(axes.ravel(), thumbs)):
        ax.imshow(t)
        ax.set_title(str(i + 1), fontsize=12)
    fig.tight_layout()
    viz.save(fig, out / "recognition_inputs.jpg")

    feats = [detect(gray(im)) for im in ims]
    n = len(ims)
    inl_count, edges = np.zeros((n, n), int), {}
    for i in range(n):
        for j in range(i + 1, n):
            m, _ = match(feats[i][1], feats[j][1])
            if len(m) < 8:
                continue
            H, inl = ransac(feats[i][0][m[:, 0]], feats[j][0][m[:, 1]], iters=1000)
            inl_count[i, j] = inl_count[j, i] = inl.sum()
            # Brown and Lowe: accept when inliers > 8 + 0.3 * matches
            if inl.sum() > 8 + 0.3 * len(m):
                edges[(i, j)] = (int(inl.sum()), H)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=130)
    im = ax.imshow(inl_count, cmap="viridis")
    for (i, j) in edges:
        for (r, c) in ((i, j), (j, i)):
            ax.text(c, r, "*", ha="center", va="center", color="white", fontsize=12)
    ax.set_xticks(range(n), [str(i + 1) for i in range(n)])
    ax.set_yticks(range(n), [str(i + 1) for i in range(n)])
    fig.colorbar(im, ax=ax, label="RANSAC inliers")
    fig.tight_layout()
    viz.save(fig, out / "recognition_graph.jpg")

    groups = components(n, edges)
    r = dict(files=files, inliers=inl_count.tolist(), groups=[[g + 1 for g in grp] for grp in groups])
    k = 0
    for grp in groups:
        if len(grp) < 2:
            continue
        k += 1
        _, Hs = to_reference(grp, edges)
        save_image(out / f"recognition_panorama_{k}.jpg", mosaic([ims[g] for g in grp], [Hs[g] for g in grp]))
    results["recognition"] = r
    print("recognition groups", r["groups"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("parts", nargs="*", default=["rotation", "multiscale", "recognition"])
    args = p.parse_args()
    path = DATA_DIR / "results_bells.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for part in args.parts:
        globals()[part](results)
    path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
