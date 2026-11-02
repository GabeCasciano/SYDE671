"""Part B: Harris + ANMS, descriptors, matching, RANSAC, and automatic mosaics.

Usage: python part_b.py [MOSAIC ...] [--ratio 0.5] [--n 500]
"""

import argparse
import json

import matplotlib.patches as mpatches
import numpy as np

import viz
from blend import chain_to_ref, mosaic
from common import DATA_DIR, MOSAICS, gray, load_image, save_image, web_dir
from features import anms, descriptors, harris, match, ransac, detect


def corner_figures(name, img, n):
    g = gray(img)
    pts, f = harris(g)
    strongest = pts[np.argsort(-f)[:n]]
    chosen, radii = anms(pts, f, n)
    out = web_dir("b1")
    viz.points(img, pts, out / f"{name}_harris_all.jpg", size=2)
    viz.points(img, strongest, out / f"{name}_strongest_{n}.jpg", size=10)
    viz.points(img, chosen, out / f"{name}_anms_{n}.jpg", size=10)
    return dict(harris=len(pts), anms_radius=float(radii.min()))


def descriptor_figures(name, img, n, count=12):
    g = gray(img)
    pts, f = harris(g)
    pts, _ = anms(pts, f, n)
    pts = pts[:count]
    d = descriptors(g, pts)
    out = web_dir("b2")

    fig, ax = viz.figure(img)
    for i, (x, y) in enumerate(pts):
        ax.add_patch(mpatches.Rectangle((x - 20, y - 20), 40, 40, fill=False, ec="yellow", lw=1.5))
        ax.text(x + 22, y - 22, str(i + 1), color="yellow", fontsize=11, fontweight="bold")
    viz.save(fig, out / f"{name}_locations.jpg")

    windows = [g[int(y) - 20:int(y) + 20, int(x) - 20:int(x) + 20] for x, y in pts]
    titles = [str(i + 1) for i in range(count)]
    viz.patch_grid(windows, out / f"{name}_windows.jpg", titles, cols=6)
    viz.patch_grid(d.reshape(-1, 8, 8), out / f"{name}_descriptors.jpg", titles, cols=6)


def stitch(name, n, ratio):
    files = MOSAICS[name]
    ims = [load_image(f) for f in files]
    feats = [detect(gray(im), n) for im in ims]
    pair_Hs, stats = [], []
    for i in range(len(ims) - 1):
        (pa, da, _, _), (pb, db, _, _) = feats[i], feats[i + 1]
        m, _ = match(da, db, ratio)
        p1, p2 = pa[m[:, 0]], pb[m[:, 1]]
        H, inl = ransac(p1, p2)
        pair_Hs.append(H)
        viz.matches(ims[i], ims[i + 1], p1, p2, web_dir("b3") / f"{name}_{i + 1}_matches.jpg")
        viz.matches(ims[i], ims[i + 1], p1, p2, web_dir("b4") / f"{name}_{i + 1}_ransac.jpg", inl)
        stats.append(dict(pair=[files[i], files[i + 1]], features=[len(pa), len(pb)],
                          matches=len(m), inliers=int(inl.sum()), H=H.tolist()))
    out = mosaic(ims, chain_to_ref(pair_Hs))
    save_image(web_dir("mosaics") / f"{name}_auto.jpg", out)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mosaics", nargs="*", default=list(MOSAICS))
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--n", type=int, default=500)
    args = p.parse_args()

    results_path = DATA_DIR / "results_part_b.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for name in args.mosaics:
        first = load_image(MOSAICS[name][0])
        r = corner_figures(name, first, args.n)
        descriptor_figures(name, first, args.n)
        r["pairs"] = stitch(name, args.n, args.ratio)
        results[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k != "pairs"}),
              [(s["matches"], s["inliers"]) for s in r["pairs"]])
    results_path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
