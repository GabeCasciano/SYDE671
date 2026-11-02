"""Part A: homographies from clicked points (A.2) and manual mosaics (A.4).

Usage: python part_a.py [MOSAIC ...]
Needs the points from click_points.py.
"""

import argparse
import json

import numpy as np

import viz
from blend import chain_to_ref, mosaic
from common import DATA_DIR, MOSAICS, load_image, load_points, pair_key, save_image, web_dir
from homography import apply_h, computeH, system


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mosaics", nargs="*", default=list(MOSAICS))
    args = p.parse_args()
    points = load_points()
    path = DATA_DIR / "results_part_a.json"
    results = json.loads(path.read_text()) if path.exists() else {}

    for name in args.mosaics:
        files = MOSAICS[name]
        keys = [pair_key(a, b) for a, b in zip(files, files[1:])]
        if any(k not in points for k in keys):
            print(f"{name}: missing clicked points, run click_points.py {name}")
            continue
        ims = [load_image(f) for f in files]
        for i, im in enumerate(ims):
            viz.save_scaled(web_dir("a1") / f"{name}_{i + 1}.jpg", im, 1200)

        pair_Hs, pairs = [], []
        for i, k in enumerate(keys):
            pa, pb = np.array(points[k]["a"]), np.array(points[k]["b"])
            H = computeH(pa, pb)
            A, b = system(pa, pb)
            err = np.linalg.norm(apply_h(H, pa) - pb, axis=1)
            viz.correspondences(ims[i], ims[i + 1], pa, pb, web_dir("a2") / f"{name}_{i + 1}_points.jpg")
            pair_Hs.append(H)
            pairs.append(dict(pair=[files[i], files[i + 1]], n=len(pa), A=A.tolist(), b=b.tolist(),
                              H=H.tolist(), rms=float(np.sqrt((err ** 2).mean()))))

        Hs = chain_to_ref(pair_Hs)
        out = web_dir("a4")
        save_image(web_dir("mosaics") / f"{name}_manual.jpg", mosaic(ims, Hs, "twoband"))
        save_image(out / f"{name}_overwrite.jpg", mosaic(ims, Hs, "overwrite"))
        save_image(out / f"{name}_feather.jpg", mosaic(ims, Hs, "feather"))
        results[name] = dict(pairs=pairs)
        print(name, [(q["n"], round(q["rms"], 2)) for q in pairs])

    path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
