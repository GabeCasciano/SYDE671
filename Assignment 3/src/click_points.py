"""Click correspondences and save them to data/points.json.

Mosaic pairs:   python click_points.py [MOSAIC ...] [--redo]
  Each consecutive pair is shown side by side. Click a point on the left image, then the
  same point on the right image, and repeat (at least 6 pairs). Use the toolbar to zoom,
  then turn zoom off to click again.
Rectification:  python click_points.py --rect photos/sign.jpg --size 60x40 [--redo]
  Click the four corners of the rectangle in order: top-left, top-right, bottom-right,
  bottom-left. --size is the real width x height (any units; only the ratio matters).

Keys: u = undo last click, Enter = done with this image or pair.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common import MOSAICS, load_image, load_points, pair_key, save_points

COLORS = plt.cm.tab10.colors


def collect(images, title, alternate):
    """Show images side by side and collect clicks; returns one point list per image."""
    offsets = np.cumsum([0] + [im.shape[1] for im in images])
    h = max(im.shape[0] for im in images)
    canvas = np.ones((h, offsets[-1], 3), np.float32)
    for im, x in zip(images, offsets):
        canvas[:im.shape[0], x:x + im.shape[1]] = im

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(canvas)
    ax.set_axis_off()
    fig.tight_layout()
    clicks, artists = [], []

    def status():
        side = "left" if len(clicks) % 2 == 0 else "right"
        hint = f"click the {side} image" if alternate else f"{len(clicks)} of 4 corners"
        ax.set_title(f"{title}: {hint}  (u = undo, Enter = done)")
        fig.canvas.draw_idle()

    def on_click(e):
        if e.inaxes is not ax or fig.canvas.toolbar.mode or e.button != 1:
            return
        i = np.searchsorted(offsets, e.xdata, side="right") - 1
        if alternate and i != len(clicks) % 2:
            print("Click the other image first")
            return
        n = len(clicks) // 2 if alternate else len(clicks)
        c = COLORS[n % len(COLORS)]
        clicks.append((i, e.xdata - offsets[i], e.ydata))
        artists.append([ax.plot(e.xdata, e.ydata, "+", color=c, ms=14, mew=2)[0],
                        ax.text(e.xdata + 8, e.ydata - 8, str(n + 1), color=c, fontsize=12,
                                fontweight="bold")])
        status()

    def on_key(e):
        if e.key == "u" and clicks:
            clicks.pop()
            for a in artists.pop():
                a.remove()
            status()
        elif e.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    status()
    plt.show()
    return [[[x, y] for j, x, y in clicks if j == i] for i in range(len(images))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mosaics", nargs="*", default=list(MOSAICS))
    p.add_argument("--rect")
    p.add_argument("--size", default="1x1")
    p.add_argument("--redo", action="store_true")
    args = p.parse_args()
    points = load_points()

    if args.rect:
        key = f"rect|{args.rect}"
        if key in points and not args.redo:
            print(f"{key} already clicked (use --redo)")
            return
        w, h = (float(v) for v in args.size.split("x"))
        pts = collect([load_image(args.rect)], args.rect, alternate=False)[0]
        if len(pts) != 4:
            print(f"Need exactly 4 corners, got {len(pts)}; nothing saved")
            return
        points[key] = {"corners": pts, "size": [w, h]}
        save_points(points)
        return

    for name in args.mosaics:
        files = MOSAICS[name]
        for a, b in zip(files, files[1:]):
            key = pair_key(a, b)
            if key in points and not args.redo:
                print(f"{key} already clicked (use --redo)")
                continue
            pa, pb = collect([load_image(a), load_image(b)], f"{name}: {a} | {b}", alternate=True)
            n = min(len(pa), len(pb))
            if n < 6:
                print(f"Need at least 6 pairs, got {n}; nothing saved for {key}")
                continue
            points[key] = {"a": pa[:n], "b": pb[:n]}
            save_points(points)
            print(f"Saved {n} pairs for {key}")


if __name__ == "__main__":
    main()
