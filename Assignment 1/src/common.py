"""Shared helpers for the Assignment 1 alignment scripts."""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_plate(path):
    """Load a glass plate scan as grayscale float32 in [0, 1]."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def split_plate(plate):
    """Split a plate into (b, g, r) channels, stacked top to bottom."""
    h = plate.shape[0] // 3
    return plate[:h], plate[h:2 * h], plate[2 * h:3 * h]


def crop(img, frac):
    """Remove frac of the height and width from each side."""
    h, w = img.shape[:2]
    dy, dx = int(h * frac), int(w * frac)
    return img[dy:h - dy, dx:w - dx]


def shift(img, d):
    """Circularly shift img by d = (dx, dy)."""
    dx, dy = d
    return np.roll(img, (dy, dx), axis=(0, 1))


def save_image(path, img):
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path, quality=95)


def find_images(pattern):
    """Sorted image paths in DATA_DIR matching comma-separated globs or names."""
    paths = set()
    for p in pattern.split(","):
        paths.update(DATA_DIR.glob(p.strip()))
    return sorted(paths)


def parse_args(defaults):
    """One --flag per default; the type comes from the default value."""
    parser = argparse.ArgumentParser()
    for name, value in defaults.items():
        parser.add_argument(f"--{name}", type=type(value), default=value)
    return vars(parser.parse_args())


def run_dir(script_file, params):
    """Create data/<script>/<timestamp>_<params>/ and return it."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tags = [f"{k}-{v}".replace(",", "+") for k, v in params.items() if k != "images"]
    path = DATA_DIR / Path(script_file).stem / "_".join([stamp] + tags)
    path.mkdir(parents=True)
    return path
