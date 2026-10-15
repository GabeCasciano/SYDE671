"""Shared helpers for the Assignment 2 scripts."""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WEB_DIR = Path(__file__).resolve().parents[2] / "website" / "images" / "assignment-2"

# Finite difference operators
DX = np.array([[1.0, 0.0, -1.0]])
DY = DX.T
D45 = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
D135 = np.fliplr(D45)


def load_image(path, gray=False):
    """Load an image as float32 in [0, 1], grayscale or RGB."""
    return np.asarray(Image.open(path).convert("L" if gray else "RGB"), dtype=np.float32) / 255.0


def save_image(path, img):
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path, quality=95)


def signed(img):
    """Map a signed image to [0, 1] with zero at mid grey."""
    return 0.5 + img / (2 * np.abs(img).max())


def web_dir(part):
    """Create and return WEB_DIR/<part>/."""
    path = WEB_DIR / part
    path.mkdir(parents=True, exist_ok=True)
    return path


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
