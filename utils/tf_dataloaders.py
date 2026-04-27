"""TF data utilities for YOLOv5-style YOLO-format datasets.

Mirrors the layout of `utils/dataloaders.py` (PyTorch) but for the TF native
training/export path. Functions here are used by:
- `train_tf.py` (calibration set + sidecar info)
- `export_tf.py` (INT8 calibration via real images)

The image-loading pipeline reuses `utils.augmentations.letterbox` from the PT
side — same letterboxing math, same stride-aware behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

from utils.augmentations import letterbox


IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_data_yaml(path: str | Path) -> dict:
    p = Path(path)
    with open(p, "r") as f:
        d = yaml.safe_load(f)
    base = Path(d.get("path", str(p.parent)))
    if not base.is_absolute():
        base = (p.parent / base).resolve()
    out = {
        "path": base,
        "train": (base / d["train"]).resolve() if "train" in d else None,
        "val": (base / d["val"]).resolve() if "val" in d else None,
        "nc": d.get("nc") or len(d.get("names", {})) or 1,
        "names": d.get("names"),
    }
    return out


def list_dataset(images_dir: Path) -> Tuple[List[Path], List[Path]]:
    images_dir = Path(images_dir)
    imgs: list[Path] = []
    for ext in IMG_EXT:
        imgs.extend(images_dir.rglob(f"*{ext}"))
    imgs.sort()
    labels: list[Path] = []
    for ip in imgs:
        sp = str(ip)
        if "/images/" in sp:
            lp = Path(sp.replace("/images/", "/labels/")).with_suffix(".txt")
        else:
            lp = ip.parent.parent / "labels" / (ip.stem + ".txt")
        labels.append(lp)
    return imgs, labels


def _read_image(path: Path) -> np.ndarray:
    import cv2
    im = cv2.imread(str(path))
    if im is None:
        raise RuntimeError(f"failed to read {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def build_calibration_generator(
    images_dir: str | Path,
    imgsz_hw: Tuple[int, int],
    n_calib: int = 100,
    seed: int = 0,
):
    """Generator yielding [1,H,W,3] float32 (in [0,1]) for TFLite INT8 calibration.

    Reads up to `n_calib` real images from `images_dir` (recursive), letterboxes
    them to `imgsz_hw` using the PT-side `letterbox`, and emits batches of size
    1. Falls back to random uint8 noise if the directory is empty.
    """
    rng = np.random.default_rng(seed)
    H, W = imgsz_hw
    try:
        imgs, _labels = list_dataset(Path(images_dir))
        if not imgs:
            raise RuntimeError(f"empty dir {images_dir}")
        idxs = list(range(len(imgs)))
        rng.shuffle(idxs)
        idxs = idxs[:n_calib]
    except Exception:
        imgs = []
        idxs = []

    def gen():
        if idxs:
            for i in idxs:
                im = _read_image(imgs[i])
                im, _r, _pad = letterbox(im, new_shape=(H, W), auto=False, scaleup=True)
                x = im[None, ...].astype(np.float32) / 255.0
                yield [x]
        else:
            for _ in range(n_calib):
                x = rng.integers(0, 256, size=(1, H, W, 3), dtype=np.uint8).astype(np.float32) / 255.0
                yield [x]

    return gen
