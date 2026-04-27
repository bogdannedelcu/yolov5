"""TF auto-anchor wrapper — reuses `utils/autoanchor.py` (PyTorch).

Builds a tiny stub object that quacks like a PT `Detect` module
(`.anchors` in cell units + `.stride`) and feeds it to `check_anchors`,
which computes BPR (Best Possible Recall) and runs `kmean_anchors` if it
falls below 0.98. The stub is mutated in-place; we read out the new
anchors and return them in the same YAML list format the loss expects.

This file contains zero re-implementation of the kmeans/BPR logic — only
the adapter glue between TF anchors (image-pixel YAML lists) and PT's
expectation of (nl, na, 2) torch tensors in cell units.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch

from utils.autoanchor import check_anchors


class _DetectStub:
    """Minimal duck-type for what `check_anchors` reads from `Detect`."""
    pass


class _ModelStub:
    pass


def _yaml_anchors_to_grid(anchors_yaml: Sequence[Sequence[float]]) -> np.ndarray:
    nl = len(anchors_yaml)
    na = len(anchors_yaml[0]) // 2
    return np.array(anchors_yaml, dtype=np.float32).reshape(nl, na, 2)


def _grid_to_yaml_anchors(anchors_grid: np.ndarray) -> List[List[float]]:
    """Inverse: (nl, na, 2) → [[w1,h1,w2,h2,...], ...]"""
    nl, na, _ = anchors_grid.shape
    out: List[List[float]] = []
    for i in range(nl):
        flat: List[float] = []
        for j in range(na):
            flat.extend([float(anchors_grid[i, j, 0]), float(anchors_grid[i, j, 1])])
        out.append(flat)
    return out


def maybe_recompute_anchors(
    dataset,
    anchors_yaml: Sequence[Sequence[float]],
    strides: Sequence[float],
    imgsz: int,
    thr: float = 4.0,
) -> Tuple[List[List[float]], bool]:
    """Run PT `check_anchors` on the TF-side anchors.

    Returns `(new_anchors_yaml, changed)`. If BPR is already > 0.98 or the
    kmeans result didn't improve BPR, returns the original YAML anchors.
    """
    nl = len(anchors_yaml)
    na = len(anchors_yaml[0]) // 2
    anchors_grid_px = _yaml_anchors_to_grid(anchors_yaml)        # (nl, na, 2) pixels
    strides_t = torch.tensor(list(strides), dtype=torch.float32)  # (nl,)

    # PT `check_anchors` expects `m.anchors` in cell units (it multiplies
    # by stride internally to get pixel-space anchors for the BPR metric,
    # then divides back if it overwrites them).
    anchors_cell = torch.from_numpy(anchors_grid_px).float() / strides_t.view(-1, 1, 1)

    stub = _DetectStub()
    stub.anchors = anchors_cell.clone()  # mutable, may be overwritten in-place
    stub.stride = strides_t

    model_stub = _ModelStub()
    model_stub.model = [stub]

    # check_anchors mutates stub.anchors in-place when it improves BPR.
    check_anchors(dataset, model_stub, thr=thr, imgsz=imgsz)

    new_grid_px = (stub.anchors * strides_t.view(-1, 1, 1)).numpy()
    changed = not np.allclose(new_grid_px, anchors_grid_px, atol=0.5)
    return _grid_to_yaml_anchors(new_grid_px), changed
