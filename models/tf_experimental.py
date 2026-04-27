"""TF-native counterparts to `models/experimental.py` (PyTorch).

Provides:
- `TFMixConv2d`: TF port of `MixConv2d` (mixed-kernel depthwise+conv, used by
  some YAML variants, e.g. yolov5n6/yolov5s6 PAN paths).
- `TFEnsemble`: average outputs of N `DetectionModelTF` instances.
- `attempt_load_tf`: helper to rebuild a TF model from weights + sidecar
  `architecture.json` (mirror of PT `attempt_load`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf
from tensorflow import keras

from models.tf_common import TFPad, autopad, PTConvKernelInit, act_layer
from models.tf_yolo import DetectionModelTF


# ---------- MixConv2d -------------------------------------------------------

class TFMixConv2d(keras.layers.Layer):
    """Mixed depth-wise convolutions (https://arxiv.org/abs/1907.09595).

    Splits c2 into len(k) groups; each group is a Conv with its own kernel size.
    Outputs are concatenated and passed through BN + SiLU. Mirror of PT
    `models.experimental.MixConv2d`.

    For TF/EdgeTPU compatibility we use plain `Conv2D` for each branch
    (PyTorch uses depthwise via `groups=gcd(c1, c_)`); on EdgeTPU dense convs
    typically map cleaner than depthwise mixed convs anyway.
    """

    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True, act="silu", **kw):
        super().__init__(**kw)
        n = len(k)
        if equal_ch:
            idx = np.floor(np.linspace(0, n - 1e-6, c2))
            c_ = [int((idx == g).sum()) for g in range(n)]
        else:
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int).tolist()

        self.branches = []
        for ki, ci in zip(k, c_):
            if ci == 0:
                continue
            pad = autopad(int(ki))
            if s == 1:
                conv = keras.layers.Conv2D(
                    filters=int(ci), kernel_size=int(ki), strides=1,
                    padding="same", use_bias=False,
                    kernel_initializer=PTConvKernelInit(),
                )
                self.branches.append(("conv", conv))
            else:
                p = TFPad(pad)
                conv = keras.layers.Conv2D(
                    filters=int(ci), kernel_size=int(ki), strides=s,
                    padding="valid", use_bias=False,
                    kernel_initializer=PTConvKernelInit(),
                )
                self.branches.append(("pad_conv", p, conv))
        self.bn = keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.97)
        self.act = act_layer(act)

    def call(self, x, training=False):
        outs = []
        for b in self.branches:
            if b[0] == "conv":
                outs.append(b[1](x))
            else:
                outs.append(b[2](b[1](x)))
        y = tf.concat(outs, axis=-1)
        return self.act(self.bn(y, training=training))


# ---------- Ensemble --------------------------------------------------------

class TFEnsemble:
    """Ensemble of `DetectionModelTF` instances. Inference outputs are
    concatenated along the anchor axis (mirror of PT `Ensemble` "nms ensemble").

    For raw multi-output TF models, ensemble is performed at the host
    decode stage by concatenating per-scale predictions across members.
    """

    def __init__(self, models: Sequence[DetectionModelTF]):
        self.models = list(models)
        if not self.models:
            raise ValueError("empty ensemble")
        self.nc = self.models[0].nc
        self.strides = self.models[0].strides
        self.anchors = self.models[0].anchors
        for m in self.models:
            assert m.nc == self.nc, "ensemble: nc mismatch"

    def __call__(self, x, training=False):
        # Returns a list aligned with self.strides, each entry is the
        # per-scale concat across members along the channel axis.
        per_scale = list(zip(*[m(x, training=training) for m in self.models]))
        # Average outputs across members for a single concatenated map per scale.
        out = []
        for scale_outs in per_scale:
            out.append(tf.add_n(scale_outs) / float(len(scale_outs)))
        return out


# ---------- attempt_load_tf -------------------------------------------------

def attempt_load_tf(weights, imgsz_hw=None, batch_size=1):
    """Load one or more TF detector(s) from `.weights.h5` files via sidecars.

    `weights`: a single path or a list of paths. Each path should have an
    `architecture.json` next to it (written by `train_tf.py`).
    Returns a `DetectionModelTF` (single) or a `TFEnsemble` (list of >1).
    """
    if not isinstance(weights, (list, tuple)):
        weights = [weights]
    models = []
    for w in weights:
        wp = Path(w)
        sidecar = wp.parent / "architecture.json"
        if not sidecar.exists():
            raise FileNotFoundError(
                f"architecture.json not found next to {wp} — pass --cfg/--imgsz instead"
            )
        sc = json.loads(sidecar.read_text())
        cfg = sc["cfg"]
        sc_imgsz = tuple(sc["imgsz_hw"]) if imgsz_hw is None else tuple(imgsz_hw)
        nc = sc.get("nc")
        act = sc.get("act", "silu")
        m = DetectionModelTF(cfg=cfg, imgsz_hw=sc_imgsz, nc=nc, act=act, batch_size=batch_size)
        m.load_weights(wp)
        models.append(m)

    if len(models) == 1:
        return models[0]
    return TFEnsemble(models)
