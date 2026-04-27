"""TF-native YOLOv5 model assembly — mirror of `models/yolo.py` (PyTorch).

Provides:
- `TFDetect`: per-scale 1x1 Conv head emitting `na*(5+nc)` channels per cell.
  Returns a list of raw NHWC scale maps (no in-graph reshape/decode/concat).
- `parse_model_tf`: walks a YOLOv5 YAML (`backbone + head`) and constructs
  a `keras.Model` from `models.tf_common` blocks.
- `DetectionModelTF`: thin façade around the parsed Keras model exposing
  `model`, `nc`, `strides`, `anchors`, paralleling PT `DetectionModel`.
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import List, Sequence, Tuple

import yaml
import tensorflow as tf
from tensorflow import keras

from models.tf_common import (
    TFConv, TFC3, TFSPPF, TFConcat, TFUpsample,
    PTConvKernelInit, PTConvBiasInit,
    make_divisible,
)
# `MixConv2d` import is deferred to avoid circular import (tf_experimental
# imports DetectionModelTF). We import lazily inside parse_model_tf.


# ---------- detection head --------------------------------------------------

class TFDetect(keras.layers.Layer):
    """Per-scale 1x1 Conv2D head. Returns `nl` raw NHWC maps `[B, H_i, W_i, na*(5+nc)]`.

    Decoding (sigmoid + xywh + anchor multiply + NMS) is performed on the host
    so the TFLite/EdgeTPU graph stays static-shape and 100% TPU-mappable.
    """

    def __init__(self, nc, anchors, ch_in_per_scale, **kw):
        super().__init__(**kw)
        self.nc = nc
        self.no = nc + 5
        self.na = len(anchors[0]) // 2
        self.heads = [
            keras.layers.Conv2D(
                filters=self.no * self.na,
                kernel_size=1,
                strides=1,
                padding="same",
                use_bias=True,
                kernel_initializer=PTConvKernelInit(),
                bias_initializer=PTConvBiasInit(fan_in=ch_in_per_scale[i]),
                name=f"detect_p{i + 3}",
            )
            for i in range(len(ch_in_per_scale))
        ]

    def call(self, inputs):
        return [head(x) for head, x in zip(self.heads, inputs)]

    def initialize_biases(self, strides, img_w=640, cf=None):
        """Mirror PT `DetectionModel._initialize_biases` — pre-bias the obj logit
        toward "no object" and the cls logit toward a small prior. Critical for
        loss stability in the first epoch."""
        import numpy as np
        for head, s in zip(self.heads, strides):
            kernel, bias = head.get_weights()
            b = bias.reshape(self.na, -1).copy()
            b[:, 4] += math.log(8 / (img_w / s) ** 2)
            if self.nc > 1:
                b[:, 5:5 + self.nc] += (
                    math.log(0.6 / (self.nc - 0.999999))
                    if cf is None else np.log(cf / cf.sum())
                )
            else:
                b[:, 5:5 + self.nc] += math.log(0.6 / 0.4)
            head.set_weights([kernel, b.reshape(-1)])


# ---------- YAML parser -----------------------------------------------------

def _parse_yaml(yaml_path: Path | str) -> dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def _resolve_module_args(m_str, args, ch, f, gw, ch_mul, nc):
    """Mirror parse_model dim resolution from `models/yolo.py`."""
    out_ch = ch[-1]
    if m_str in ("Conv", "C3", "SPPF", "MixConv2d"):
        c1 = ch[f]
        c2 = args[0]
        c2 = make_divisible(c2 * gw, ch_mul)
        return [c1, c2, *args[1:]], c2
    if m_str == "Concat":
        c2 = sum(ch[-1 if x == -1 else x + 1] for x in f)
        return list(args), c2
    if m_str == "nn.Upsample":
        return list(args), ch[f]
    if m_str == "Detect":
        ch_in_per_scale = [ch[x + 1] for x in f]
        return [args[0], args[1], ch_in_per_scale], None
    raise NotImplementedError(f"module not supported: {m_str}")


def parse_model_tf(
    cfg: dict,
    img_hw: Tuple[int, int],
    nc_override: int | None = None,
    act: str = "silu",
    batch_size: int = 1,
    dynamic_shape: bool = False,
) -> Tuple[keras.Model, int, List[float], TFDetect]:
    """Walk `cfg["backbone"] + cfg["head"]` and produce a Keras model.

    Returns `(model, nc, strides, detect_module)`. Mirrors the role of
    `models.yolo.parse_model` in PyTorch.
    """
    H, W = img_hw
    cfg = deepcopy(cfg)
    nc = nc_override if nc_override is not None else cfg["nc"]
    cfg["nc"] = nc
    anchors = cfg["anchors"]
    gd = cfg.get("depth_multiple", 1.0)
    gw = cfg.get("width_multiple", 1.0)
    ch_mul = cfg.get("channel_multiple", 8)

    if dynamic_shape:
        inputs = keras.Input(shape=(None, None, 3), batch_size=batch_size, name="images")
    else:
        inputs = keras.Input(shape=(H, W, 3), batch_size=batch_size, name="images")
    ch: list = [3]
    layer_outs: list = []
    detect_module = None

    layers_def = cfg["backbone"] + cfg["head"]
    x = inputs
    eval_ns = {"nc": nc, "anchors": anchors}

    def _maybe_eval(a):
        if not isinstance(a, str):
            return a
        try:
            return eval(a, {"__builtins__": {}}, eval_ns)
        except (NameError, SyntaxError):
            return a

    for i, (f, n_rep, m_str, args) in enumerate(layers_def):
        n_rep = max(round(n_rep * gd), 1) if n_rep > 1 else n_rep
        args = [_maybe_eval(a) for a in args]
        new_args, _out_ch = _resolve_module_args(m_str, args, ch, f, gw, ch_mul, nc)

        if isinstance(f, int):
            x_in = x if f == -1 else layer_outs[f]
        else:
            x_in = [x if j == -1 else layer_outs[j] for j in f]

        if m_str == "Conv":
            c1, c2, *rest = new_args
            k = rest[0] if len(rest) > 0 else 1
            s = rest[1] if len(rest) > 1 else 1
            p = rest[2] if len(rest) > 2 else None
            layer = TFConv(c2=c2, k=k, s=s, p=p, act=act, name=f"L{i}_Conv")
            x = layer(x_in)
        elif m_str == "C3":
            c1, c2, *rest = new_args
            shortcut = rest[0] if len(rest) > 0 else True
            layer = TFC3(c1=c1, c2=c2, n=n_rep, shortcut=shortcut, act=act, name=f"L{i}_C3")
            x = layer(x_in)
        elif m_str == "SPPF":
            c1, c2, *rest = new_args
            k = rest[0] if len(rest) > 0 else 5
            layer = TFSPPF(c1=c1, c2=c2, k=k, act=act, name=f"L{i}_SPPF")
            x = layer(x_in)
        elif m_str == "nn.Upsample":
            _size, scale_factor, mode = new_args[0], new_args[1], new_args[2]
            layer = TFUpsample(scale_factor=scale_factor, mode=mode,
                               dynamic=dynamic_shape, name=f"L{i}_Upsample")
            x = layer(x_in)
        elif m_str == "MixConv2d":
            from models.tf_experimental import TFMixConv2d
            c1, c2, *rest = new_args
            k = rest[0] if len(rest) > 0 else (1, 3)
            s = rest[1] if len(rest) > 1 else 1
            layer = TFMixConv2d(c1=c1, c2=c2, k=tuple(k), s=s, act=act,
                                name=f"L{i}_MixConv2d")
            x = layer(x_in)
        elif m_str == "Concat":
            layer = TFConcat(name=f"L{i}_Concat")
            x = layer(x_in)
        elif m_str == "Detect":
            nc_in, anchors_in, ch_in_per_scale = new_args
            detect_module = TFDetect(nc=nc_in, anchors=anchors_in,
                                     ch_in_per_scale=ch_in_per_scale, name=f"L{i}_Detect")
            outs = detect_module(x_in)
            layer_outs.append(outs)
            ch.append(None)
            break
        else:
            raise NotImplementedError(m_str)

        layer_outs.append(x)
        ch.append(_out_ch)

    if detect_module is None:
        raise ValueError("No Detect layer found in YAML")

    outputs = layer_outs[-1]
    if dynamic_shape:
        # cannot infer strides from output shape (None); compute via probe at H,W
        probe = parse_model_tf(cfg, img_hw, nc_override=nc_override, act=act,
                               batch_size=1, dynamic_shape=False)
        strides = probe[2]  # (model, nc, strides, detect)
    else:
        strides = [float(H) / float(o.shape[1]) for o in outputs]

    named = []
    for j, o in enumerate(outputs):
        named.append(keras.layers.Layer(name=f"raw_p{j + 3}")(o))
    model = keras.Model(inputs=inputs, outputs=named, name=cfg.get("_name", "yolov5_tf"))

    detect_module.initialize_biases(strides=strides, img_w=W)
    return model, nc, strides, detect_module


# ---------- model facade ----------------------------------------------------

class DetectionModelTF:
    """Thin facade around the parsed Keras model. Mirror of PT `DetectionModel`.

    Exposes: `model` (keras.Model), `nc`, `strides` (list), `anchors`
    (raw YAML, image-pixel units), `detect` (the TFDetect layer instance).

    Usage:
        m = DetectionModelTF(cfg='models/yolov5n.yaml', imgsz_hw=(640, 640))
        preds = m(x)             # list of raw scale maps
    """

    def __init__(
        self,
        cfg: str | dict,
        imgsz_hw: Tuple[int, int] = (640, 640),
        nc: int | None = None,
        act: str = "silu",
        batch_size: int = 1,
        dynamic_shape: bool = False,
    ):
        if isinstance(cfg, str):
            self.cfg_path = str(cfg)
            cfg_dict = _parse_yaml(cfg)
            cfg_dict["_name"] = Path(cfg).stem + "_tf"
        else:
            self.cfg_path = None
            cfg_dict = cfg
        self.yaml = deepcopy(cfg_dict)
        self.anchors = cfg_dict["anchors"]
        self.imgsz_hw = imgsz_hw
        self.act = act
        self.dynamic_shape = dynamic_shape
        self.model, self.nc, self.strides, self.detect = parse_model_tf(
            cfg_dict, img_hw=imgsz_hw, nc_override=nc, act=act,
            batch_size=batch_size, dynamic_shape=dynamic_shape,
        )

    def __call__(self, x, training=False):
        return self.model(x, training=training)

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    @property
    def variables(self):
        return self.model.variables

    def save_weights(self, path):
        self.model.save_weights(str(path))

    def load_weights(self, path):
        self.model.load_weights(str(path))

    def save(self, path):
        self.model.save(str(path))


# ---------- back-compat shim ------------------------------------------------

def build_tf_model_from_yaml(yaml_path, img_hw, nc_override=None, act="silu",
                              batch_size=1):
    """Legacy entry point used by the older script names. Prefer `DetectionModelTF`."""
    cfg = _parse_yaml(yaml_path)
    cfg["_name"] = Path(yaml_path).stem + "_tf"
    model, nc, strides, _detect = parse_model_tf(
        cfg, img_hw=img_hw, nc_override=nc_override, act=act, batch_size=batch_size,
    )
    return model, nc, strides
