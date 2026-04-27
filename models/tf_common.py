"""TF-native NHWC building blocks — layout mirror of `models/common.py` (PyTorch).

Layers in this module are pure TF/Keras (no PyTorch dependency, no torch
weight transfer). They are consumed by `models/tf_yolo.py:parse_model_tf` to
build a Keras detector from any standard YOLOv5 YAML.

Design choices for EdgeTPU compatibility:
- Conv stride>1 uses an explicit symmetric `TFPad` (PyTorch-parity padding)
  followed by `padding="valid"` Conv2D — avoids TF "same" asymmetric padding.
- Upsample uses `tf.image.resize(..., 'nearest')` (lowers to
  `RESIZE_NEAREST_NEIGHBOR`), NOT `keras.layers.UpSampling2D` which on TF≥2.19
  lowers to a 5D EXPAND_DIMS/TILE pattern that fails edgetpu_compiler v16.
- Conv kernel/bias init replicates PyTorch defaults
  (`kaiming_uniform_(a=sqrt(5))`) for parity in initial weight magnitudes.
"""

from __future__ import annotations

import math

import tensorflow as tf
from tensorflow import keras


# ---------- helpers ----------------------------------------------------------

def autopad(k, p=None):
    """Mirror `models.common.autopad`: pad to keep shape on stride 1."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def make_divisible(x, divisor=8):
    return int(math.ceil(x / divisor) * divisor)


class PTConvKernelInit(keras.initializers.Initializer):
    """Replicate PyTorch nn.Conv2d default init: kaiming_uniform_(a=sqrt(5)).

    PyTorch's effective bound is `1/sqrt(fan_in)`. Keras default is
    `glorot_uniform` with `sqrt(6/(fan_in + fan_out))` — different scale.
    """

    def __call__(self, shape, dtype=None):
        fan_in = int(shape[0]) * int(shape[1]) * int(shape[2])
        limit = (1.0 / fan_in) ** 0.5
        return tf.random.uniform(shape, -limit, limit, dtype=dtype)

    def get_config(self):
        return {}


class PTConvBiasInit(keras.initializers.Initializer):
    """PyTorch Conv2d default bias init: uniform `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`."""

    def __init__(self, fan_in: int):
        self.fan_in = int(fan_in)

    def __call__(self, shape, dtype=None):
        limit = (1.0 / self.fan_in) ** 0.5
        return tf.random.uniform(shape, -limit, limit, dtype=dtype)

    def get_config(self):
        return {"fan_in": self.fan_in}


def act_layer(name: str):
    """Resolve activation by name: silu/swish, relu, relu6, leaky, linear."""
    name = (name or "silu").lower()
    if name in ("silu", "swish"):
        return keras.layers.Activation(keras.activations.swish)
    if name == "relu":
        return keras.layers.ReLU()
    if name == "relu6":
        return keras.layers.ReLU(max_value=6.0)
    if name == "leaky":
        return keras.layers.LeakyReLU(0.1)
    if name in ("linear", "none"):
        return keras.layers.Activation("linear")
    raise ValueError(f"unsupported activation: {name}")


# ---------- core layers ------------------------------------------------------

class TFPad(keras.layers.Layer):
    """Symmetric spatial pad for stride>1 convs (PyTorch-parity)."""

    def __init__(self, pad, **kw):
        super().__init__(**kw)
        if isinstance(pad, int):
            self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
        else:
            self.pad = tf.constant([[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]])

    def call(self, x):
        return tf.pad(x, self.pad, mode="constant")


class TFConv(keras.layers.Layer):
    """Conv2D + BN + activation, NHWC native, with PyTorch-parity padding."""

    def __init__(self, c2, k=1, s=1, p=None, act="silu", **kw):
        super().__init__(**kw)
        self.s = s
        self.k = k
        kinit = PTConvKernelInit()
        if s == 1:
            self.pad = None
            self.conv = keras.layers.Conv2D(
                filters=c2, kernel_size=k, strides=1, padding="same",
                use_bias=False, kernel_initializer=kinit,
            )
        else:
            self.pad = TFPad(autopad(k, p))
            self.conv = keras.layers.Conv2D(
                filters=c2, kernel_size=k, strides=s, padding="valid",
                use_bias=False, kernel_initializer=kinit,
            )
        # PT BN: eps=1e-3, momentum=0.03 ↔ Keras BN momentum=0.97
        self.bn = keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.97)
        self.act = act_layer(act)

    def call(self, x, training=False):
        if self.pad is not None:
            x = self.pad(x)
        return self.act(self.bn(self.conv(x), training=training))


class TFBottleneck(keras.layers.Layer):
    def __init__(self, c1, c2, shortcut=True, e=0.5, act="silu", **kw):
        super().__init__(**kw)
        c_ = int(c2 * e)
        self.cv1 = TFConv(c_, 1, 1, act=act)
        self.cv2 = TFConv(c2, 3, 1, act=act)
        self.add = shortcut and c1 == c2

    def call(self, x, training=False):
        y = self.cv2(self.cv1(x, training=training), training=training)
        return x + y if self.add else y


class TFC3(keras.layers.Layer):
    """CSP bottleneck with 3 convolutions — TF mirror of `models.common.C3`."""

    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, act="silu", **kw):
        super().__init__(**kw)
        c_ = int(c2 * e)
        self.cv1 = TFConv(c_, 1, 1, act=act)
        self.cv2 = TFConv(c_, 1, 1, act=act)
        self.cv3 = TFConv(c2, 1, 1, act=act)
        self.m = [TFBottleneck(c_, c_, shortcut, e=1.0, act=act) for _ in range(n)]

    def call(self, x, training=False):
        y1 = self.cv1(x, training=training)
        for b in self.m:
            y1 = b(y1, training=training)
        y2 = self.cv2(x, training=training)
        return self.cv3(tf.concat([y1, y2], axis=-1), training=training)


class TFSPPF(keras.layers.Layer):
    """SPPF: 1x1 reduce → 3 chained MaxPool(stride=1, same) → concat → 1x1 expand."""

    def __init__(self, c1, c2, k=5, act="silu", **kw):
        super().__init__(**kw)
        c_ = c1 // 2
        self.cv1 = TFConv(c_, 1, 1, act=act)
        self.cv2 = TFConv(c2, 1, 1, act=act)
        self.mp = keras.layers.MaxPool2D(pool_size=k, strides=1, padding="same")

    def call(self, x, training=False):
        x = self.cv1(x, training=training)
        y1 = self.mp(x)
        y2 = self.mp(y1)
        y3 = self.mp(y2)
        return self.cv2(tf.concat([x, y1, y2, y3], axis=-1), training=training)


class TFConcat(keras.layers.Layer):
    """Channel concat in NHWC (always axis=-1; YAML uses dim=1 in NCHW)."""

    def call(self, inputs):
        return tf.concat(inputs, axis=-1)


class TFUpsample(keras.layers.Layer):
    """Nearest-neighbor upsample via `tf.image.resize` → `RESIZE_NEAREST_NEIGHBOR`.

    Avoid `keras.layers.UpSampling2D` (TF≥2.19 lowers it to TILE which fails
    `edgetpu_compiler` v16).

    `dynamic`: when True, target H/W are computed from `tf.shape(x)` at runtime,
    enabling variable-shape input (multi-scale training). Default False keeps
    shape static for clean Edge TPU export. Only enable for training; rebuild
    the model with dynamic=False before export.
    """

    def __init__(self, scale_factor=2, mode="nearest", dynamic=False, **kw):
        super().__init__(**kw)
        self.scale = scale_factor
        self.mode = mode
        self.use_dynamic_shape = bool(dynamic)

    def call(self, x):
        if self.use_dynamic_shape:
            sh = tf.shape(x)
            h = sh[1] * self.scale
            w = sh[2] * self.scale
        else:
            h = x.shape[1] * self.scale
            w = x.shape[2] * self.scale
        return tf.image.resize(x, (h, w), method=self.mode)
