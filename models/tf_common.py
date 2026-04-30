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

import numpy as np
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


class TFC2f(keras.layers.Layer):
    """C2f from YOLOv8 — split + N bottlenecks each chained, all concatenated.

    Drop-in replacement for C3. Better gradient flow (each bottleneck output
    is concatenated, not just the final) at marginally more channel ops.
    """

    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, act="silu", **kw):
        super().__init__(**kw)
        self.c_ = int(c2 * e)
        self.cv1 = TFConv(2 * self.c_, 1, 1, act=act)
        self.cv2 = TFConv(c2, 1, 1, act=act)
        self.m = [TFBottleneck(self.c_, self.c_, shortcut, e=1.0, act=act) for _ in range(n)]

    def call(self, x, training=False):
        y = tf.split(self.cv1(x, training=training), 2, axis=-1)
        y = list(y)
        for b in self.m:
            y.append(b(y[-1], training=training))
        return self.cv2(tf.concat(y, axis=-1), training=training)


class TFGELAN(keras.layers.Layer):
    """GELAN block (YOLOv9) — Generalized Efficient Layer Aggregation.

    Same shape as ELAN/CSP but with cleaner gradient pathways: split input
    into 2 halves, run a chain of bottlenecks on the second half collecting
    intermediate features, concat all (initial halves + each intermediate)
    and project. EdgeTPU-friendly: pure Conv + Concat + Add (via shortcut).
    """

    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, act="silu", **kw):
        super().__init__(**kw)
        self.c_ = int(c2 * e)
        self.cv1 = TFConv(2 * self.c_, 1, 1, act=act)
        # Two chained bottlenecks per "step", n steps. Equivalent to
        # GELAN's "block" wrapping ELAN's gradient flow.
        self.m = [
            [TFBottleneck(self.c_, self.c_, shortcut, e=1.0, act=act) for _ in range(2)]
            for _ in range(n)
        ]
        self.cv2 = TFConv(c2, 1, 1, act=act)

    def call(self, x, training=False):
        y = list(tf.split(self.cv1(x, training=training), 2, axis=-1))
        cur = y[-1]
        for b1, b2 in self.m:
            cur = b2(b1(cur, training=training), training=training)
            y.append(cur)
        return self.cv2(tf.concat(y, axis=-1), training=training)


class TFPConv(keras.layers.Layer):
    """Partial Convolution (FasterNet, CVPR 2023).

    Operates conv only on the first `r` channels (default r = c/4),
    pass-through the rest. ~2× faster than full Conv at small accuracy
    cost. EdgeTPU-friendly (just Conv + Concat).
    """

    def __init__(self, c2, k=3, s=1, ratio=0.25, act="silu", **kw):
        super().__init__(**kw)
        self.c2 = c2
        self.ratio = ratio
        self.k = k
        self.s = s
        self.act_name = act
        self.conv = None  # built lazily after we know c1

    def build(self, input_shape):
        c1 = int(input_shape[-1])
        self.c_partial = max(int(c1 * self.ratio), 1)
        self.c_passthrough = c1 - self.c_partial
        if self.s == 1:
            self.pad = None
            self.conv = keras.layers.Conv2D(
                filters=self.c_partial, kernel_size=self.k, strides=1,
                padding="same", use_bias=False, kernel_initializer=PTConvKernelInit(),
            )
        else:
            self.pad = TFPad(autopad(self.k))
            self.conv = keras.layers.Conv2D(
                filters=self.c_partial, kernel_size=self.k, strides=self.s,
                padding="valid", use_bias=False, kernel_initializer=PTConvKernelInit(),
            )
        self.bn = keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.97)
        self.act = act_layer(self.act_name)
        # final 1x1 to mix back to c2 channels (FasterNet uses two 1x1 around PConv)
        self.proj = TFConv(self.c2, 1, 1, act=self.act_name)
        super().build(input_shape)

    def call(self, x, training=False):
        x_p = x[..., : self.c_partial]
        x_rest = x[..., self.c_partial:]
        if self.pad is not None:
            x_p = self.pad(x_p)
        x_p = self.act(self.bn(self.conv(x_p), training=training))
        # at stride > 1, x_rest must also be downsampled to match — fall back
        # to strided slice (simple subsampling) for EdgeTPU compatibility
        if self.s > 1:
            x_rest = x_rest[:, :: self.s, :: self.s, :]
        y = tf.concat([x_p, x_rest], axis=-1)
        return self.proj(y, training=training)


class TFGhost(keras.layers.Layer):
    """Ghost module (GhostNet, CVPR 2020) — produce more features cheaply.

    Half output channels via standard 1×1 Conv, the other half via cheap
    DepthwiseConv on the first half (the "ghosts"). EdgeTPU-friendly.
    """

    def __init__(self, c2, k=1, dw_size=3, ratio=2, act="silu", **kw):
        super().__init__(**kw)
        self.c2 = c2
        self.k = k
        self.dw_size = dw_size
        self.ratio = ratio
        self.act_name = act

    def build(self, input_shape):
        init_ch = self.c2 // self.ratio
        new_ch = self.c2 - init_ch
        self.primary = TFConv(init_ch, k=self.k, s=1, act=self.act_name)
        self.cheap = keras.Sequential([
            keras.layers.DepthwiseConv2D(
                kernel_size=self.dw_size, strides=1, padding="same",
                use_bias=False, depthwise_initializer=PTConvKernelInit(),
            ),
            keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.97),
            act_layer(self.act_name),
        ])
        self.new_ch = new_ch
        super().build(input_shape)

    def call(self, x, training=False):
        x1 = self.primary(x, training=training)
        x2 = self.cheap(x1, training=training)[..., : self.new_ch]
        return tf.concat([x1, x2], axis=-1)


class TFMSBlock(keras.layers.Layer):
    """MS-Block (YOLO-MS, 2023) — multi-scale parallel kernels concat.

    Branches with kernels (1, 3, 5) — different receptive fields collected
    in parallel and concatenated. Good for varied object scales.
    """

    def __init__(self, c1, c2, kernels=(1, 3, 5), e=0.5, act="silu", **kw):
        super().__init__(**kw)
        n = len(kernels)
        self.c_ = max(int(c2 * e) // n, 1)
        self.branches = [TFConv(self.c_, k=k, s=1, act=act) for k in kernels]
        self.cv_in = TFConv(self.c_ * n, 1, 1, act=act)
        self.cv_out = TFConv(c2, 1, 1, act=act)

    def call(self, x, training=False):
        x = self.cv_in(x, training=training)
        # split equally among branches
        chunks = tf.split(x, len(self.branches), axis=-1)
        outs = [b(c, training=training) for b, c in zip(self.branches, chunks)]
        return self.cv_out(tf.concat(outs, axis=-1), training=training)


class TFCoordConvStem(keras.layers.Layer):
    """CoordConv (Liu et al, NeurIPS 2018) — append normalized x/y channels
    before the first Conv. Helps localization for small objects.

    Coord grid is pre-computed at build() as a fixed `tf.constant`, so the
    INT8 quantizer sees it as a constant tensor (not a runtime SHAPE/LINSPACE
    op) and can fuse the Concat into the Edge TPU partition.
    """

    def __init__(self, c2, k=3, s=2, p=None, act="silu", **kw):
        super().__init__(**kw)
        self.conv = TFConv(c2, k=k, s=s, p=p, act=act)
        self._grid = None

    def build(self, input_shape):
        H, W = int(input_shape[1]), int(input_shape[2])
        # normalized [-1, 1] grid baked as a constant — single-batch, so
        # batch dim is broadcast at concat time.
        ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)
        xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)
        gy = np.broadcast_to(ys[:, None, None], (H, W, 1)).copy()
        gx = np.broadcast_to(xs[None, :, None], (H, W, 1)).copy()
        grid_hw_2 = np.concatenate([gx, gy], axis=-1)             # (H, W, 2)
        self._grid = tf.constant(grid_hw_2[None, ...], dtype=tf.float32)  # (1, H, W, 2)
        super().build(input_shape)

    def call(self, x, training=False):
        # tf.concat does NOT broadcast batch dim — must tile the (1,H,W,2)
        # grid up to match. At export with batch_size=1 the multiplier is
        # constant 1 and the converter folds the tile into a no-op.
        bs = tf.shape(x)[0]
        grid = tf.tile(self._grid, [bs, 1, 1, 1])
        x_aug = tf.concat([x, grid], axis=-1)
        return self.conv(x_aug, training=training)




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
