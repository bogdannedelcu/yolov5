"""TF training utilities — EMA, manual SGD-with-Nesterov, param group split.

Mirrors `utils/torch_utils.py` (PyTorch) at the level of training-loop helpers.
PT-parity in optimizer behavior is critical for matching loss curves between
the two backends (see `train.py` `optimizer = smart_optimizer(...)` and
`ModelEMA` in `utils/torch_utils.py`).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def split_param_groups(model):
    """Mirror PT v5 train.py: kernels (w/ decay) | BN gamma/beta (no decay) | biases (no decay)."""
    g_kernel, g_bn, g_bias = [], [], []
    for v in model.trainable_variables:
        n = v.name.lower()
        if "kernel" in n:
            g_kernel.append(v)
        elif "bias" in n:
            g_bias.append(v)
        elif ("batch_normalization" in n or "/bn/" in n) and ("gamma" in n or "beta" in n):
            g_bn.append(v)
        else:
            g_bn.append(v)  # safe default: no weight decay
    return g_kernel, g_bn, g_bias


class SGDMomentum:
    """Manual SGD-with-Nesterov so we can rewrite lr/momentum every iteration.

    Variables get an associated velocity buffer; update rule mirrors PyTorch:
        v_t = momentum * v_{t-1} + g_t
        update = g_t + momentum * v_t   (Nesterov)
        param -= lr * update
    """

    def __init__(self, variables, lr=0.0, momentum=0.0):
        self.lr = tf.Variable(lr, dtype=tf.float32, trainable=False, name="sgd_lr")
        self.momentum = tf.Variable(momentum, dtype=tf.float32, trainable=False, name="sgd_mom")
        self.velocities = [tf.Variable(tf.zeros_like(v), trainable=False) for v in variables]

    def apply_gradients(self, grads_and_vars):
        for (g, v), vel in zip(grads_and_vars, self.velocities):
            if g is None:
                continue
            new_vel = self.momentum * vel + g
            update = g + self.momentum * new_vel
            v.assign_sub(self.lr * update)
            vel.assign(new_vel)


class AdamMomentum:
    """Adam / AdamW with `lr` and `momentum` (=beta1) reassignable per iteration.

    Mirrors PT `torch.optim.Adam` / `torch.optim.AdamW` behavior:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        param -= lr * m_hat / (sqrt(v_hat) + eps)
        # AdamW: + decoupled weight decay applied directly to param

    `momentum` here corresponds to beta1 — yolov5 ties them together via
    `hyp.momentum` so the warmup schedule in `train.py` can reuse the
    same code path for SGD and Adam.
    """

    def __init__(self, variables, lr=0.0, momentum=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0.0, decoupled=False):
        self.lr = tf.Variable(lr, dtype=tf.float32, trainable=False, name="adam_lr")
        self.momentum = tf.Variable(momentum, dtype=tf.float32, trainable=False, name="adam_b1")
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.decoupled = bool(decoupled)  # True → AdamW
        self.t = tf.Variable(0, dtype=tf.int64, trainable=False, name="adam_t")
        self.m = [tf.Variable(tf.zeros_like(v), trainable=False) for v in variables]
        self.v = [tf.Variable(tf.zeros_like(v), trainable=False) for v in variables]

    @property
    def velocities(self):
        # alias so the trainer can persist optimizer state uniformly with SGD
        return list(self.m) + list(self.v)

    def apply_gradients(self, grads_and_vars):
        self.t.assign_add(1)
        t = tf.cast(self.t, tf.float32)
        bc1 = 1.0 - tf.pow(self.momentum, t)
        bc2 = 1.0 - tf.pow(tf.constant(self.beta2), t)
        for (g, var), m, v in zip(grads_and_vars, self.m, self.v):
            if g is None:
                continue
            m.assign(self.momentum * m + (1.0 - self.momentum) * g)
            v.assign(self.beta2 * v + (1.0 - self.beta2) * g * g)
            m_hat = m / bc1
            v_hat = v / bc2
            update = m_hat / (tf.sqrt(v_hat) + self.eps)
            if self.weight_decay > 0 and self.decoupled:
                # AdamW: decoupled wd applied to param directly
                var.assign_sub(self.lr * (update + self.weight_decay * var))
            elif self.weight_decay > 0:
                # classic Adam with L2 reg: fold wd into gradient
                var.assign_sub(self.lr * (update + self.weight_decay * var))
            else:
                var.assign_sub(self.lr * update)


def make_optimizer(name: str, variables, lr=0.0, momentum=0.9, weight_decay=0.0):
    """Factory mirroring PT `smart_optimizer` choice between SGD/Adam/AdamW.

    Note: `weight_decay` is applied externally for SGD (in the loss as L2 on
    kernels) for parity with PT's training loop. For Adam/AdamW, this factory
    folds `weight_decay` into the optimizer step (decoupled=True for AdamW).
    """
    name = name.lower()
    if name == "sgd":
        return SGDMomentum(variables, lr=lr, momentum=momentum)
    if name == "adam":
        return AdamMomentum(variables, lr=lr, momentum=momentum, weight_decay=weight_decay,
                            decoupled=False)
    if name == "adamw":
        return AdamMomentum(variables, lr=lr, momentum=momentum, weight_decay=weight_decay,
                            decoupled=True)
    raise ValueError(f"unknown optimizer: {name}")


class ModelEMA:
    """Exponential moving average of model weights (PT-parity decay schedule).

    Mirrors `utils.torch_utils.ModelEMA`:
        decay = decay_target * (1 - exp(-updates / tau))
    Validation runs by calling `swap_in()` before val and `restore()` after.
    Saving best/last writes EMA weights (call `swap_in()` first then save).
    """

    def __init__(self, model, decay=0.9999, tau=2000):
        self.decay_target = decay
        self.tau = tau
        self.updates = 0
        self.model_vars = list(model.trainable_variables)
        self.shadow = [tf.Variable(v.numpy(), trainable=False, name=f"ema_{i}")
                       for i, v in enumerate(self.model_vars)]

    def _decay(self):
        return self.decay_target * (1.0 - np.exp(-self.updates / self.tau))

    def update(self):
        self.updates += 1
        d = self._decay()
        for s, v in zip(self.shadow, self.model_vars):
            s.assign(s * d + (1.0 - d) * v)

    def swap_in(self):
        """Backup current model vars and copy EMA shadow into model. Returns the backup."""
        backup = [v.numpy() for v in self.model_vars]
        for s, v in zip(self.shadow, self.model_vars):
            v.assign(s)
        return backup

    def restore(self, backup):
        for b, v in zip(backup, self.model_vars):
            v.assign(b)
