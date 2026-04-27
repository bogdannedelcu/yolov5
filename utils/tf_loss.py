"""
TF port of YOLOv5 ComputeLoss (utils/loss.py).

Mirrors the PyTorch loss used by `train.py`:
- Anchor-based matching with offset bias 0.5 (3 candidate cells per target).
- Box: CIoU on decoded predictions.
- Objectness: BCE with target = IoU(pred, gt).detach() (clipped to [0,1]).
- Class: BCE multi-label.
- Per-scale balance [4.0, 1.0, 0.4] for P3/P4/P5.
- Hyperparams default to v5 defaults: box=0.05, obj=1.0, cls=0.5, anchor_t=4.

Inputs to call():
- preds: list of 3 raw scale maps [B, H_i, W_i, na*(5+nc)] (model output)
- targets: tensor [n_targets, 6] = [batch_idx, cls, cx, cy, w, h] (normalized [0,1])
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import tensorflow as tf


def _bbox_iou_ciou(box1, box2, eps=1e-7):
    """CIoU between two same-shape (..., 4) tensors of (cx, cy, w, h)."""
    b1_x1 = box1[..., 0] - box1[..., 2] / 2
    b1_x2 = box1[..., 0] + box1[..., 2] / 2
    b1_y1 = box1[..., 1] - box1[..., 3] / 2
    b1_y2 = box1[..., 1] + box1[..., 3] / 2
    b2_x1 = box2[..., 0] - box2[..., 2] / 2
    b2_x2 = box2[..., 0] + box2[..., 2] / 2
    b2_y1 = box2[..., 1] - box2[..., 3] / 2
    b2_y2 = box2[..., 1] + box2[..., 3] / 2

    inter = tf.maximum(tf.minimum(b1_x2, b2_x2) - tf.maximum(b1_x1, b2_x1), 0) * \
            tf.maximum(tf.minimum(b1_y2, b2_y2) - tf.maximum(b1_y1, b2_y1), 0)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = tf.maximum(b1_x2, b2_x2) - tf.minimum(b1_x1, b2_x1)
    ch = tf.maximum(b1_y2, b2_y2) - tf.minimum(b1_y1, b2_y1)
    c2 = cw * cw + ch * ch + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    v = (4 / (math.pi ** 2)) * tf.pow(tf.atan(w2 / h2) - tf.atan(w1 / h1), 2)
    alpha = v / (v - iou + (1 + eps))
    ciou = iou - (rho2 / c2 + v * alpha)
    return ciou, iou


class ComputeLossTF:
    """YOLOv5 detection loss (PT-parity, anchor-based, no DFL)."""

    def __init__(
        self,
        anchors: Sequence[Sequence[float]],
        strides: Sequence[float],
        nc: int,
        hyp: dict | None = None,
    ):
        # anchors: list of [w1,h1,w2,h2,...] per scale (image-pixel units)
        # We store anchors normalized by stride for matching against grid.
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.nc = nc
        self.no = nc + 5
        self.strides = tf.constant(list(strides), dtype=tf.float32)
        anc = tf.constant(anchors, dtype=tf.float32)             # (nl, na*2)
        anc = tf.reshape(anc, (self.nl, self.na, 2))             # (nl, na, 2)
        self.anchors_grid = anc                                   # in pixels
        self.anchors_norm = anc / tf.reshape(self.strides, (self.nl, 1, 1))  # in cells

        h = {
            "box": 0.05,
            "obj": 1.0,
            "cls": 0.5,
            "anchor_t": 4.0,
            "label_smoothing": 0.0,
            "obj_pw": 1.0,
            "cls_pw": 1.0,
            "gr": 1.0,
        }
        if hyp:
            h.update(hyp)
        self.hyp = h
        self.balance = {3: [4.0, 1.0, 0.4]}.get(self.nl, [4.0, 1.0, 0.25, 0.06, 0.02][: self.nl])

    # ---- anchor matching ----------------------------------------------------

    def _build_targets(self, targets, feature_hw):
        """
        Per-scale anchor matching with offset bias 0.5.

        targets: (nt, 6) = [batch, cls, cx, cy, w, h], normalized [0,1].
        feature_hw: list of (H_i, W_i) per scale.

        Returns lists of length nl:
          tcls : (M_i,) int
          tbox : (M_i, 4) target box (gx_in_cell, gy_in_cell, gw_in_cells, gh_in_cells)
          indices : (M_i, 4) [batch, anchor_idx, gj, gi]
          anch : (M_i, 2) anchor (cells)
        """
        na = self.na
        nt = tf.shape(targets)[0]

        # Append anchor index axis: (na, nt, 7) where last col is ai
        ai = tf.cast(tf.range(na)[:, None], targets.dtype)
        ai = tf.tile(ai, [1, nt])                                # (na, nt)
        targets_rep = tf.tile(targets[None, ...], [na, 1, 1])    # (na, nt, 6)
        t_full = tf.concat([targets_rep, ai[..., None]], axis=-1)  # (na, nt, 7)

        # offsets for the 5 candidate cells (center, +x, +y, -x, -y) — bias 0.5
        g = 0.5
        off = tf.constant(
            [[0, 0],
             [1, 0], [0, 1], [-1, 0], [0, -1]],
            dtype=tf.float32,
        ) * g  # (5, 2)

        out_tcls, out_tbox, out_idx, out_anch = [], [], [], []
        for i in range(self.nl):
            H, W = feature_hw[i]
            anchors_i = self.anchors_norm[i]                      # (na, 2)  in cells
            gain = tf.constant([1, 1, W, H, W, H, 1], dtype=tf.float32)  # scale normalized→grid
            t = t_full * gain                                     # (na, nt, 7)

            # ratio filter: max(r, 1/r) < anchor_t for both w and h
            wh = t[..., 4:6]                                      # (na, nt, 2) in cells
            r = wh / anchors_i[:, None, :]                        # (na, nt, 2)
            jmask = tf.reduce_max(tf.maximum(r, 1.0 / r), axis=-1) < self.hyp["anchor_t"]
            t = tf.boolean_mask(t, jmask)                         # (M0, 7)

            # offsets — for each surviving target, decide which neighboring cells also accept it
            gxy = t[..., 2:4]                                     # cell coords
            gxi = gain[2:4] - gxy                                 # mirror
            j = tf.logical_and(tf.math.floormod(gxy, 1.0) < g, gxy > 1.0)  # (M0, 2)
            l = tf.logical_and(tf.math.floormod(gxi, 1.0) < g, gxi > 1.0)
            jx, jy = j[..., 0], j[..., 1]
            lx, ly = l[..., 0], l[..., 1]
            ones = tf.ones_like(jx, dtype=tf.bool)
            keep = tf.stack([ones, jx, jy, lx, ly], axis=0)   # (5, M0) bool

            t5 = tf.tile(t[None, ...], [5, 1, 1])                 # (5, M0, 7)
            t5 = tf.boolean_mask(t5, keep)                        # (M, 7)
            offsets = tf.boolean_mask(
                tf.tile(off[:, None, :], [1, tf.shape(t)[0], 1]),  # (5, M0, 2)
                keep,
            )                                                      # (M, 2)

            b = tf.cast(t5[:, 0], tf.int32)
            c = tf.cast(t5[:, 1], tf.int32)
            gxy = t5[:, 2:4]
            gwh = t5[:, 4:6]
            gij = tf.cast(gxy - offsets, tf.int32)                 # (M, 2)
            gi = tf.clip_by_value(gij[:, 0], 0, W - 1)
            gj = tf.clip_by_value(gij[:, 1], 0, H - 1)
            ai_i = tf.cast(t5[:, 6], tf.int32)

            # tensor layout is (B, H, W, na, no) -> index order [b, gj, gi, ai]
            out_idx.append(tf.stack([b, gj, gi, ai_i], axis=1))    # (M, 4)
            out_tbox.append(tf.concat([gxy - tf.cast(gij, tf.float32), gwh], axis=1))  # (M, 4)
            out_tcls.append(c)
            out_anch.append(tf.gather(anchors_i, ai_i))            # (M, 2)
        return out_tcls, out_tbox, out_idx, out_anch

    # ---- main call ----------------------------------------------------------

    def __call__(self, preds: List[tf.Tensor], targets: tf.Tensor):
        """
        preds: list of (B, H_i, W_i, na*(5+nc)) raw maps.
        targets: (nt, 6).
        Returns: total_loss (scalar), (lbox, lobj, lcls).
        """
        device = preds[0].dtype
        lbox = tf.constant(0.0, dtype=device)
        lobj = tf.constant(0.0, dtype=device)
        lcls = tf.constant(0.0, dtype=device)

        feature_hw = [(int(p.shape[1]), int(p.shape[2])) for p in preds]
        tcls, tbox, indices, anchors = self._build_targets(targets, feature_hw)

        for i, p in enumerate(preds):
            B = tf.shape(p)[0]
            H, W = feature_hw[i]
            # reshape to (B, H, W, na, 5+nc)
            p_i = tf.reshape(p, (B, H, W, self.na, self.no))
            tobj = tf.zeros((B, H, W, self.na), dtype=p.dtype)

            idx = indices[i]                                       # (M, 4)
            M = int(tf.shape(idx)[0])
            if M > 0:
                # gather predictions at matched cells: (M, 5+nc)
                gathered = tf.gather_nd(p_i, idx)                 # (M, 5+nc)
                pxy = tf.sigmoid(gathered[:, 0:2]) * 2.0 - 0.5    # (M, 2) in cells
                pwh = (tf.sigmoid(gathered[:, 2:4]) * 2.0) ** 2 * anchors[i]  # (M, 2) in cells
                pbox = tf.concat([pxy, pwh], axis=-1)             # (M, 4)

                ciou, iou = _bbox_iou_ciou(pbox, tbox[i])
                lbox = lbox + tf.reduce_mean(1.0 - ciou)

                # objectness target = iou (clipped, gradient-stopped)
                iou_target = tf.stop_gradient(tf.clip_by_value(iou, 0.0, 1.0))
                # gr blend: (1-gr)*1 + gr*iou
                obj_target = (1.0 - self.hyp["gr"]) + self.hyp["gr"] * iou_target
                tobj = tf.tensor_scatter_nd_update(tobj, idx, obj_target)

                # classification (only when nc > 1, like PT)
                if self.nc > 1:
                    one_hot = tf.one_hot(tcls[i], self.nc, on_value=1.0 - self.hyp["label_smoothing"],
                                         off_value=self.hyp["label_smoothing"] / max(self.nc - 1, 1),
                                         dtype=p.dtype)
                    bce_cls = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=one_hot, logits=gathered[:, 5:]
                    )
                    lcls = lcls + tf.reduce_mean(bce_cls)

            # objectness BCE on full grid (per-cell, per-anchor)
            bce_obj = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tobj, logits=p_i[..., 4]
            )
            lobj = lobj + tf.reduce_mean(bce_obj) * self.balance[i]

        bs = tf.cast(tf.shape(preds[0])[0], lbox.dtype)
        lbox = lbox * self.hyp["box"]
        lobj = lobj * self.hyp["obj"]
        lcls = lcls * self.hyp["cls"]
        total = (lbox + lobj + lcls) * bs
        return total, (lbox, lobj, lcls)
