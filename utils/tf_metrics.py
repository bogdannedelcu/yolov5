"""TF-side validation: host-side decode of multi-output raw maps + reuse of PT
NMS / mAP utilities (`utils/general.py:non_max_suppression`,
`utils/metrics.py:ap_per_class`, `val.py:process_batch`).

The TF model output format is `[1, H_i, W_i, na*(5+nc)]` per scale (raw logits,
no in-graph decode). This module:
1. dequantizes (or accepts already-dequantized) per-scale outputs,
2. applies sigmoid + xy/wh anchor decode in NumPy,
3. concatenates across scales to a `[B, N_total, 5+nc]` tensor,
4. wraps to torch and reuses the PT non-max-suppression + per-class AP
   computation.

This avoids re-implementing NMS or mAP — same code path PT uses internally.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

from utils.general import non_max_suppression, xywh2xyxy, xyxy2xywh
from utils.metrics import ap_per_class, ConfusionMatrix
from val import process_batch  # PT util


def anchors_to_grid(anchors_yaml: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert YAML-style anchors `[[w1,h1,w2,h2,...], ...]` to (nl, na, 2)."""
    nl = len(anchors_yaml)
    na = len(anchors_yaml[0]) // 2
    return np.array(anchors_yaml, dtype=np.float32).reshape(nl, na, 2)


class TFLiteModelWrapper:
    """Adapter that makes a TFLite interpreter look like a Keras model.

    Exposes `__call__(x, training=False)` returning a list of dequantized
    fp32 tensors per scale (matching DetectionModelTF.model output).
    Handles uint8/int8 quantization at I/O automatically.
    """

    def __init__(self, tflite_path):
        import tensorflow as tf
        self.interp = tf.lite.Interpreter(model_path=str(tflite_path))
        self.interp.allocate_tensors()
        self.in_d = self.interp.get_input_details()[0]
        self.out_d = self.interp.get_output_details()
        # Sort outputs by feature map area (largest first → P3, then P4, P5)
        self._out_order = sorted(
            range(len(self.out_d)),
            key=lambda i: -int(np.prod(self.out_d[i]["shape"])),
        )

    @property
    def input_hw(self):
        return tuple(self.in_d["shape"][1:3])

    def __call__(self, x, training=False):
        import tensorflow as tf
        # x: NHWC float32 in [0, 1]
        x_np = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
        in_dtype = self.in_d["dtype"]
        if in_dtype in (np.uint8, np.int8):
            scale, zp = self.in_d["quantization"]
            lo, hi = (0, 255) if in_dtype == np.uint8 else (-128, 127)
            if scale > 0:
                # standard quantize: q = round(f / scale + zp)
                x_q = (x_np / scale + zp).round().clip(lo, hi).astype(in_dtype)
            else:
                # interpreter sometimes reports scale=0 for the no-op uint8
                # passthrough used by INT8 graphs with `inference_input_type=uint8`
                # (input range is just [0, 255]).
                x_q = (x_np * 255.0).round().clip(0, 255).astype(in_dtype)
        else:
            x_q = x_np.astype(np.float32)

        outs = []
        for bi in range(x_q.shape[0]):
            self.interp.set_tensor(self.in_d["index"], x_q[bi : bi + 1])
            self.interp.invoke()
            scale_outs = []
            for oi in self._out_order:
                d = self.out_d[oi]
                t = self.interp.get_tensor(d["index"])
                if d["dtype"] in (np.int8, np.uint8):
                    scale, zp = d["quantization"]
                    t = (t.astype(np.float32) - zp) * scale
                scale_outs.append(t)
            outs.append(scale_outs)

        # Stack across batch per scale → [B, H, W, C] per scale
        n_scales = len(outs[0])
        stacked = []
        for si in range(n_scales):
            stacked.append(np.concatenate([outs[bi][si] for bi in range(len(outs))], axis=0))
        return [tf.constant(s) for s in stacked]


def host_decode(
    preds: List[np.ndarray],
    anchors_pixel: np.ndarray,
    strides: Sequence[float],
    nc: int,
) -> torch.Tensor:
    """Decode multi-output raw maps to a single `[B, N_total, 5+nc]` tensor.

    Mirror of PT `Detect.forward` inference path (sigmoid + xy/wh decode).

    Args:
        preds: list of `nl` arrays, each `[B, H_i, W_i, na*(5+nc)]` fp32.
        anchors_pixel: `(nl, na, 2)` anchors in image-pixel units.
        strides: list of `nl` strides.
        nc: number of classes.

    Returns:
        torch.Tensor `[B, N_total, 5+nc]` with xywh (pixel units), obj_prob,
        cls_prob (all sigmoid-ed). Format expected by `non_max_suppression`.
    """
    no = nc + 5
    na = anchors_pixel.shape[1]
    out = []
    for i, p in enumerate(preds):
        B, H, W, _ = p.shape
        p = p.reshape(B, H, W, na, no).astype(np.float32)
        p_sig = 1.0 / (1.0 + np.exp(-p))
        gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        grid = np.stack([gx, gy], axis=-1).astype(np.float32)  # (H, W, 2)
        grid = grid[None, :, :, None, :]                        # (1, H, W, 1, 2)
        xy = (p_sig[..., 0:2] * 2.0 - 0.5 + grid) * strides[i]
        anchor = anchors_pixel[i].reshape(1, 1, 1, na, 2)
        wh = (p_sig[..., 2:4] * 2.0) ** 2 * anchor
        conf = p_sig[..., 4:5]
        cls = p_sig[..., 5:]
        decoded = np.concatenate([xy, wh, conf, cls], axis=-1)
        decoded = decoded.reshape(B, H * W * na, no)
        out.append(decoded)
    pred = np.concatenate(out, axis=1)
    return torch.from_numpy(pred).float()


def tf_validate(
    model,
    val_loader,
    anchors_yaml: Sequence[Sequence[float]],
    strides: Sequence[float],
    nc: int,
    conf_thres: float = 0.001,
    iou_thres: float = 0.6,
    max_det: int = 300,
    iouv: torch.Tensor | None = None,
    names: dict | None = None,
    plot: bool = False,
    save_dir: str | Path = ".",
    confusion_matrix: bool = False,
    save_json: bool = False,
    json_path: str | Path | None = None,
) -> Tuple[float, float, float, float, np.ndarray | None]:
    """Run a full validation loop over `val_loader`.

    Returns `(mp, mr, map50, map50_95, ap_class)`. `ap_class` is the array of
    class indices that had at least one prediction (None if no detections at all).

    `val_loader` yields `(imgs_uint8_NCHW, targets_Nx6, paths, shapes)` from
    PT's `create_dataloader`. Same convention as `train_tf.py` already uses.
    """
    if iouv is None:
        iouv = torch.linspace(0.5, 0.95, 10)
    niou = iouv.numel()
    anchors_pixel = anchors_to_grid(anchors_yaml)

    cm = ConfusionMatrix(nc=nc) if confusion_matrix else None
    stats = []  # (correct, conf, pcls, tcls)
    jdict = [] if save_json else None  # COCO-format predictions
    for batch in val_loader:
        imgs, targets, paths, _shapes = batch
        imgs_np = imgs.numpy()                          # (B, 3, H, W) uint8
        imgs_nhwc = np.transpose(imgs_np, (0, 2, 3, 1)).astype(np.float32) / 255.0
        B, _, H, W = imgs_np.shape

        preds_tf = model(imgs_nhwc, training=False)
        if not isinstance(preds_tf, (list, tuple)):
            preds_tf = [preds_tf]
        preds_np = [p.numpy() for p in preds_tf]

        # decode -> [B, N, 5+nc] (xywh in pixel space, scaled by stride internally)
        pred = host_decode(preds_np, anchors_pixel, strides, nc)
        # NMS (PT util)
        out_per_img = non_max_suppression(
            pred, conf_thres=conf_thres, iou_thres=iou_thres,
            max_det=max_det, multi_label=True,
        )

        targets_np = targets.numpy()
        for si, det in enumerate(out_per_img):
            det = det.cpu()
            labels_si = targets_np[targets_np[:, 0] == si, 1:]   # (n_gt, 5) cls + xywh normalized
            tcls = labels_si[:, 0].tolist() if labels_si.shape[0] else []

            # COCO json (xywh top-left convention)
            if jdict is not None and det.shape[0] > 0:
                p = Path(paths[si]) if not isinstance(paths[si], Path) else paths[si]
                image_id = int(p.stem) if p.stem.isdigit() else p.stem
                box = xyxy2xywh(det[:, :4].clone())
                box[:, :2] -= box[:, 2:] / 2  # xy center → top-left
                for b, score, cls_id in zip(box.tolist(), det[:, 4].tolist(), det[:, 5].tolist()):
                    jdict.append({
                        "image_id": image_id,
                        "category_id": int(cls_id),
                        "bbox": [round(float(x), 3) for x in b],
                        "score": round(float(score), 5),
                    })

            if det.shape[0] == 0:
                if labels_si.shape[0]:
                    if cm is not None:
                        cm.process_batch(detections=None,
                                         labels=torch.from_numpy(labels_si[:, 0:1]).float())
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.zeros(0),
                                  torch.zeros(0), tcls))
                continue

            if labels_si.shape[0]:
                tbox = xywh2xyxy(torch.from_numpy(labels_si[:, 1:5]).float())
                tbox[:, [0, 2]] *= W
                tbox[:, [1, 3]] *= H
                labelsn = torch.cat((torch.from_numpy(labels_si[:, 0:1]).float(), tbox), 1)
                correct = process_batch(det, labelsn, iouv)
                if isinstance(correct, np.ndarray):
                    correct = torch.from_numpy(correct)
                if cm is not None:
                    cm.process_batch(det, labelsn)
            else:
                correct = torch.zeros(det.shape[0], niou, dtype=torch.bool)

            stats.append((correct, det[:, 4], det[:, 5], tcls))

    if not stats:
        return 0.0, 0.0, 0.0, 0.0, None

    # concat over batches
    stats_cat = [np.concatenate(x, 0) for x in zip(*[
        (c.numpy() if isinstance(c, torch.Tensor) else np.asarray(c),
         conf.numpy() if isinstance(conf, torch.Tensor) else np.asarray(conf),
         pcls.numpy() if isinstance(pcls, torch.Tensor) else np.asarray(pcls),
         np.asarray(tcls, dtype=np.float32))
        for c, conf, pcls, tcls in stats
    ])]
    if stats_cat[0].shape[0] == 0:
        return 0.0, 0.0, 0.0, 0.0, None

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(
        *stats_cat, plot=plot, save_dir=save_dir, names=names or {},
    )
    ap50, ap50_95 = ap[:, 0], ap.mean(1)
    mp = p.mean() if len(p) else 0.0
    mr = r.mean() if len(r) else 0.0
    map50 = ap50.mean() if len(ap50) else 0.0
    map50_95 = ap50_95.mean() if len(ap50_95) else 0.0

    if cm is not None and plot:
        try:
            cm.plot(save_dir=save_dir, names=list((names or {}).values()))
        except Exception as e:
            print(f"[tf_validate] confusion matrix plot failed: {e}")

    if jdict is not None:
        import json as _json
        jp = Path(json_path) if json_path else save_dir / "predictions.json"
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(_json.dumps(jdict))
        print(f"[tf_validate] saved {len(jdict)} predictions to {jp} (COCO format)")

    return float(mp), float(mr), float(map50), float(map50_95), ap_class
