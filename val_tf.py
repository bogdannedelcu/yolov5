"""TF-native validation script — mirror of `val.py` (PyTorch).

Loads a trained TF model from `--weights` (using sidecar `architecture.json`
to rebuild the graph) and computes P / R / mAP@0.5 / mAP@0.5:0.95 on the
val split of `--data`.

Decode + NMS happens host-side via `utils.tf_metrics.tf_validate`, which
reuses PT's `non_max_suppression` and `ap_per_class`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.tf_yolo import DetectionModelTF
from utils.tf_dataloaders import parse_data_yaml
from utils.tf_metrics import tf_validate, TFLiteModelWrapper
from utils.dataloaders import create_dataloader


def parse_imgsz(values):
    if len(values) == 1:
        return (int(values[0]), int(values[0]))
    if len(values) == 2:
        return (int(values[0]), int(values[1]))
    raise ValueError("--imgsz takes 1 or 2 values (H W)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",
                    help="Path to .weights.h5 from train_tf.py (Keras float32 path)")
    ap.add_argument("--tflite",
                    help="Path to *.tflite model (INT8 or fp16) — runs val on the quantized graph")
    ap.add_argument("--data", required=True, help="data.yaml (must contain val split)")
    ap.add_argument("--cfg", help="model YAML (auto-loaded from sidecar architecture.json if not given)")
    ap.add_argument("--imgsz", nargs="+",
                    help="H or H W (auto-loaded from sidecar if not given)")
    ap.add_argument("--nc", type=int, default=None)
    ap.add_argument("--act", default=None,
                    choices=[None, "silu", "swish", "relu", "relu6", "leaky"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--conf-thres", type=float, default=0.001)
    ap.add_argument("--iou-thres", type=float, default=0.6)
    ap.add_argument("--plot", action="store_true",
                    help="save PR/F1 curves + confusion_matrix.png next to weights")
    ap.add_argument("--save-dir", default=None,
                    help="output dir for plots (default: weights parent dir)")
    ap.add_argument("--save-json", action="store_true",
                    help="save predictions.json in COCO format (for pycocotools eval)")
    ap.add_argument("--task", default="val", choices=["val", "speed", "study"],
                    help="val: standard mAP loop. speed: latency benchmark. "
                         "study: sweep imgsz vs mAP/latency")
    ap.add_argument("--study-sizes", nargs="+", type=int,
                    default=[320, 416, 512, 608, 640, 736],
                    help="imgsz values for --task study (each rounded to stride)")
    ap.add_argument("--speed-warmup", type=int, default=10,
                    help="warmup iterations for --task speed")
    ap.add_argument("--speed-runs", type=int, default=100,
                    help="timed iterations for --task speed")
    args = ap.parse_args()

    if not args.weights and not args.tflite:
        raise SystemExit("Need --weights or --tflite")

    wp = Path(args.weights or args.tflite)
    sidecar = wp.parent / "architecture.json"
    if sidecar.exists():
        sc = json.loads(sidecar.read_text())
        args.cfg = args.cfg or sc["cfg"]
        if not args.imgsz:
            args.imgsz = [str(x) for x in sc["imgsz_hw"]]
        if args.act is None:
            args.act = sc.get("act", "silu")
        if args.nc is None:
            args.nc = sc.get("nc")
        print(f"[load] sidecar -> cfg={args.cfg} imgsz={args.imgsz} act={args.act} nc={args.nc}")
    if not args.cfg or not args.imgsz:
        raise SystemExit("Need --cfg + --imgsz (or sidecar architecture.json next to weights)")
    args.act = args.act or "silu"

    img_hw = parse_imgsz(args.imgsz)

    print(f"[data] reading {args.data}")
    data = parse_data_yaml(args.data)
    nc = args.nc or data["nc"]
    print(f"[data] nc={nc} val={data['val']}")

    if args.tflite:
        print(f"[build] TFLite model from {args.tflite}")
        tflite_model = TFLiteModelWrapper(args.tflite)
        # we still need a DetectionModelTF for anchors + strides metadata
        detmodel_meta = DetectionModelTF(
            cfg=args.cfg, imgsz_hw=img_hw, nc=nc, act=args.act, batch_size=1,
        )
        eval_model = tflite_model
        anchors = detmodel_meta.anchors
        strides = detmodel_meta.strides
        # adopt model's own input HW (may differ from sidecar imgsz if exported separately)
        tflite_hw = tflite_model.input_hw
        if tflite_hw[0] > 0 and tflite_hw[1] > 0:
            img_hw = tflite_hw
            print(f"[build] TFLite input shape={img_hw}")
    else:
        print(f"[build] cfg={args.cfg} imgsz={img_hw} act={args.act}")
        detmodel = DetectionModelTF(
            cfg=args.cfg, imgsz_hw=img_hw, nc=nc, act=args.act, batch_size=None,
        )
        detmodel.load_weights(wp)
        print(f"[load] weights from {wp}")
        eval_model = detmodel.model
        anchors = detmodel.anchors
        strides = detmodel.strides

    pt_imgsz = max(img_hw)
    stride = int(max(strides))
    val_loader, val_ds = create_dataloader(
        path=str(data["val"]),
        imgsz=pt_imgsz, batch_size=args.batch, stride=stride,
        single_cls=False, hyp=None, augment=False, cache=False,
        rect=True, rank=-1, workers=args.workers, prefix="val: ",
        shuffle=False, seed=0,
    )
    print(f"[data] val={len(val_ds)} images")

    save_dir = Path(args.save_dir) if args.save_dir else wp.parent

    if args.task == "speed":
        # Latency benchmark on a single image, batch-1
        import time
        # Use first val image as a stable input
        sample = next(iter(val_loader))
        imgs = sample[0].numpy()
        x = np.transpose(imgs[:1], (0, 2, 3, 1)).astype(np.float32) / 255.0
        # warmup
        for _ in range(args.speed_warmup):
            eval_model(x, training=False)
        # timed
        t = []
        for _ in range(args.speed_runs):
            t0 = time.time()
            eval_model(x, training=False)
            t.append((time.time() - t0) * 1000)
        t = np.array(t)
        print()
        print("=" * 60)
        print(f"  SPEED: imgsz={img_hw}  batch=1  iters={args.speed_runs}")
        print(f"    mean={t.mean():.2f}ms  std={t.std():.2f}ms  "
              f"p50={np.percentile(t, 50):.2f}  p95={np.percentile(t, 95):.2f}  "
              f"FPS={1000.0 / t.mean():.1f}")
        print("=" * 60)
        return

    if args.task == "study":
        # Sweep imgsz, build a fresh model + loader for each, report mAP + speed
        import time
        print()
        print("=" * 80)
        print(f"  STUDY: sweeping imgsz {args.study_sizes} on val={data['val']}")
        print(f"  {'imgsz':>6}  {'P':>8}  {'R':>8}  {'mAP50':>8}  {'mAP50-95':>10}  {'ms/img':>8}  {'FPS':>6}")
        for sz in args.study_sizes:
            sz = (sz // stride) * stride  # snap to stride
            # rebuild loader and model at this imgsz
            study_loader, study_ds = create_dataloader(
                path=str(data["val"]), imgsz=sz, batch_size=args.batch, stride=stride,
                single_cls=False, hyp=None, augment=False, cache=False,
                rect=True, rank=-1, workers=args.workers, prefix=f"study {sz}: ",
                shuffle=False, seed=0,
            )
            if args.tflite:
                # TFLite is tied to its built-in input shape; skip mismatched sizes
                if tflite_model.input_hw != (sz, sz):
                    print(f"  {sz:>6}  (TFLite input is {tflite_model.input_hw}; skipped)")
                    continue
                study_eval = tflite_model
            else:
                study_dm = DetectionModelTF(cfg=args.cfg, imgsz_hw=(sz, sz),
                                            nc=nc, act=args.act, batch_size=None)
                study_dm.load_weights(wp)
                study_eval = study_dm.model

            # speed
            sample = next(iter(study_loader))
            imgs = sample[0].numpy()
            x = np.transpose(imgs[:1], (0, 2, 3, 1)).astype(np.float32) / 255.0
            for _ in range(5):
                study_eval(x, training=False)
            t = []
            for _ in range(20):
                t0 = time.time()
                study_eval(x, training=False)
                t.append((time.time() - t0) * 1000)
            ms = float(np.mean(t))
            # mAP
            mp, mr, map50, map50_95, _ = tf_validate(
                study_eval, study_loader, anchors, strides, nc,
                conf_thres=args.conf_thres, iou_thres=args.iou_thres,
                names=data["names"],
            )
            print(f"  {sz:>6}  {mp:>8.4f}  {mr:>8.4f}  {map50:>8.4f}  {map50_95:>10.4f}  "
                  f"{ms:>8.2f}  {1000.0 / ms:>6.1f}")
        print("=" * 80)
        return

    # default --task val
    mp, mr, map50, map50_95, ap_class = tf_validate(
        eval_model, val_loader,
        anchors, strides, nc,
        conf_thres=args.conf_thres, iou_thres=args.iou_thres,
        names=data["names"], plot=args.plot, save_dir=save_dir,
        confusion_matrix=args.plot,
        save_json=args.save_json,
        json_path=(save_dir / "predictions.json") if args.save_json else None,
    )

    print()
    print("=" * 60)
    print(f"  Class       Images  Instances    P       R    mAP50  mAP50-95")
    print(f"  all         {len(val_ds):>5d}        --   {mp:.4f}  {mr:.4f}  {map50:.4f}  {map50_95:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
