"""TF-native YOLOv5 inference script — mirror of `detect.py` (PyTorch).

Usage:
    python detect_tf.py --weights runs/.../best.weights.h5 \
        --source path/to/image_or_folder_or_video \
        --conf-thres 0.25 --iou-thres 0.45 \
        --out runs/tf_native/detect

Reuses PT's `LoadImages` for source enumeration and `Annotator` for box drawing.
Decode + NMS via `utils.tf_metrics.host_decode` + `utils.general.non_max_suppression`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.tf_yolo import DetectionModelTF
from utils.tf_metrics import host_decode, anchors_to_grid
from utils.tf_dataloaders import parse_data_yaml
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes
from ultralytics.utils.plotting import Annotator, colors


def parse_imgsz(values):
    if values is None:
        return None
    if len(values) == 1:
        return (int(values[0]), int(values[0]))
    if len(values) == 2:
        return (int(values[0]), int(values[1]))
    raise ValueError("--imgsz takes 1 or 2 values (H W)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True,
                    help="path to image / folder / video / webcam '0'")
    ap.add_argument("--cfg", default=None,
                    help="model YAML (auto-loaded from sidecar architecture.json)")
    ap.add_argument("--imgsz", nargs="+", default=None,
                    help="H or H W (auto-loaded from sidecar)")
    ap.add_argument("--nc", type=int, default=None)
    ap.add_argument("--act", default=None)
    ap.add_argument("--data", default=None,
                    help="data.yaml — used to load class names for labels")
    ap.add_argument("--conf-thres", type=float, default=0.25)
    ap.add_argument("--iou-thres", type=float, default=0.45)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--line-thickness", type=int, default=2)
    ap.add_argument("--hide-labels", action="store_true")
    ap.add_argument("--hide-conf", action="store_true")
    ap.add_argument("--save-txt", action="store_true",
                    help="save predictions as YOLO-format txt files")
    ap.add_argument("--save-crop", action="store_true",
                    help="save each detected bbox as a separate image under crops/<class>/")
    ap.add_argument("--classes", nargs="+", type=int, default=None,
                    help="filter by class id(s); only these classes are kept after NMS")
    ap.add_argument("--agnostic-nms", action="store_true",
                    help="class-agnostic NMS (treat all classes as one)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wp = Path(args.weights)
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
    if not args.cfg or not args.imgsz:
        raise SystemExit("Need --cfg + --imgsz (or sidecar architecture.json)")
    args.act = args.act or "silu"
    img_hw = parse_imgsz(args.imgsz)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_txt:
        (out_dir / "labels").mkdir(exist_ok=True)
    if args.save_crop:
        (out_dir / "crops").mkdir(exist_ok=True)

    print(f"[build] cfg={args.cfg} imgsz={img_hw} act={args.act} nc={args.nc}")
    detmodel = DetectionModelTF(
        cfg=args.cfg, imgsz_hw=img_hw, nc=args.nc, act=args.act, batch_size=1,
    )
    detmodel.load_weights(wp)
    print(f"[load] weights from {wp}")
    anchors_pixel = anchors_to_grid(detmodel.anchors)

    names = None
    if args.data:
        d = parse_data_yaml(args.data)
        names = d.get("names")
    if names is None:
        names = {i: str(i) for i in range(detmodel.nc)}
    elif isinstance(names, dict):
        names = {int(k): v for k, v in names.items()}
    elif isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}

    pt_imgsz = max(img_hw)
    stride = int(max(detmodel.strides))
    dataset = LoadImages(args.source, img_size=pt_imgsz, stride=stride, auto=False)

    seen = 0
    n_dets = 0
    for path, im, im0s, vid_cap, s in dataset:
        # PT LoadImages returns NCHW uint8 letterboxed; convert to NHWC float32
        im_np = im.astype(np.float32) / 255.0
        im_np = np.transpose(im_np, (1, 2, 0))[None, ...]  # (1, H, W, 3)
        # If LoadImages letterboxed to a different shape (auto=False with stride), pad/crop:
        H_in, W_in, _ = im_np.shape[1:]
        if (H_in, W_in) != img_hw:
            # resize (no aspect preservation; LoadImages should already handle this)
            im_np = tf.image.resize(im_np, img_hw, method="bilinear").numpy()

        preds_tf = detmodel.model(im_np, training=False)
        if not isinstance(preds_tf, (list, tuple)):
            preds_tf = [preds_tf]
        preds_np = [p.numpy() for p in preds_tf]

        pred = host_decode(preds_np, anchors_pixel, detmodel.strides, detmodel.nc)
        out_per_img = non_max_suppression(
            pred, conf_thres=args.conf_thres, iou_thres=args.iou_thres,
            classes=args.classes, agnostic=args.agnostic_nms,
            max_det=args.max_det,
        )
        det = out_per_img[0]

        # rescale boxes from img_hw (model input) to im0s original shape
        if det.shape[0] > 0:
            det[:, :4] = scale_boxes(img_hw, det[:, :4], im0s.shape).round()
            n_dets += det.shape[0]

        annotator = Annotator(im0s.copy(), line_width=args.line_thickness, example=str(names))
        for di, (*xyxy, conf, cls) in enumerate(reversed(det.cpu().tolist())):
            cls = int(cls)
            label = None if args.hide_labels else (
                names.get(cls, str(cls)) if args.hide_conf else f"{names.get(cls, str(cls))} {conf:.2f}"
            )
            annotator.box_label(xyxy, label, color=colors(cls, True))
            if args.save_crop:
                x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(im0s.shape[1], x2); y2 = min(im0s.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    crop_dir = out_dir / "crops" / names.get(cls, str(cls))
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    crop = cv2.cvtColor(im0s[y1:y2, x1:x2], cv2.COLOR_RGB2BGR) \
                           if im0s.shape[2] == 3 else im0s[y1:y2, x1:x2]
                    cv2.imwrite(str(crop_dir / f"{Path(path).stem}_{di}.jpg"), crop)

        out_im = annotator.result()
        save_path = out_dir / Path(path).name
        cv2.imwrite(str(save_path), out_im)

        if args.save_txt and det.shape[0] > 0:
            h0, w0 = im0s.shape[:2]
            with (out_dir / "labels" / (Path(path).stem + ".txt")).open("w") as f:
                for *xyxy, conf, cls in det.cpu().tolist():
                    x1, y1, x2, y2 = xyxy
                    cx = (x1 + x2) / 2 / w0
                    cy = (y1 + y2) / 2 / h0
                    w = (x2 - x1) / w0
                    h = (y2 - y1) / h0
                    f.write(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")

        seen += 1
        print(f"[{seen}] {Path(path).name}: {det.shape[0]} detections -> {save_path}")

    print()
    print(f"[done] processed {seen} image(s); {n_dets} total detections; out={out_dir}")


if __name__ == "__main__":
    main()
