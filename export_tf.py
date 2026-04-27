"""TF-native export: YAML or trained weights → INT8 TFLite + Edge TPU.

Mirror of `export.py` (PyTorch) for the TF-native NHWC path. Reads either:
- `--cfg models/yolov5*.yaml --imgsz H W`  (fresh export, random weights)
- `--weights runs/.../best.weights.h5` + `architecture.json` sidecar (trained)

Produces:
- `{out}/model.keras`                 saved Keras model
- `{out}/detector_int8.tflite`        INT8 quantized (uint8 in, int8 out)
- `{out}/detector_int8_edgetpu.tflite` (after edgetpu_compiler)
- `{out}/edgetpu_compile.log`         compiler output

Calibration source priority: `--calib-images` > `--data` (uses train split) >
random uint8 noise (smoke fallback).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import tensorflow as tf  # noqa: E402

from models.tf_yolo import DetectionModelTF  # noqa: E402
from utils.tf_dataloaders import build_calibration_generator, parse_data_yaml  # noqa: E402


def parse_imgsz(values):
    if len(values) == 1:
        return (int(values[0]), int(values[0]))
    if len(values) == 2:
        return (int(values[0]), int(values[1]))
    raise ValueError("--imgsz takes 1 or 2 values (H W)")


def random_dataset(img_hw, n=100, seed=0):
    rng = np.random.default_rng(seed)
    H, W = img_hw

    def gen():
        for _ in range(n):
            x = rng.integers(0, 256, size=(1, H, W, 3), dtype=np.uint8).astype(np.float32) / 255.0
            yield [x]
    return gen


def quantize_to_tflite(model, img_hw, out_path: Path, calib_gen=None, n_calib=100, seed=0):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = calib_gen if calib_gen is not None \
        else random_dataset(img_hw, n=n_calib, seed=seed)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.int8
    tflite_bytes = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_bytes)
    return out_path


def fp16_to_tflite(model, out_path: Path):
    """Half-precision TFLite (no calibration needed). Larger than INT8 but
    closer to fp32 accuracy. Mirrors PT export's `--half` TFLite path.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_bytes = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_bytes)
    return out_path


def saved_model_export(model, out_dir: Path):
    """Export TF SavedModel (full graph fp32, no quantization)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(model, str(out_dir))
    return out_dir


def op_histogram_via_interpreter(tflite_path: Path) -> dict:
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    ops: dict = {}
    for d in interp._get_ops_details():
        ops[d["op_name"]] = ops.get(d["op_name"], 0) + 1
    return ops


def run_edgetpu_compiler(tflite_path: Path):
    if shutil.which("edgetpu_compiler") is None:
        return None, "edgetpu_compiler not found in PATH"
    out_dir = tflite_path.parent
    proc = subprocess.run(
        ["edgetpu_compiler", "-s", "-o", str(out_dir), str(tflite_path)],
        capture_output=True, text=True,
    )
    log = (proc.stdout or "") + (proc.stderr or "")
    compiled = tflite_path.with_name(tflite_path.stem + "_edgetpu.tflite")
    return (compiled if compiled.exists() else None), log


def parse_compiler_summary(log: str) -> dict:
    out = {"total": None, "edgetpu": 0, "cpu": 0, "subgraphs": None}
    for line in log.splitlines():
        s = line.strip()
        if s.startswith("Number of Edge TPU subgraphs"):
            try: out["subgraphs"] = int(s.split(":")[1])
            except Exception: pass
        elif s.startswith("Total number of operations"):
            try: out["total"] = int(s.split(":")[1])
            except Exception: pass
        elif "Mapped to Edge TPU" in s:
            parts = s.split()
            if len(parts) >= 5:
                try: out["edgetpu"] += int(parts[1])
                except Exception: pass
        elif s.endswith("CPU"):
            parts = s.split()
            if len(parts) >= 3:
                try: out["cpu"] += int(parts[1])
                except Exception: pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", help="YAML model config (e.g. models/yolov5n.yaml). "
                                  "Optional if --weights has architecture.json sidecar.")
    ap.add_argument("--weights", help="Path to a saved .weights.h5 from train_tf.py")
    ap.add_argument("--imgsz", nargs="+", help="H or H W (required if --weights without sidecar)")
    ap.add_argument("--nc", type=int, default=None)
    ap.add_argument("--act", default="silu", choices=["silu", "swish", "relu", "relu6", "leaky"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-calib", type=int, default=100)
    ap.add_argument("--data", default=None,
                    help="data.yaml — calibration set built from train split")
    ap.add_argument("--calib-images", default=None,
                    help="explicit calibration folder (overrides --data)")
    ap.add_argument("--no-edgetpu", action="store_true")
    ap.add_argument("--include", nargs="+",
                    default=["tflite_int8", "edgetpu"],
                    choices=["tflite_int8", "tflite_fp16", "saved_model", "edgetpu", "keras"],
                    help="formats to export (default: INT8 TFLite + EdgeTPU)")
    args = ap.parse_args()
    if args.no_edgetpu and "edgetpu" in args.include:
        args.include = [x for x in args.include if x != "edgetpu"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.weights:
        wp = Path(args.weights)
        sidecar = wp.parent / "architecture.json"
        if not args.cfg and sidecar.exists():
            import json
            sc = json.loads(sidecar.read_text())
            args.cfg = sc["cfg"]
            if not args.imgsz:
                args.imgsz = [str(x) for x in sc["imgsz_hw"]]
            args.act = sc.get("act", args.act)
            if args.nc is None:
                args.nc = sc.get("nc")
            print(f"[load] sidecar -> cfg={args.cfg} imgsz={args.imgsz} "
                  f"act={args.act} nc={args.nc}")
        if not args.cfg or not args.imgsz:
            raise SystemExit("Need --cfg + --imgsz (or sidecar architecture.json next to weights)")
        img_hw = parse_imgsz(args.imgsz)
        print(f"[build] cfg={args.cfg} imgsz={img_hw} nc={args.nc or 'yaml'} act={args.act}")
        detmodel = DetectionModelTF(
            cfg=args.cfg, imgsz_hw=img_hw, nc=args.nc, act=args.act, batch_size=1,
        )
        detmodel.load_weights(wp)
        print(f"[load] weights from {wp}")
    else:
        if not args.cfg or not args.imgsz:
            raise SystemExit("Need --cfg and --imgsz, or --weights")
        img_hw = parse_imgsz(args.imgsz)
        print(f"[build] cfg={args.cfg} imgsz={img_hw} nc={args.nc or 'yaml'} act={args.act}")
        detmodel = DetectionModelTF(
            cfg=args.cfg, imgsz_hw=img_hw, nc=args.nc, act=args.act, batch_size=1,
        )

    model = detmodel.model
    nc = detmodel.nc
    strides = detmodel.strides
    n_params = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"[build] params={n_params:,} nc={nc} strides={strides}")
    for o in model.outputs:
        print(f"[build]   output {o.name}: shape={o.shape}")

    if "keras" in args.include:
        keras_path = out_dir / "model.keras"
        model.save(keras_path)
        print(f"[save]  {keras_path}")

    if "saved_model" in args.include:
        sm_dir = out_dir / "saved_model"
        saved_model_export(model, sm_dir)
        print(f"[save]  {sm_dir}/  (SavedModel fp32)")

    if "tflite_fp16" in args.include:
        fp16_path = out_dir / "detector_fp16.tflite"
        fp16_to_tflite(model, fp16_path)
        size_mb = fp16_path.stat().st_size / (1024 * 1024)
        print(f"[save]  {fp16_path} ({size_mb:.2f} MB, fp16)")

    if not any(x in args.include for x in ("tflite_int8", "edgetpu")):
        return

    calib_gen = None
    if args.calib_images:
        print(f"[quant] calibration from {args.calib_images}")
        calib_gen = build_calibration_generator(args.calib_images, img_hw, n_calib=args.n_calib)
    elif args.data:
        d = parse_data_yaml(args.data)
        if d["train"] and Path(d["train"]).exists():
            print(f"[quant] calibration from data.yaml train -> {d['train']}")
            calib_gen = build_calibration_generator(d["train"], img_hw, n_calib=args.n_calib)
        else:
            print("[quant] data.yaml train missing; falling back to random calibration")
    else:
        print("[quant] no --data / --calib-images given; using random calibration")

    print(f"[quant] INT8 quantize (uint8 in, int8 out, n_calib={args.n_calib})")
    tflite_path = out_dir / "detector_int8.tflite"
    quantize_to_tflite(model, img_hw, tflite_path, calib_gen=calib_gen, n_calib=args.n_calib)
    size_mb = tflite_path.stat().st_size / (1024 * 1024)
    print(f"[quant] {tflite_path} ({size_mb:.2f} MB)")

    print("[ops]   TFLite op histogram:")
    hist = op_histogram_via_interpreter(tflite_path)
    for k, v in sorted(hist.items()):
        print(f"        {k:<28s} {v}")

    if "edgetpu" not in args.include:
        return

    print("[edgetpu] running edgetpu_compiler...")
    compiled, log = run_edgetpu_compiler(tflite_path)
    log_path = out_dir / "edgetpu_compile.log"
    log_path.write_text(log)
    summary = parse_compiler_summary(log)
    print(f"[edgetpu] log -> {log_path}")
    if compiled is None:
        print("[edgetpu] compile FAILED — see log above")
        for line in log.splitlines()[-30:]:
            print(f"          {line}")
        sys.exit(2)
    print(f"[edgetpu] compiled: {compiled}")
    print(f"[edgetpu] subgraphs={summary['subgraphs']} "
          f"total={summary['total']} edgetpu={summary['edgetpu']} cpu={summary['cpu']}")


if __name__ == "__main__":
    main()
