"""Compare TF native vs PT exported TFLite/EdgeTPU models on the same image."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import cv2


def parse_compile_log(log_path: Path) -> dict:
    if not log_path.exists():
        return {}
    txt = log_path.read_text()
    out = {"mapped": 0, "cpu": 0, "subgraphs": None, "total": None}
    for line in txt.splitlines():
        s = line.strip()
        if "Mapped to Edge TPU" in s:
            parts = s.split()
            try:
                out["mapped"] += int(parts[1])
            except Exception:
                pass
        elif s.endswith("CPU") or "More than one subgraph" in s or "unspecified limitation" in s:
            parts = s.split()
            try:
                out["cpu"] += int(parts[1])
            except Exception:
                pass
        elif s.startswith("Number of Edge TPU subgraphs"):
            out["subgraphs"] = int(s.split(":")[1])
        elif s.startswith("Total number of operations"):
            out["total"] = int(s.split(":")[1])
    return out


def op_histogram(tflite_path: Path) -> dict:
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    h: dict = {}
    for d in interp._get_ops_details():
        h[d["op_name"]] = h.get(d["op_name"], 0) + 1
    return h


def io_spec(tflite_path: Path) -> dict:
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = [(d["name"], d["dtype"].__name__, list(d["shape"])) for d in interp.get_input_details()]
    out = [(d["name"], d["dtype"].__name__, list(d["shape"])) for d in interp.get_output_details()]
    return {"inputs": inp, "outputs": out}


def run_inference(tflite_path: Path, image_path: Path, n_runs: int = 5) -> dict:
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()

    H, W = inp["shape"][1], inp["shape"][2]
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]
    r = min(H / h0, W / w0)
    new = (int(round(w0 * r)), int(round(h0 * r)))
    img = cv2.resize(img, new, interpolation=cv2.INTER_LINEAR)
    pad_y = (H - new[1]) // 2
    pad_x = (W - new[0]) // 2
    canvas = np.full((H, W, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new[1], pad_x:pad_x + new[0]] = img

    if inp["dtype"] == np.uint8:
        x = canvas[None, ...].astype(np.uint8)
    elif inp["dtype"] == np.int8:
        scale, zp = inp["quantization"]
        x = ((canvas.astype(np.float32) / 255.0) / scale + zp).astype(np.int8)[None, ...]
    else:
        x = (canvas.astype(np.float32) / 255.0)[None, ...]

    # warmup + time
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    times = []
    for _ in range(n_runs):
        interp.set_tensor(inp["index"], x)
        t0 = time.time()
        interp.invoke()
        times.append((time.time() - t0) * 1000)
    out_tensors = []
    for od in outs:
        t = interp.get_tensor(od["index"])
        if od["dtype"] in (np.int8, np.uint8):
            scale, zp = od["quantization"]
            tf32 = (t.astype(np.float32) - zp) * scale
        else:
            tf32 = t.astype(np.float32)
        out_tensors.append((od["name"], list(t.shape), tf32))

    return {"times_ms": times, "outputs": out_tensors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf-tflite", required=True)
    ap.add_argument("--pt-tflite", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--tf-log")
    ap.add_argument("--pt-log")
    args = ap.parse_args()

    tf_p = Path(args.tf_tflite)
    pt_p = Path(args.pt_tflite)
    img_p = Path(args.image)

    print("=" * 78)
    print("FILE SIZE")
    print(f"  TF native: {tf_p.stat().st_size / 1024:.1f} KiB  ({tf_p.name})")
    print(f"  PT export: {pt_p.stat().st_size / 1024:.1f} KiB  ({pt_p.name})")

    print()
    print("=" * 78)
    print("EDGETPU COMPILE SUMMARY")
    if args.tf_log:
        s = parse_compile_log(Path(args.tf_log))
        print(f"  TF native: subgraphs={s.get('subgraphs')} total={s.get('total')} "
              f"mapped={s.get('mapped')} cpu={s.get('cpu')}")
    if args.pt_log:
        s = parse_compile_log(Path(args.pt_log))
        print(f"  PT export: subgraphs={s.get('subgraphs')} total={s.get('total')} "
              f"mapped={s.get('mapped')} cpu={s.get('cpu')}")

    print()
    print("=" * 78)
    print("OP HISTOGRAM (pre-EdgeTPU INT8 TFLite)")
    th = op_histogram(tf_p.parent / tf_p.name.replace("_edgetpu", "")) if "_edgetpu" in tf_p.name else op_histogram(tf_p)
    ph = op_histogram(pt_p.parent / pt_p.name.replace("_edgetpu", "")) if "_edgetpu" in pt_p.name else op_histogram(pt_p)
    keys = sorted(set(th) | set(ph))
    print(f"  {'OP':<28s} {'TF':>5s} {'PT':>5s} {'Δ':>5s}")
    for k in keys:
        a, b = th.get(k, 0), ph.get(k, 0)
        print(f"  {k:<28s} {a:>5d} {b:>5d} {b - a:>+5d}")
    print(f"  {'TOTAL':<28s} {sum(th.values()):>5d} {sum(ph.values()):>5d} "
          f"{sum(ph.values()) - sum(th.values()):>+5d}")

    # for I/O + inference, use the non-edgetpu version (compiles on CPU runtime).
    tf_int = tf_p.parent / tf_p.name.replace("_edgetpu", "")
    pt_int = pt_p.parent / pt_p.name.replace("_edgetpu", "")

    print()
    print("=" * 78)
    print("I/O SPEC (CPU TFLite)")
    for name, p in (("TF native", tf_int), ("PT export", pt_int)):
        spec = io_spec(p)
        print(f"  {name}:")
        for n, d, s in spec["inputs"]:
            print(f"    in  [{d}] {s}  {n}")
        for n, d, s in spec["outputs"]:
            print(f"    out [{d}] {s}  {n}")

    print()
    print("=" * 78)
    print(f"INFERENCE on {img_p.name} (CPU TFLite)")
    tf_r = run_inference(tf_int, img_p)
    pt_r = run_inference(pt_int, img_p)
    print(f"  TF native: latency CPU TFLite = {np.mean(tf_r['times_ms']):.1f} ± {np.std(tf_r['times_ms']):.1f} ms")
    print(f"  PT export: latency CPU TFLite = {np.mean(pt_r['times_ms']):.1f} ± {np.std(pt_r['times_ms']):.1f} ms")

    print()
    print("OUTPUT STATS (dequantized fp32)")
    for label, r in (("TF native", tf_r), ("PT export", pt_r)):
        print(f"  {label}:")
        for n, s, t in r["outputs"]:
            print(f"    {n:<48s} shape={s}  mean={t.mean():+.4f}  std={t.std():.4f}  "
                  f"min={t.min():+.3f}  max={t.max():+.3f}")


if __name__ == "__main__":
    main()
