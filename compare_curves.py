"""Print TF vs PT loss curves side-by-side for the iarna 10-ep run."""
import csv
from pathlib import Path

tf_log = Path("runs/tf_native/iarna_10ep/training.log")
pt_csv = Path("runs/pt_train/iarna_10ep/results.csv")

# parse TF (TSV)
tf_rows = []
with open(tf_log) as f:
    next(f)
    for line in f:
        c = line.strip().split("\t")
        tf_rows.append({
            "ep": int(c[0]),
            "tr_total": float(c[2]),
            "tr_box": float(c[3]),
            "tr_obj": float(c[4]),
            "v_total": float(c[6]),
            "v_box": float(c[7]),
            "v_obj": float(c[8]),
        })

# parse PT (CSV)
pt_rows = []
with open(pt_csv) as f:
    rdr = csv.DictReader(f, skipinitialspace=True)
    for row in rdr:
        row = {k.strip(): v.strip() for k, v in row.items()}
        ep = int(row["epoch"]) + 1
        b = float(row["train/box_loss"])
        o = float(row["train/obj_loss"])
        vb = float(row["val/box_loss"])
        vo = float(row["val/obj_loss"])
        # PT-style total scaled by bs=8 like TF's reporting:
        tr_total = (b + o) * 8
        v_total = (vb + vo) * 8
        pt_rows.append({"ep": ep, "tr_total": tr_total, "tr_box": b, "tr_obj": o,
                        "v_total": v_total, "v_box": vb, "v_obj": vo,
                        "P": float(row["metrics/precision"]),
                        "R": float(row["metrics/recall"]),
                        "mAP50": float(row["metrics/mAP_0.5"]),
                        "mAP50_95": float(row["metrics/mAP_0.5:0.95"])})

print("=" * 100)
print("TRAIN LOSS (total = (lbox + lobj + lcls)*bs)")
print(f"{'ep':>4} | {'TF tot':>9} {'TF box':>8} {'TF obj':>8} | {'PT tot':>9} {'PT box':>8} {'PT obj':>8} | {'Δtot':>7}")
for t, p in zip(tf_rows, pt_rows):
    delta = t["tr_total"] - p["tr_total"]
    print(f"{t['ep']:>4} | {t['tr_total']:>9.4f} {t['tr_box']:>8.4f} {t['tr_obj']:>8.4f} | "
          f"{p['tr_total']:>9.4f} {p['tr_box']:>8.4f} {p['tr_obj']:>8.4f} | {delta:>+7.4f}")

print()
print("=" * 100)
print("VAL LOSS")
print(f"{'ep':>4} | {'TF tot':>9} {'TF box':>8} {'TF obj':>8} | {'PT tot':>9} {'PT box':>8} {'PT obj':>8} | {'Δtot':>7}")
for t, p in zip(tf_rows, pt_rows):
    delta = t["v_total"] - p["v_total"]
    print(f"{t['ep']:>4} | {t['v_total']:>9.4f} {t['v_box']:>8.4f} {t['v_obj']:>8.4f} | "
          f"{p['v_total']:>9.4f} {p['v_box']:>8.4f} {p['v_obj']:>8.4f} | {delta:>+7.4f}")

print()
print("=" * 100)
print("PT-only metrics (TF has no in-training mAP loop yet)")
print(f"{'ep':>4} | {'P':>10} {'R':>10} {'mAP50':>10} {'mAP50-95':>10}")
for p in pt_rows:
    print(f"{p['ep']:>4} | {p['P']:>10.5f} {p['R']:>10.5f} {p['mAP50']:>10.5f} {p['mAP50_95']:>10.5f}")
