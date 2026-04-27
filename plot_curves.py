"""Plot TF vs PT training/val loss curves side-by-side.

Reads:
- TF training log: TSV from train_tf_native.py
- PT results.csv: standard yolov5 train.py output

Produces a single PNG with 6 subplots: train/val × total/box/obj.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_tf_log(path: Path) -> dict:
    rows = {"ep": [], "tr_total": [], "tr_box": [], "tr_obj": [],
            "v_total": [], "v_box": [], "v_obj": []}
    with open(path) as f:
        next(f)  # header
        for line in f:
            c = line.strip().split("\t")
            if len(c) < 11:
                continue
            rows["ep"].append(int(c[0]))
            rows["tr_total"].append(float(c[2]))
            rows["tr_box"].append(float(c[3]))
            rows["tr_obj"].append(float(c[4]))
            rows["v_total"].append(float(c[6]))
            rows["v_box"].append(float(c[7]))
            rows["v_obj"].append(float(c[8]))
    return rows


def parse_pt_csv(path: Path, bs: int = 8) -> dict:
    rows = {"ep": [], "tr_total": [], "tr_box": [], "tr_obj": [],
            "v_total": [], "v_box": [], "v_obj": [],
            "P": [], "R": [], "mAP50": [], "mAP50_95": []}
    with open(path) as f:
        rdr = csv.DictReader(f, skipinitialspace=True)
        for row in rdr:
            row = {k.strip(): v.strip() for k, v in row.items()}
            ep = int(row["epoch"]) + 1
            tb = float(row["train/box_loss"])
            to = float(row["train/obj_loss"])
            vb = float(row["val/box_loss"])
            vo = float(row["val/obj_loss"])
            rows["ep"].append(ep)
            rows["tr_box"].append(tb)
            rows["tr_obj"].append(to)
            rows["tr_total"].append((tb + to) * bs)
            rows["v_box"].append(vb)
            rows["v_obj"].append(vo)
            rows["v_total"].append((vb + vo) * bs)
            rows["P"].append(float(row["metrics/precision"]))
            rows["R"].append(float(row["metrics/recall"]))
            rows["mAP50"].append(float(row["metrics/mAP_0.5"]))
            rows["mAP50_95"].append(float(row["metrics/mAP_0.5:0.95"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf-log", required=True)
    ap.add_argument("--pt-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="TF native vs PyTorch — iarna")
    args = ap.parse_args()

    tf = parse_tf_log(Path(args.tf_log))
    pt = parse_pt_csv(Path(args.pt_csv))

    has_metrics = bool(pt.get("mAP50"))
    n_rows = 3 if has_metrics else 2
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows), squeeze=False)
    fig.suptitle(args.title, fontsize=14)

    def line(ax, x_tf, y_tf, x_pt, y_pt, ylabel, ymin=None):
        ax.plot(x_tf, y_tf, "o-", label="TF native", color="#1f77b4")
        ax.plot(x_pt, y_pt, "s-", label="PyTorch", color="#d62728")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        if ymin is not None:
            ax.set_ylim(bottom=ymin)

    line(axes[0][0], tf["ep"], tf["tr_total"], pt["ep"], pt["tr_total"], "train total")
    line(axes[0][1], tf["ep"], tf["tr_box"], pt["ep"], pt["tr_box"], "train box")
    line(axes[0][2], tf["ep"], tf["tr_obj"], pt["ep"], pt["tr_obj"], "train obj")

    line(axes[1][0], tf["ep"], tf["v_total"], pt["ep"], pt["v_total"], "val total")
    line(axes[1][1], tf["ep"], tf["v_box"], pt["ep"], pt["v_box"], "val box")
    line(axes[1][2], tf["ep"], tf["v_obj"], pt["ep"], pt["v_obj"], "val obj")

    if has_metrics:
        # PT only — TF doesn't compute these in-loop yet
        ax = axes[2][0]
        ax.plot(pt["ep"], pt["P"], "s-", label="P", color="#2ca02c")
        ax.plot(pt["ep"], pt["R"], "s-", label="R", color="#9467bd")
        ax.set_xlabel("epoch")
        ax.set_ylabel("PT precision / recall")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2][1]
        ax.plot(pt["ep"], pt["mAP50"], "s-", label="mAP@0.5", color="#2ca02c")
        ax.plot(pt["ep"], pt["mAP50_95"], "s-", label="mAP@0.5:0.95", color="#9467bd")
        ax.set_xlabel("epoch")
        ax.set_ylabel("PT mAP")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # epoch-by-epoch delta on val total
        ax = axes[2][2]
        n = min(len(tf["ep"]), len(pt["ep"]))
        delta = [tf["v_total"][i] - pt["v_total"][i] for i in range(n)]
        rel = [100.0 * delta[i] / pt["v_total"][i] for i in range(n)]
        ax.plot(tf["ep"][:n], rel, "o-", color="#ff7f0e")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel("(TF − PT) / PT  val total  [%]")
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved: {out}")

    # also a compact summary
    print()
    print(f"{'ep':>4} | {'TFtr':>7} {'PTtr':>7} {'Δ%':>7} | {'TFv':>7} {'PTv':>7} {'Δ%':>7}")
    n = min(len(tf["ep"]), len(pt["ep"]))
    for i in range(n):
        e = tf["ep"][i]
        rt = 100 * (tf["tr_total"][i] - pt["tr_total"][i]) / pt["tr_total"][i]
        rv = 100 * (tf["v_total"][i] - pt["v_total"][i]) / pt["v_total"][i]
        print(f"{e:>4} | {tf['tr_total'][i]:>7.3f} {pt['tr_total'][i]:>7.3f} {rt:>+6.1f}% | "
              f"{tf['v_total'][i]:>7.3f} {pt['v_total'][i]:>7.3f} {rv:>+6.1f}%")


if __name__ == "__main__":
    main()
