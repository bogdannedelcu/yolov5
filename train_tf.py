"""TF-native YOLOv5 trainer (no PyTorch graph / no ONNX).

Mirror of `train.py` (PyTorch) — same data pipeline (`utils/dataloaders.py`),
same loss semantics (`utils/tf_loss.py`), same optimizer mechanics
(`utils/tf_torch_utils.py`), same hyperparameter file format.

Outputs:
    {out}/best.weights.h5      EMA weights at best val loss
    {out}/last.weights.h5      EMA weights after last epoch
    {out}/training.log         per-epoch metrics (TSV)
    {out}/architecture.json    sidecar for `export_tf.py`
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.tf_yolo import DetectionModelTF
from utils.tf_loss import ComputeLossTF
from utils.tf_dataloaders import parse_data_yaml
from utils.tf_torch_utils import ModelEMA, SGDMomentum, AdamMomentum, split_param_groups, make_optimizer
from utils.tf_metrics import tf_validate
from utils.tf_autoanchor import maybe_recompute_anchors
from utils.dataloaders import create_dataloader
from utils.plots import plot_results, plot_labels


def parse_imgsz(values):
    if len(values) == 1:
        return (int(values[0]), int(values[0]))
    if len(values) == 2:
        return (int(values[0]), int(values[1]))
    raise ValueError("--imgsz takes 1 or 2 values (H W)")


def get_anchors_from_yaml(cfg_path: str) -> list:
    with open(cfg_path, "r") as f:
        d = yaml.safe_load(f)
    return d["anchors"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--hyp", default="data/hyps/hyp.scratch-low.yaml",
                    help="hyperparameters yaml — same file PT train.py would use")
    ap.add_argument("--imgsz", nargs="+", required=True)
    ap.add_argument("--nc", type=int, default=None)
    ap.add_argument("--act", default="silu", choices=["silu", "swish", "relu", "relu6", "leaky"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rect", action="store_true",
                    help="rectangular training (disables mosaic; matches PT --rect)")
    ap.add_argument("--no-augment", action="store_true",
                    help="disable all augmentations (override hyp file)")
    ap.add_argument("--cos-lr", action="store_true",
                    help="cosine LR scheduler instead of PT default linear")
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adam", "adamw"],
                    help="optimizer choice (mirrors PT --optimizer)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--ema-tau", type=float, default=2000)
    ap.add_argument("--patience", type=int, default=0,
                    help="early stop after N epochs without mAP@0.5:0.95 improvement (0=off)")
    ap.add_argument("--conf-thres", type=float, default=0.001,
                    help="val NMS confidence threshold")
    ap.add_argument("--iou-thres", type=float, default=0.6,
                    help="val NMS IoU threshold")
    ap.add_argument("--noautoanchor", action="store_true",
                    help="skip auto-anchor BPR check + kmeans recompute (mirrors PT --noautoanchor)")
    ap.add_argument("--save-period", type=int, default=0,
                    help="save checkpoint every N epochs as epoch{N}.weights.h5 (0=off)")
    ap.add_argument("--resume", default=None,
                    help="path to last.weights.h5 to resume from (auto-loads opt_state.npz next to it)")
    ap.add_argument("--no-tensorboard", action="store_true",
                    help="disable TensorBoard scalar logging")
    ap.add_argument("--amp", action="store_true",
                    help="mixed precision training (fp16 forward/backward, fp32 master). "
                         "GPU only — no-op on CPU. Mirrors PT autocast.")
    ap.add_argument("--multi-scale", action="store_true",
                    help="multi-scale training: per-batch random imgsz in "
                         "[0.5, 1.5]*imgsz, rounded to stride. Mirrors PT --multi-scale.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    img_hw = parse_imgsz(args.imgsz)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(args.seed)

    # Device announce — TF picks GPU automatically when available
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[device] TF version={tf.__version__}  GPUs detected={len(gpus)}"
          + (f"  ({[g.name for g in gpus]})" if gpus else ""))

    # AMP (mixed precision). Effective only on GPU with fp16 support.
    if args.amp:
        if gpus:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("[amp]  enabled mixed_float16 policy (forward/backward fp16, master fp32)")
        else:
            print("[amp]  --amp requested but no GPU detected — keeping fp32 (no-op)")

    # TensorBoard writer
    tb_writer = None
    if not args.no_tensorboard:
        tb_dir = out_dir / "tb"
        tb_writer = tf.summary.create_file_writer(str(tb_dir))
        print(f"[tb] writer -> {tb_dir} (run `tensorboard --logdir {tb_dir.parent}`)")

    print(f"[data] reading {args.data}")
    data = parse_data_yaml(args.data)
    nc = args.nc or data["nc"]
    print(f"[data] nc={nc} train={data['train']} val={data['val']}")

    print(f"[hyp]  loading {args.hyp}")
    with open(args.hyp, "r") as f:
        hyp = yaml.safe_load(f)
    print(f"[hyp]  mosaic={hyp.get('mosaic', 0)} mixup={hyp.get('mixup', 0)} "
          f"hsv_h={hyp.get('hsv_h', 0)} hsv_s={hyp.get('hsv_s', 0)} hsv_v={hyp.get('hsv_v', 0)} "
          f"scale={hyp.get('scale', 0)} fliplr={hyp.get('fliplr', 0)}")

    print(f"[build] cfg={args.cfg} imgsz={img_hw} act={args.act}"
          + (" [dynamic_shape for multi-scale]" if args.multi_scale else ""))
    detmodel = DetectionModelTF(
        cfg=args.cfg, imgsz_hw=img_hw, nc=nc, act=args.act, batch_size=None,
        dynamic_shape=args.multi_scale,
    )
    model = detmodel.model
    strides = detmodel.strides
    anchors = detmodel.anchors
    n_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"[build] params={n_params:,} strides={strides}")

    loss_fn = ComputeLossTF(
        anchors=anchors, strides=strides, nc=nc,
        hyp={
            "box": hyp.get("box", 0.05),
            "obj": hyp.get("obj", 1.0),
            "cls": hyp.get("cls", 0.5),
            "anchor_t": hyp.get("anchor_t", 4.0),
            "label_smoothing": hyp.get("label_smoothing", 0.0),
            "obj_pw": hyp.get("obj_pw", 1.0),
            "cls_pw": hyp.get("cls_pw", 1.0),
        },
    )

    lr0 = float(hyp.get("lr0", 0.01))
    lrf = float(hyp.get("lrf", 0.01))
    momentum = float(hyp.get("momentum", 0.937))
    weight_decay = float(hyp.get("weight_decay", 5e-4))
    warmup_epochs = float(hyp.get("warmup_epochs", 3.0))
    warmup_momentum = float(hyp.get("warmup_momentum", 0.8))
    warmup_bias_lr = float(hyp.get("warmup_bias_lr", 0.1))

    g_kernel, g_bn, g_bias = split_param_groups(model)
    print(f"[opt]  param groups: kernel={len(g_kernel)} bn={len(g_bn)} bias={len(g_bias)} (optimizer={args.optimizer})")
    # Adam/AdamW take wd inside; SGD applies wd in the loss (existing path).
    opt_wd = weight_decay if args.optimizer in ("adam", "adamw") else 0.0
    opt_main = make_optimizer(args.optimizer, g_kernel + g_bn,
                              lr=0.0, momentum=warmup_momentum, weight_decay=opt_wd)
    opt_bias = make_optimizer(args.optimizer, g_bias,
                              lr=warmup_bias_lr, momentum=warmup_momentum, weight_decay=0.0)

    use_cosine = bool(args.cos_lr)
    if use_cosine:
        def lf(x):
            return ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - lrf) + lrf
    else:
        def lf(x):
            return (1 - x / args.epochs) * (1 - lrf) + lrf
    print(f"[opt]  lr0={lr0} lrf={lrf} momentum={momentum} wd={weight_decay} "
          f"scheduler={'cosine' if use_cosine else 'linear'} "
          f"warmup_epochs={warmup_epochs} warmup_bias_lr={warmup_bias_lr}")

    pt_imgsz = max(img_hw)
    stride = int(max(strides))
    augment = (not args.no_augment)
    print(f"[data] PT loader: imgsz={pt_imgsz} stride={stride} augment={augment} rect={args.rect}")
    train_loader, train_ds = create_dataloader(
        path=str(data["train"]),
        imgsz=pt_imgsz, batch_size=args.batch, stride=stride,
        single_cls=False, hyp=hyp, augment=augment, cache=False,
        rect=args.rect, rank=-1, workers=args.workers, prefix="train: ",
        shuffle=not args.rect, seed=args.seed,
    )
    val_loader = val_ds = None
    if data["val"] and Path(data["val"]).exists():
        val_loader, val_ds = create_dataloader(
            path=str(data["val"]),
            imgsz=pt_imgsz, batch_size=args.batch, stride=stride,
            single_cls=False, hyp=hyp, augment=False, cache=False,
            rect=True, rank=-1, workers=args.workers, prefix="val: ",
            shuffle=False, seed=args.seed,
        )
    print(f"[data] train={len(train_ds)} val={len(val_ds) if val_ds else 0}")

    # ---- plot label distribution (Phase 3) -----------------------------------
    def _names_list(n):
        if isinstance(n, dict):
            return [n[k] for k in sorted(n.keys())]
        if isinstance(n, list):
            return list(n)
        return []
    try:
        if hasattr(train_ds, "labels") and len(train_ds.labels):
            labels_np = np.concatenate(train_ds.labels, 0)  # (N, 5) cls + xywh normalized
            plot_labels(labels_np, names=_names_list(data.get("names")), save_dir=out_dir)
            print(f"[plot] labels.jpg / labels_correlogram.jpg -> {out_dir}")
    except Exception as e:
        print(f"[plot] plot_labels failed: {e}")

    # ---- auto-anchor (Phase 2) -----------------------------------------------
    if not args.noautoanchor:
        print("[anchor] running BPR check (utils.autoanchor.check_anchors)...")
        anchors, changed = maybe_recompute_anchors(
            train_ds, anchors, strides, imgsz=pt_imgsz, thr=4.0,
        )
        if changed:
            print(f"[anchor] anchors updated by kmeans: {anchors}")
            # rebuild loss with updated anchors
            loss_fn = ComputeLossTF(
                anchors=anchors, strides=strides, nc=nc,
                hyp={
                    "box": hyp.get("box", 0.05), "obj": hyp.get("obj", 1.0),
                    "cls": hyp.get("cls", 0.5), "anchor_t": hyp.get("anchor_t", 4.0),
                    "label_smoothing": hyp.get("label_smoothing", 0.0),
                    "obj_pw": hyp.get("obj_pw", 1.0), "cls_pw": hyp.get("cls_pw", 1.0),
                },
            )
            detmodel.anchors = anchors
        else:
            print("[anchor] BPR ok — keeping YAML anchors")

    def pt_batch_to_tf(batch):
        imgs, targets, _paths, _shapes = batch
        imgs_np = imgs.numpy()
        imgs_np = np.transpose(imgs_np, (0, 2, 3, 1))
        imgs_tf = tf.constant(imgs_np.astype(np.float32) / 255.0)
        tgt_tf = tf.constant(targets.numpy().astype(np.float32))
        return imgs_tf, tgt_tf

    # Multi-scale: per-batch random imgsz in [0.5, 1.5]*imgsz, multiple of stride.
    # Targets are normalized so they don't need rescaling — only resize the image.
    # Caveat: model is built at fixed imgsz; we accept variable HxW because all
    # convolutions are shape-agnostic. If model.input was bound to a specific
    # H, W via Input(shape=(H,W,3)), Keras will reject other shapes — we built
    # with batch_size=None, but H/W are still pinned. In multi-scale mode we
    # rebuild inputs with shape=(None, None, 3) below.
    ms_active = args.multi_scale
    ms_min = int(0.5 * pt_imgsz / stride) * stride
    ms_max = int(1.5 * pt_imgsz / stride) * stride
    ms_rng = np.random.default_rng(args.seed)
    if ms_active:
        print(f"[multi-scale] enabled: random imgsz in [{ms_min}, {ms_max}] step={stride}")

    main_vars = g_kernel + g_bn

    sgd_wd_in_loss = (args.optimizer == "sgd")  # only SGD path uses loss-side L2
    # Static loss scale for AMP gradient stability (avoids fp16 underflow).
    # PT uses dynamic GradScaler; for typical v5 loss magnitudes (~1-10) a
    # fixed 1024× scale is ample. Disabled when AMP is off.
    amp_active = args.amp and bool(gpus)
    loss_scale = tf.constant(1024.0 if amp_active else 1.0, dtype=tf.float32)

    def train_step(images, targets):
        with tf.GradientTape() as tape:
            preds = model(images, training=True)
            total, parts = loss_fn(preds, targets)
            if sgd_wd_in_loss:
                wd_terms = [tf.nn.l2_loss(v) for v in g_kernel]
                wd = tf.add_n(wd_terms) if wd_terms else tf.constant(0.0)
                loss = total + weight_decay * wd
            else:
                loss = total  # Adam/AdamW handle wd internally
            scaled_loss = loss * loss_scale
        grads_main = tape.gradient(scaled_loss, main_vars + g_bias)
        if amp_active:
            grads_main = [g / loss_scale if g is not None else None for g in grads_main]
        gm, gb = grads_main[: len(main_vars)], grads_main[len(main_vars):]
        opt_main.apply_gradients(zip(gm, main_vars))
        opt_bias.apply_gradients(zip(gb, g_bias))
        return total, parts

    def val_loss():
        if val_loader is None:
            return None
        running = np.zeros(4, dtype=np.float64)
        n = 0
        for batch in val_loader:
            images, targets = pt_batch_to_tf(batch)
            preds = model(images, training=False)
            total, (lbox, lobj, lcls) = loss_fn(preds, targets)
            running += [float(total), float(lbox), float(lobj), float(lcls)]
            n += 1
        return None if n == 0 else running / n

    ema = ModelEMA(model, decay=args.ema_decay, tau=args.ema_tau)
    print(f"[ema]  decay={args.ema_decay} tau={args.ema_tau}")

    log_path = out_dir / "training.log"
    log_path.write_text(
        "epoch\tbatches\ttrain_total\ttrain_box\ttrain_obj\ttrain_cls\t"
        "val_total\tval_box\tval_obj\tval_cls\tP\tR\tmAP50\tmAP50_95\tsec\n"
    )
    # PT-format results.csv so `utils.plots.plot_results` works directly
    csv_path = out_dir / "results.csv"
    csv_path.write_text(
        "               epoch,      train/box_loss,      train/obj_loss,      train/cls_loss,"
        "   metrics/precision,      metrics/recall,     metrics/mAP_0.5,metrics/mAP_0.5:0.95,"
        "        val/box_loss,        val/obj_loss,        val/cls_loss,"
        "               x/lr0,               x/lr1,               x/lr2\n"
    )
    best_fitness = -1.0
    epochs_since_improve = 0
    last_path = out_dir / "last.weights.h5"
    best_path = out_dir / "best.weights.h5"

    nb_per_epoch = len(train_ds) // args.batch + (1 if len(train_ds) % args.batch else 0)
    nw = max(round(warmup_epochs * nb_per_epoch), 100)
    print(f"[opt]  nb_per_epoch={nb_per_epoch} warmup_iters={nw}")

    # ---- resume (Phase 3) ----------------------------------------------------
    start_epoch = 0
    if args.resume:
        rp = Path(args.resume)
        opt_state_path = rp.parent / "opt_state.npz"
        print(f"[resume] loading weights from {rp}")
        detmodel.load_weights(rp)
        if opt_state_path.exists():
            print(f"[resume] loading optimizer state from {opt_state_path}")
            state = np.load(opt_state_path, allow_pickle=True)
            for v, arr in zip(opt_main.velocities, state["vel_main"]):
                v.assign(arr)
            for v, arr in zip(opt_bias.velocities, state["vel_bias"]):
                v.assign(arr)
            for s, arr in zip(ema.shadow, state["ema_shadow"]):
                s.assign(arr)
            ema.updates = int(state["ema_updates"])
            start_epoch = int(state["epoch"]) + 1
            best_fitness = float(state["best_fitness"])
            print(f"[resume] resumed at epoch {start_epoch}, best_fitness={best_fitness:.4f}")
        else:
            print(f"[resume] no opt_state.npz found — only weights restored")

    def _save_opt_state(ep_idx: int):
        np.savez(
            out_dir / "opt_state.npz",
            vel_main=np.array([v.numpy() for v in opt_main.velocities], dtype=object),
            vel_bias=np.array([v.numpy() for v in opt_bias.velocities], dtype=object),
            ema_shadow=np.array([s.numpy() for s in ema.shadow], dtype=object),
            ema_updates=np.int64(ema.updates),
            epoch=np.int64(ep_idx),
            best_fitness=np.float64(best_fitness),
        )

    for ep in range(start_epoch, args.epochs):
        t0 = time.time()
        running = np.zeros(4, dtype=np.float64)
        nb = 0
        cur_lf = lf(ep)
        for batch in train_loader:
            ni = ep * nb_per_epoch + nb
            if ni <= nw:
                xi = [0, nw]
                main_lr = float(np.interp(ni, xi, [0.0, lr0 * cur_lf]))
                bias_lr = float(np.interp(ni, xi, [warmup_bias_lr, lr0 * cur_lf]))
                cur_mom = float(np.interp(ni, xi, [warmup_momentum, momentum]))
            else:
                main_lr = lr0 * cur_lf
                bias_lr = lr0 * cur_lf
                cur_mom = momentum
            opt_main.lr.assign(main_lr)
            opt_bias.lr.assign(bias_lr)
            opt_main.momentum.assign(cur_mom)
            opt_bias.momentum.assign(cur_mom)

            images, targets = pt_batch_to_tf(batch)
            if ms_active:
                ns = int(ms_rng.integers(ms_min // stride, ms_max // stride + 1)) * stride
                if ns != images.shape[1] or ns != images.shape[2]:
                    images = tf.image.resize(images, (ns, ns), method="bilinear")
            total, (lbox, lobj, lcls) = train_step(images, targets)
            ema.update()
            running += [float(total), float(lbox), float(lobj), float(lcls)]

            if tb_writer is not None:
                with tb_writer.as_default():
                    tf.summary.scalar("step/lr_main", main_lr, step=ni)
                    tf.summary.scalar("step/lr_bias", bias_lr, step=ni)
                    tf.summary.scalar("step/momentum", cur_mom, step=ni)
                    tf.summary.scalar("step/loss_total", float(total), step=ni)
                    tf.summary.scalar("step/loss_box", float(lbox), step=ni)
                    tf.summary.scalar("step/loss_obj", float(lobj), step=ni)
                    tf.summary.scalar("step/loss_cls", float(lcls), step=ni)

            nb += 1
            if nb >= nb_per_epoch:
                break
        train_avg = running / max(nb, 1)

        # validation with EMA weights swapped in
        backup = ema.swap_in()
        val_avg = val_loss()
        # mAP via host decode + PT NMS + ap_per_class
        if val_loader is not None:
            mp, mr, map50, map50_95, _ = tf_validate(
                model, val_loader, anchors, strides, nc,
                conf_thres=args.conf_thres, iou_thres=args.iou_thres,
                names=data["names"],
            )
        else:
            mp = mr = map50 = map50_95 = 0.0
        ema.restore(backup)

        sec = time.time() - t0
        msg = (f"[ep {ep+1:>3}/{args.epochs}] "
               f"batches={nb} "
               f"train: total={train_avg[0]:.3f} box={train_avg[1]:.3f} "
               f"obj={train_avg[2]:.3f} cls={train_avg[3]:.3f}"
               + (f"  val: total={val_avg[0]:.3f} box={val_avg[1]:.3f} "
                  f"obj={val_avg[2]:.3f} cls={val_avg[3]:.3f}"
                  if val_avg is not None else "")
               + f"  | P={mp:.4f} R={mr:.4f} mAP50={map50:.4f} mAP50-95={map50_95:.4f}"
               + f"  ({sec:.1f}s)")
        print(msg)
        with log_path.open("a") as f:
            row = [ep + 1, nb, *train_avg.tolist()]
            row += list(val_avg.tolist()) if val_avg is not None else [float("nan")] * 4
            row += [mp, mr, map50, map50_95, sec]
            f.write("\t".join(f"{x}" for x in row) + "\n")

        # epoch-level TensorBoard scalars
        if tb_writer is not None:
            fitness = 0.1 * map50 + 0.9 * map50_95
            with tb_writer.as_default():
                tf.summary.scalar("epoch/train_total", float(train_avg[0]), step=ep + 1)
                tf.summary.scalar("epoch/train_box", float(train_avg[1]), step=ep + 1)
                tf.summary.scalar("epoch/train_obj", float(train_avg[2]), step=ep + 1)
                tf.summary.scalar("epoch/train_cls", float(train_avg[3]), step=ep + 1)
                if val_avg is not None:
                    tf.summary.scalar("epoch/val_total", float(val_avg[0]), step=ep + 1)
                    tf.summary.scalar("epoch/val_box", float(val_avg[1]), step=ep + 1)
                    tf.summary.scalar("epoch/val_obj", float(val_avg[2]), step=ep + 1)
                    tf.summary.scalar("epoch/val_cls", float(val_avg[3]), step=ep + 1)
                tf.summary.scalar("metrics/precision", mp, step=ep + 1)
                tf.summary.scalar("metrics/recall", mr, step=ep + 1)
                tf.summary.scalar("metrics/mAP_0.5", map50, step=ep + 1)
                tf.summary.scalar("metrics/mAP_0.5:0.95", map50_95, step=ep + 1)
                tf.summary.scalar("metrics/fitness", fitness, step=ep + 1)
                tf.summary.scalar("epoch/sec", sec, step=ep + 1)
                tb_writer.flush()

        # PT-format results.csv (epoch is 0-indexed in PT csv)
        with csv_path.open("a") as f:
            tb = train_avg[1]; to = train_avg[2]; tc = train_avg[3]
            vb, vo, vc = (val_avg[1], val_avg[2], val_avg[3]) if val_avg is not None else (0.0, 0.0, 0.0)
            f.write(
                f"{ep:>20},{tb:>20.6g},{to:>20.6g},{tc:>20.6g},"
                f"{mp:>20.6g},{mr:>20.6g},{map50:>20.6g},{map50_95:>20.6g},"
                f"{vb:>20.6g},{vo:>20.6g},{vc:>20.6g},"
                f"{lr0 * cur_lf:>20.6g},{lr0 * cur_lf:>20.6g},{lr0 * cur_lf:>20.6g}\n"
            )

        # save: best by fitness = 0.1*mAP50 + 0.9*mAP50_95 (PT convention from utils.metrics.fitness)
        fitness = 0.1 * map50 + 0.9 * map50_95
        backup = ema.swap_in()
        model.save_weights(last_path)
        if fitness > best_fitness:
            best_fitness = fitness
            epochs_since_improve = 0
            model.save_weights(best_path)
            print(f"        -> new best fitness={fitness:.4f} (mAP50={map50:.4f}, mAP50-95={map50_95:.4f}); saved {best_path}")
        else:
            epochs_since_improve += 1
        # save-period checkpoint
        if args.save_period > 0 and (ep + 1) % args.save_period == 0:
            ckpt = out_dir / f"epoch{ep+1}.weights.h5"
            model.save_weights(ckpt)
            print(f"        save-period -> {ckpt}")
        ema.restore(backup)
        # save optimizer / EMA / epoch state for --resume
        _save_opt_state(ep)
        print(f"        main_lr={opt_main.lr.numpy():.5f} "
              f"bias_lr={opt_bias.lr.numpy():.5f} mom={opt_main.momentum.numpy():.3f}")

        if args.patience > 0 and epochs_since_improve >= args.patience:
            print(f"[early-stop] no fitness improvement for {args.patience} epochs — stopping.")
            break

    sidecar = {
        "cfg": str(Path(args.cfg).resolve()),
        "imgsz_hw": list(img_hw),
        "nc": nc,
        "act": args.act,
        "strides": strides,
        "anchors": anchors,
    }
    (out_dir / "architecture.json").write_text(json.dumps(sidecar, indent=2))

    # Phase 3: native results.png plot via PT util
    try:
        plot_results(file=csv_path)
        print(f"[plot] results.png -> {out_dir / 'results.png'}")
    except Exception as e:
        print(f"[plot] failed: {e}")

    # Final val on best.weights.h5 with PR curve + confusion matrix plots
    if val_loader is not None and best_path.exists():
        print(f"[final-val] running on {best_path} with plots enabled...")
        try:
            detmodel.load_weights(best_path)
            mp, mr, map50, map50_95, _ = tf_validate(
                detmodel.model, val_loader, anchors, strides, nc,
                conf_thres=args.conf_thres, iou_thres=args.iou_thres,
                names=data["names"], plot=True, save_dir=out_dir,
                confusion_matrix=True,
            )
            print(f"[final-val] best: P={mp:.4f} R={mr:.4f} "
                  f"mAP50={map50:.4f} mAP50-95={map50_95:.4f}")
            print(f"[final-val] PR_curve.png / confusion_matrix.png -> {out_dir}")
        except Exception as e:
            print(f"[final-val] failed: {e}")

    print(f"[done] last={last_path}  best={best_path}  sidecar={out_dir/'architecture.json'}")


if __name__ == "__main__":
    main()
