# Agent guide — TensorFlow port of YOLOv5

This file is the operating manual for any agent (or human) working on this
repository. The repository's primary goal is to **port the YOLOv5 PyTorch
training/export pipeline to TensorFlow** while **reusing as much of the
existing PT code as possible**, so that the same YAMLs, hyperparameters,
dataloaders, augmentations, NMS, mAP computation, and plotting code drive
both backends.

## 1. Mission

- **Primary**: a TF-native NHWC training + export pipeline that consumes the
  standard `models/yolov5*.yaml` files and produces 100% Edge-TPU-mappable
  INT8 TFLite models (no PyTorch graph, no ONNX in the path).
- **Constraint from the user**: never go through PyTorch → ONNX → TFLite for
  the TF deployment artefact. PT is channel-first, TF / Edge TPU are
  channel-last; the ONNX converter inserts NCHW↔NHWC adapters
  (TRANSPOSE / RESHAPE / PAD) which fragment the Edge TPU partition. See
  [experiment.md](experiment.md) for the empirical proof.
- **Secondary**: keep functional parity with the PT side
  (loss curves, mAP, optimizer mechanics, augmentation pipeline) so that
  a model trained on either backend lands at comparable accuracy.

## 2. Repository layout

The TF code mirrors the PT layout 1:1. Each PT file `<x>.py` has a TF
counterpart `tf_<x>.py` in the same directory.

```
yolov5/
├── train.py             (PT)            ←→  train_tf.py             (TF)
├── val.py               (PT)            ←→  val_tf.py               (TF)
├── detect.py            (PT)            ←→  detect_tf.py            (TF)
├── export.py            (PT, legacy)    ←→  export_tf.py            (TF)
├── benchmarks.py        (PT, marginal)
├── compare_models.py / compare_curves.py / plot_curves.py    (cross-backend tooling, root)
│
├── models/
│   ├── common.py        (PT layers)     ←→  tf_common.py            (TF layers)
│   ├── yolo.py          (PT model)      ←→  tf_yolo.py              (TF model + parser + DetectionModelTF)
│   ├── experimental.py  (PT)            ←→  tf_experimental.py      (TFMixConv2d, TFEnsemble, attempt_load_tf)
│   ├── tf.py            (legacy PT-driven TF; only used by export.py)
│   └── *.yaml           (shared between backends)
│
└── utils/
    ├── general.py       (PT)            ←→  reused directly (NMS, xywh2xyxy, scale_boxes)
    ├── loss.py          (PT)            ←→  tf_loss.py              (ComputeLossTF)
    ├── metrics.py       (PT)            ←→  reused directly (ap_per_class, ConfusionMatrix, fitness)
    ├── torch_utils.py   (PT)            ←→  tf_torch_utils.py       (ModelEMA, SGDMomentum, AdamMomentum, split_param_groups)
    ├── dataloaders.py   (PT)            ←→  tf_dataloaders.py       (calibration generator + parse_data_yaml; rest reused directly)
    ├── autoanchor.py    (PT)            ←→  tf_autoanchor.py        (thin stub adapter around check_anchors / kmean_anchors)
    ├── plots.py         (PT)            ←→  reused directly (plot_results, plot_labels, Annotator)
    ├── augmentations.py (PT)            ←→  reused directly (letterbox, mosaic etc. via LoadImagesAndLabels)
    └── tf_metrics.py                    (NEW: host_decode + tf_validate + TFLiteModelWrapper)
```

A TF script with no `tf_` counterpart in PT means it is genuinely TF-only
(e.g. `compare_models.py` cross-backend tooling).

## 3. Decision rule: reuse PT vs write TF-native

Apply this rule at every change.

### Reuse PT directly when

- The function is **pure NumPy / pure Python**
  (e.g. `ap_per_class`, `fitness`, `xywh2xyxy`, `scale_boxes`, `letterbox`).
- The function accepts **torch tensors at the boundary** but the body is
  arithmetic with no graph/backprop semantics
  (e.g. `non_max_suppression`, `process_batch`, `ConfusionMatrix.process_batch`).
  Convert TF tensor → NumPy → torch only at entry, NumPy at exit.
- The function takes a **dataset object with NumPy attributes**
  (e.g. `kmean_anchors(dataset, n, img_size)` reads `dataset.shapes` and
  `dataset.labels` as NumPy arrays — feed our PT `LoadImagesAndLabels` directly).
- The function consumes a **disk artefact** in a defined format
  (e.g. `plot_results(csv_path)` reads `results.csv` and produces PNGs —
  if we write the CSV in PT format, the plotter works without any TF code).

### Write TF-native when

- The code is part of the **forward graph** that has to run inside Keras /
  `tf.function` / `GradientTape` (every layer in `tf_common.py`).
- The code maintains **`tf.Variable` state**
  (`ModelEMA` shadow vars, `SGDMomentum` velocity buffers, `AdamMomentum`
  m/v moments).
- The code has to be **autodiff-compatible**
  (`ComputeLossTF` — must produce a TF scalar that `GradientTape` can
  differentiate through the model variables).
- The code interfaces with `TFLiteConverter` / `tf.saved_model.save` /
  `tf.lite.Interpreter`.

### Wrapper / adapter when

- A PT util needs a tiny shim to accept TF inputs.
  Example: `utils/tf_autoanchor.py` builds a 5-line `_DetectStub` so that
  PT `check_anchors` (which expects `model.model[-1].anchors` as a torch
  tensor in cell units) can run on the YAML anchor list TF uses.
  **Rule**: if the wrapper is more than ~50 LoC, you are probably
  re-implementing — stop and check whether you can call the PT util via
  a different pathway.

## 4. Conventions

### Naming

- TF layers and modules: `TF<Name>` (e.g. `TFConv`, `TFC3`, `TFSPPF`,
  `TFConcat`, `TFUpsample`, `TFDetect`, `TFMixConv2d`).
- TF builders / facade classes: `<Name>TF` (e.g. `DetectionModelTF`).
  Suffix form when the name overlaps an existing PT one
  (PT has `DetectionModel`, TF has `DetectionModelTF`).
- TF utilities mirror the PT name with `tf_` prefix at file level
  (`tf_loss.py`, `tf_metrics.py`, `tf_autoanchor.py`, etc.) and keep the
  same symbol names inside (`ComputeLossTF`, not `TFComputeLoss`).
- Initialiser classes: `PT<Whatever>Init` to signal they replicate PT
  defaults (e.g. `PTConvKernelInit`, `PTConvBiasInit`).

### File-level conventions

- Every TF entry-point script (`train_tf.py`, `val_tf.py`, `detect_tf.py`,
  `export_tf.py`) reads/writes an `architecture.json` sidecar next to the
  weights. This stores `cfg`, `imgsz_hw`, `nc`, `act`, `strides`, `anchors`
  so that any of the other TF scripts can rebuild the exact graph from
  `<run>/best.weights.h5` without re-passing all flags.
- Output directories follow PT layout:
  - `last.weights.h5`, `best.weights.h5` (TF weights, not full Keras model)
  - `opt_state.npz` (resume state — velocity, EMA, epoch, best_fitness)
  - `training.log` (TSV, TF-side detailed)
  - `results.csv` (PT format — consumed by `utils.plots.plot_results`)
  - `results.png`, `confusion_matrix.png`, `PR_curve.png`, `P_curve.png`,
    `R_curve.png`, `F1_curve.png`, `labels_correlogram.jpg` (PT plotters)
  - `tb/` (TensorBoard event files)
  - `architecture.json` (sidecar)
- Save weights with `model.save_weights(path)` (HDF5), NOT `model.save()`
  with the full Keras model. Custom layers don't deserialize cleanly, and
  weights-only is portable across rebuilds at different imgsz.

### Hyperparameters and YAMLs

- The TF trainer accepts the **same `hyp.*.yaml` files** PT uses
  (`data/hyps/hyp.scratch-low.yaml`, `hyp.no-augmentation.yaml`, etc.).
  Do not introduce TF-specific hyp formats. If you need new fields, add
  them to a hyp file and document defaults that match PT behavior when
  unspecified.
- The TF parser in `models/tf_yolo.py:parse_model_tf` consumes the same
  YAML format (`backbone + head`, `[from, number, module, args]`) as
  `models/yolo.py:parse_model`. Module names supported today: `Conv`,
  `C3`, `SPPF`, `nn.Upsample`, `Concat`, `Detect`, `MixConv2d`. To add a
  new module, port its forward to a `TF<Name>` class in
  `models/tf_common.py` and wire it in `_resolve_module_args` +
  the `if m_str == "..."` chain in `parse_model_tf`.

### Edge TPU graph rules (must hold in every TF layer)

These are not stylistic preferences — violating them breaks the 100% TPU
mapping. See [experiment.md](experiment.md) "EdgeTPU-friendly graph design
rules" for full history.

1. **Upsample**: use `tf.image.resize(..., method="nearest")` (lowers to
   `RESIZE_NEAREST_NEIGHBOR`). NEVER `keras.layers.UpSampling2D` — on
   TF ≥ 2.19 it lowers to a 5D `EXPAND_DIMS` / `TILE` pattern that
   `edgetpu_compiler` v16 rejects.
2. **No in-graph reshape / flatten / decode**. Detect head emits raw
   per-scale maps `[B, H_i, W_i, na*(5+nc)]`; sigmoid + anchor decode +
   xywh + NMS happen on the host (`utils.tf_metrics.host_decode` →
   `utils.general.non_max_suppression`).
3. **Static shapes everywhere**. `keras.Input(shape=(H, W, 3), batch_size=1)`.
   No `tf.where`, `tf.boolean_mask`, `tf.top_k` over predictions inside
   the graph — they introduce dynamic shapes which block delegate search.
4. **PyTorch-parity padding for stride>1 convs**: explicit `TFPad` then
   `padding="valid"`, not Keras `padding="same"` (which is asymmetric for
   stride>1).
5. **BN momentum / epsilon**: `momentum=0.97, epsilon=1e-3`
   (≡ PT `nn.BatchNorm2d(momentum=0.03, eps=1e-3)`).
6. **Conv kernel + bias init**: `PTConvKernelInit` /
   `PTConvBiasInit` to replicate PT `kaiming_uniform_(a=sqrt(5))` —
   the Keras default `glorot_uniform` produces noticeably different
   initial output magnitudes that propagate as a sustained val loss
   delta.
7. **Detect head bias init**: call `TFDetect.initialize_biases(strides, img_w)`
   after model build; pre-biases obj logit toward "no object" and cls
   toward a small prior. Mirror of PT `DetectionModel._initialize_biases`.

## 5. Patterns

### Reusing a torch-input PT util on TF tensors

```python
# Common boundary: TF model output → host decode → PT NMS → AP computation.
preds_tf = model(images_nhwc, training=False)   # list of TF tensors
preds_np = [p.numpy() for p in preds_tf]
pred = host_decode(preds_np, anchors_pixel, strides, nc)   # torch.Tensor
out_per_img = non_max_suppression(pred, conf_thres, iou_thres)   # PT util
correct = process_batch(det_torch, labelsn_torch, iouv)            # PT util
```

The conversion happens in NumPy because:
- TF→NumPy is `tensor.numpy()`, NumPy→torch is `torch.from_numpy(arr).float()`.
- Crossing through fp32 NumPy avoids any `dtype`/device mismatch.
- The PT NMS/AP utils are unchanged — they work on torch tensors that
  came from NumPy as if they came from a PT model.

### Adding a new PT-mirrored TF utility

1. Create the file with `tf_` prefix in the same directory as the PT file.
2. Module-level imports: bring in PT counterparts you reuse, then TF.
3. Re-use over re-implement: if a PT function is pure NumPy or accepts
   numpy/torch boundary types, **call it directly** rather than rewriting
   in TF. Add a 5-15 line adapter shim if needed (see `tf_autoanchor.py`).
4. Document in module docstring **what is reused from PT** so future agents
   don't accidentally re-implement what's already a one-line delegation.

### Adding a new YAML module to the parser

1. Implement `TF<Module>` in `models/tf_common.py` (or `tf_experimental.py`
   if it's an experimental block). Make sure stride>1 uses `TFPad` and
   any upsample uses `tf.image.resize`.
2. In `models/tf_yolo.py:_resolve_module_args`, add the module name to
   the appropriate `if` arm (most simple modules join the Conv/C3/SPPF
   arm if they take `[c2, ...]` args).
3. In `models/tf_yolo.py:parse_model_tf`, add an `elif m_str == "..."`
   branch that constructs the layer and applies it.
4. Smoke test: `python export_tf.py --cfg <yaml> --imgsz 640 --out
   runs/tf_native/<name> --no-edgetpu` — confirm Keras model builds and
   TFLite quant runs without error.

### Adding a new persisted training feature (resume-aware)

If the feature has state that must survive `--resume`:
- Add the state to `_save_opt_state` in `train_tf.py`.
- Add the corresponding load logic in the `if args.resume:` block.
- The `opt_state.npz` is the single source of truth for per-run state.

## 6. Verification after changes

Always smoke-test with **1 epoch on iarna** (140 images, ~25s on CPU):

```
python train_tf.py \
    --cfg models/yolov5n_iarna_p2p4.yaml \
    --data data/iarna_abs.yaml \
    --hyp data/hyps/hyp.iarna-small.yaml \
    --imgsz 640 --epochs 1 --batch 8 \
    --out runs/tf_native/<smoke_name> --workers 2 --seed 0
```

Expected output (with current code at end of A/B/C session):
- ep 1 train total ≈ 1.25, val total ≈ 1.25
- All 7 PNGs generated in run dir
- `architecture.json`, `opt_state.npz`, `training.log`, `results.csv`,
  `tb/events.out.tfevents.*` present.

Then verify **EdgeTPU compile**:

```
python export_tf.py \
    --weights runs/tf_native/<smoke_name>/best.weights.h5 \
    --data data/iarna_abs.yaml \
    --out runs/tf_native/<smoke_name>_export --n-calib 30
```

Expected: `subgraphs=1 total=187 edgetpu=187 cpu=0` (P2P4 architecture
at 480×640 / 640×640 — keep it 100% TPU mapped after every layer change).

## 7. State of the migration (what is done, what remains)

For the up-to-date narrative including phase-by-phase deltas and
quantitative parity comparisons (loss curves, op histograms, mAP), see
[experiment.md](experiment.md). Summary at the file's end.

### Done across phases 1–3 + A/B/C + TensorBoard + AdamW + multi-format export

- TF-native NHWC graph build from any standard YOLOv5 YAML (Phase 0
  refactor — `models/tf_common.py` + `models/tf_yolo.py`).
- ComputeLossTF with PT-parity (CIoU + BCE-obj(target=IoU) + BCE-cls,
  per-scale balance, anchor matching with offset bias 0.5,
  `anchor_t=4`).
- Optimizer mechanics: 3 param groups, manual SGDMomentum and
  AdamMomentum (lr/momentum reassignable per iteration), warmup,
  linear/cosine LR schedule, ModelEMA with PT decay schedule.
- Auto-anchor (Phase 2) via PT `check_anchors` + `kmean_anchors`.
- mAP / P / R per epoch (Phase 1), best by fitness =
  `0.1·mAP50 + 0.9·mAP50-95`, early stopping on plateau.
- Resume / save-period / native `results.png` / labels distribution
  / PR-F1-CM curves at end (Phase 3 + C).
- INT8 + EdgeTPU export with real-image calibration; `--include` flag
  for SavedModel / fp16 TFLite / Keras / EdgeTPU.
- TFLiteModelWrapper for end-to-end mAP measurement on the quantized
  graph (uint8 *and* int8 input/output supported).
- `detect_tf.py` CLI for inference (image / folder / video) reusing
  PT `LoadImages` and `Annotator`.
- `models/tf_experimental.py`: `TFMixConv2d` (parser-aware), `TFEnsemble`,
  `attempt_load_tf`.
- TensorBoard scalar logging (`tb/` directory, per-step + per-epoch).

### Remaining (deferred, by impact)

| Area | Status |
|---|---|
| **Phase 5 — residual val_obj +15% gap** | sustained TF > PT delta on val obj loss; worth investigating only if 30+ epoch run shows TF mAP materially below PT |
| ~~Mixed precision (AMP) for GPU training~~ | ✅ implemented (`--amp` flag in train_tf.py; no-op on CPU, activates `mixed_float16` policy + 1024× loss scale on GPU) |
| ~~Multi-scale training~~ | ✅ implemented (`--multi-scale` flag in train_tf.py; rebuilds model with `dynamic_shape=True` so `TFUpsample` uses `tf.shape(x)` at runtime; per-batch random imgsz in `[0.5, 1.5]·imgsz`) |
| ~~detect_tf.py extensions~~ | ✅ `--save-crop` (per-class folders), `--classes` filter, `--agnostic-nms` |
| ~~--save-json (COCO predictions)~~ | ✅ in `tf_validate(save_json=True)` and `val_tf.py --save-json` |
| ~~val_tf.py --task speed~~ | ✅ batch-1 latency benchmark (mean / p50 / p95 / FPS) |
| ~~val_tf.py --task study~~ | ✅ imgsz sweep with mAP + ms/img + FPS table |
| Image weights | default off, niche |
| `--evolve` hyperparameter GA | research workflow, separate |
| `--save-json` (COCO predictions) | only needed for benchmark COCO official |
| `detect_tf.py`: `--save-crop`, `--classes`, `--agnostic-nms` | minor UX |
| `val_tf.py --task speed/study` | benchmark modes |
| Comet / ClearML hooks | TB covers single-dev case |

## 8. Anti-patterns (do not do)

- **Do not duplicate PT logic in TF** when a 5-line shim can call the PT
  util. Always check `utils.general`, `utils.metrics`, `utils.plots`,
  `utils.augmentations` first.
- **Do not use `keras.layers.UpSampling2D`**, `Reshape((n, c))`, or any
  dynamic-shape op in the export graph (see Edge TPU rules above).
- **Do not save the full Keras model** with custom layers — the
  re-deserialization fails. Always `save_weights` and rebuild from cfg.
- **Do not introduce a new hyp YAML format**. Reuse the PT hyp files.
- **Do not skip the smoke test** after layer/optimizer/loss changes.
  Even when only refactoring, verify ep1 loss values and EdgeTPU
  compile result match the previous baseline.
- **Do not fork the dataloader pipeline**. The TF trainer consumes
  `utils.dataloaders.create_dataloader` directly with a 4-line
  `pt_batch_to_tf` adapter — keep it that way so mosaic / mixup / HSV /
  scale / fliplr stay in lockstep with the PT path.
- **Do not change `models/tf.py`** — it is legacy PT-driven export
  used only by `export.py`. The new TF code lives in `tf_common.py`,
  `tf_yolo.py`, `tf_experimental.py`.
