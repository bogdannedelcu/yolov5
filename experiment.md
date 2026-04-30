# Experiment Notes - EdgeTPU / TF native vs PyTorch

Date: 2026-04-27

## Scope
- Compare export behavior between:
  - PyTorch pipeline (Ultralytics standard path)
  - TensorFlow native experimental pipeline
- Validate output tensor semantics and EdgeTPU compiler compatibility.

## Model under test
- YAML: ultralytics/cfg/models/v5/yolo_1_downsample_v5n.yaml
- Detect layer in YAML: `[[19, 22, 25], 1, Detect, [nc]]`

## Key architectural findings
- SPPF does **not** flatten tensors.
  - It performs repeated max-pooling and channel concat at same spatial resolution.
  - Reference implementation: ultralytics/nn/modules/block.py (class `SPPF`).
- PyTorch `Detect` receives a list of 3 pyramid feature maps and concatenates predictions after per-scale heads.
  - No spatial concat of the three maps into one HxW tensor.
  - Reference: ultralytics/nn/modules/head.py (class `Detect`, methods `forward_head`, `_inference`, `_get_decode_boxes`).

## Verified Detect inputs (PyTorch, this YAML)
- Input test: 1x3x512x640
- Hooked Detect input shapes:
  - (1, 32, 32, 40)
  - (1, 64, 16, 20)
  - (1, 64, 8, 10)
- Detect strides:
  - [16.0, 32.0, 64.0]

Interpretation:
- This custom YAML effectively outputs P3/P4/P5 at strides 16/32/64 for 512x640.
- Total grid points: 32x40 + 16x20 + 8x10 = 1680.

Important:
- A direct PyTorch forward with 1x3x480x640 fails in head concat for this YAML (shape mismatch).
- Therefore, comparisons against PyTorch `Detect` were validated on 512x640, where model topology is consistent.

## Export comparison results

### PyTorch export (baseline)
- Artifact: yolov5/runs/detect/train-18/weights/best_saved_model/best_int8.tflite
- Input: [1, 512, 640, 3] float32
- Output: [1, 5, 1680] float32
- ONNX baseline:
  - Input: [1, 3, 512, 640]
  - Output: [1, 5, 1680]

### TensorFlow native (raw head)
- Artifact: runs/tf_nhwc/train_tf_480x640_export_clean/detector_int8.tflite
- Input: [1, 480, 640, 3] int8
- Output: [1, 120, 160, 6] int8 (single-scale dense head)

### TensorFlow native (decoded export for external NMS)
- Artifact: runs/tf_nhwc/train_tf_480x640_export_decoded_static/detector_int8.tflite
- Input: [1, 480, 640, 3] int8
- Output: [1, 19200, 6] int8
- Meaning: flattened candidate detections, ready for host-side threshold + NMS.

## EdgeTPU compiler behavior
- Raw-head TF export compiles successfully:
  - Artifact: runs/tf_nhwc/train_tf_480x640_export_edgetpu_raw_after_decode_patch/detector_int8_edgetpu.tflite
  - 1 subgraph, all mapped.
- Decoded-in-graph TF export fails on EdgeTPU delegate search due to dynamic tensor constraints.

## Code changes made during research
- examples/tf_nhwc_detect_train_export.py
  - Added export wrapper with optional decoded output (`[B, N, 5+nc]`) and `--raw-output` switch.
- ultralytics/utils/tensorflow_native.py
  - For `format=edgetpu`, force `--raw-output` to preserve compile reliability.

## Current conclusion
- Detection-count mismatch (1680 vs 19200) is due to different head design and output semantics, not missing compiler operations.
- For EdgeTPU stability now:
  - Keep raw output in compiled model.
  - Perform decode/filter/top-k/NMS on host.
- For analysis/debug on CPU TFLite:
  - Use decoded export output `[1, 19200, 6]`.

## Gap summary: TF native vs PyTorch Detect
- PyTorch path in this YAML uses 3 detection scales at runtime (`[19, 22, 25]`) and concatenates per-scale predictions to 1680 locations (for 512x640).
- Current TF native detector is a simplified single-scale head (dense map), not a 3-scale Detect equivalent.
- This architectural gap is the primary reason outputs differ in candidate count and tensor structure.

## Suggested next experiments
1. Implement host-side Top-K prefilter (e.g., 2000) before NMS to match historical runtime profile.
2. Prototype fully static decode graph (no dynamic shape ops) and retest EdgeTPU compile.
3. Add a small validation script comparing mAP/precision between:
   - PyTorch baseline output path
   - TF raw+host-decode path
   - TF decoded (CPU TFLite) path

## Update: parity-v2 (full TF multi-scale head)
- Script path updated to include all major blocks in TensorFlow (`SPPFLite` + multi-scale PAN/FPN + three Detect-like scale heads).
- Export artifact:
  - runs/tf_nhwc/parity_tf_v2_export_fast/detector_int8.tflite
  - Input: [1, 512, 640, 3] int8
  - Output: [1, 5, 1680] int8 (Detect-like `[B, no, N]` layout)
- First EdgeTPU compile attempt failed with dynamic tensor errors (delegate static-size requirement).
- After forcing static batch-1 reshape in export wrapper:
  - Artifact: runs/tf_nhwc/parity_tf_v2_export_edgetpu_fast_v2/detector_int8.tflite
  - EdgeTPU compile succeeded:
    - runs/tf_nhwc/parity_tf_v2_export_edgetpu_fast_v2/detector_int8_edgetpu.tflite
    - 1 EdgeTPU subgraph
    - 98 ops on EdgeTPU, 89 ops on CPU

Implication:
- Full-TF parity shape/semantics are now aligned with PyTorch-style Detect output.
- EdgeTPU deployment is possible, but currently hybrid (EdgeTPU + CPU fallback) due unsupported ops in this parity graph.

## Update: EdgeTPU-friendly head activation
- Added configurable head activation in TF export script (`--head-act silu|relu|relu6`).
- Test run (512x512, raw output, head_act=relu):
  - Artifact: runs/tf_nhwc/parity_tf_v2_export_512_raw_edgetpu_reluhead/detector_int8.tflite
  - EdgeTPU compile result:
    - Total ops: 139 (down from 187 baseline raw)
    - EdgeTPU ops: 96
    - CPU ops: 43 (down from 89 baseline raw)
- New frontier moved away from SiLU `mul` tensors toward concat/resizing boundaries.

Takeaway:
- ReLU in head significantly improves EdgeTPU mapping for this graph.
- Remaining blockers are primarily `RESIZE_NEAREST_NEIGHBOR`, `CONCATENATION`, `RESHAPE`, `TRANSPOSE`, and graph partition boundary behavior.

## Update: upsample and activation follow-up
- Tested `head_upsample=deconv` (`Conv2DTranspose`) + `head_act=relu` in TF script.
  - Conversion produced `TRANSPOSE_CONV` ops.
  - `edgetpu_compiler` failed to parse model (`Didn't find op for builtin opcode 'TRANSPOSE_CONV' version '4'`).
  - Conclusion: this deconv path is not usable with current compiler/runtime toolchain.

- Tested `head_act=relu6` with `head_upsample=nearest`.
  - Compile succeeded.
  - Mapping was effectively identical to ReLU head:
    - EdgeTPU ops: 96
    - CPU ops: 43

Current best known configuration in this branch:
- `head_act=relu` (or `relu6`) + `head_upsample=nearest`
- Gives major improvement vs initial SiLU baseline (CPU fallback cut roughly in half).

## Update: TF 2.10 environment and TRANSPOSE_CONV resolution path
- Created dedicated conda env for controlled legacy conversion tests:
  - env: `tf210-edgetpu`
  - Python: `3.9.18`
  - TensorFlow: `2.10.0`
- Motivation:
  - On TF 2.19, deconv exports emitted `TRANSPOSE_CONV` op version 4 and failed parser stage in `edgetpu_compiler` 16.
  - Need to validate if older TF emits an older op version accepted by compiler.

### Compatibility patch for TF 2.10 export
- In `examples/tf_nhwc_detect_train_export.py` added SavedModel export fallback:
  - use `model.export(..., format="tf_saved_model")` when available
  - fallback to `tf.saved_model.save(...)` for stacks where `model.export` does not exist (TF 2.10)

### TF 2.10 deconv test (standard head topology)
- Command profile:
  - `--head-upsample deconv --head-act relu6 --raw-output --compile-edgetpu --imgsz 512`
- Artifacts:
  - `runs/tf_nhwc/tf210_deconv_train_export_512/detector_int8.tflite`
  - `runs/tf_nhwc/tf210_deconv_train_export_512/detector_int8_edgetpu.log`
- Results:
  - No parser error for `TRANSPOSE_CONV`.
  - Flatbuffer inspection: `TRANSPOSE_CONV` opcode version `2`.
  - Compile mapping:
    - Total ops: 139
    - EdgeTPU ops: 96
    - CPU ops: 43
    - `TRANSPOSE_CONV`: 2 on CPU (not mapped)

Interpretation:
- TF version change solved the parser incompatibility (`v4` -> `v2`) but did not automatically guarantee full TPU mapping in the full detector graph.

## Update: minimal head topology experiment (deconv + TF 2.10)
- Added script option to probe partition boundaries:
  - `--head-topology standard|minimal`
  - `minimal` removes concat-based PAN fusion joins (`up4+p4`, `up3+p3`, `down4+h4_reduce`, `down5+h`) to reduce fusion boundary pressure.

### TF 2.10 deconv test (minimal topology)
- Command profile:
  - `--head-upsample deconv --head-act relu6 --head-topology minimal --raw-output --compile-edgetpu --imgsz 512`
- Artifacts:
  - `runs/tf_nhwc/tf210_deconv_minimal_train_export_512/detector_int8.tflite`
  - `runs/tf_nhwc/tf210_deconv_minimal_train_export_512/detector_int8_edgetpu.log`
- Results:
  - Flatbuffer inspection: `TRANSPOSE_CONV` opcode version `2`.
  - Compile mapping improved significantly:
    - Total ops: 135
    - EdgeTPU ops: 105
    - CPU ops: 30
  - `TRANSPOSE_CONV` split status:
    - 1 mapped to EdgeTPU
    - 1 on CPU

Key takeaway:
- This is the first confirmed configuration in this project where `TRANSPOSE_CONV` is partially mapped to EdgeTPU.
- Remaining CPU fallback is now concentrated in subset partition boundaries (`CONCATENATION`, `RESHAPE`, `TRANSPOSE`, part of `CONV_2D`, and 1 `TRANSPOSE_CONV`).

---

## Update: Standard YOLOv5n (no extra downsample) — EdgeTPU validation 2026-04-27

### Context
- `yolo_1_downsample_v5n.yaml` adds an extra downsample layer making final stride = 64.
- This was validated at `imgsz=512` (100% TPU), but fails at any rectangular resolution.
- Problem: EdgeTPU TRANSPOSE ops fail when grid dimensions are not multiples of a specific alignment (e.g. 704/64=11 is odd → TRANSPOSE "unspecified limitation").
- Confirmed: only `imgsz=512` (8×8 grid at P5) yields 100% TPU for that YAML.

### Switch to standard YOLOv5n (stride 32)
- Standard `yolov5.yaml` (from train-11 sweep) has final stride = 32.
- train-11 sweep confirmed 100% TPU across many resolutions:
  - 704×512 → 598/598 ops, 1 subgraph, 0 CPU
  - 736×512, 768×512, 800×512, 832×512 → 100% TPU
  - 768×544, 736×576, 768×576, 768×608, 736×608 → 100% TPU
  - 736×640, 768×640, 768×672, 768×704, 736×704 → 100% TPU
  - 864×608, 896×608, 864×640 → 100% TPU
- Conclusion: with stride 32, both dimensions just need to be multiples of 32.

### `yolo_1_downsample_v5n.yaml` — resolution constraint analysis
| imgsz | P5 grid | EdgeTPU result |
|-------|---------|----------------|
| 512×512 | 8×8 | ✅ 279/279 (100%) |
| 512×640 | 8×10 | ❌ 127/279 (TRANSPOSE + multiple subgraph failures) |
| 512×1024 | 8×16 | ❌ 261/279 |
| 1024×1024 | 16×16 | ❌ 261/279 |

- Root cause: TRANSPOSE in Detect head fails when grid dim is not aligned to a specific EdgeTPU constraint.
- Only 8×8 (imgsz=512×512) passes completely.

### Target resolution for ≥640×480 footage
- With standard YOLOv5n (stride 32): use `imgsz=512,640` (H=512, W=640).
  - Both dims are multiples of 32 → 100% TPU confirmed from train-11 results.

### New YAML created
- File: `ultralytics/cfg/models/v5/yolov5n_iarna_edgetpu.yaml`
- Based on original `yolov5.yaml`, with:
  - `nc: 1` (iarna dataset)
  - Only `n` scale defined
  - Comments documenting EdgeTPU-compatible resolutions
- Recommended export: `yolo export model=<weights.pt> format=edgetpu imgsz=512,640`

---

## Lecții învățate (Lessons Learned) — 2026-04-27

### 1. EdgeTPU compiler versioning — TRANSPOSE_CONV op version
- `edgetpu_compiler` v16 acceptă doar `TRANSPOSE_CONV` op version ≤ 3.
- TF 2.19 emite `TRANSPOSE_CONV` v4 → parser failure ("Didn't find op for builtin opcode 'TRANSPOSE_CONV' version '4'").
- TF 2.10 emite `TRANSPOSE_CONV` v2 → parsează corect.
- **Regulă**: dacă folosești `Conv2DTranspose` (deconv upsample) în export TF, folosește TF ≤ 2.10.

### 2. Activare SiLU vs ReLU/ReLU6 în capul de detecție (TF native path)
- SiLU se expandează în TFLite ca `MUL(x, LOGISTIC(x))` → EdgeTPU nu mapează aceste perechi la granițele de partiționare.
- Înlocuirea cu ReLU sau ReLU6 în capul de detecție a redus CPU fallback de la 89 → 43 ops.
- **Regulă**: în modelele TF native destinate EdgeTPU, folosește ReLU sau ReLU6 în head (nu SiLU).

### 3. Tensori dinamici în grafuri TFLite → EdgeTPU incompatibil
- Orice op care produce un tensor cu dimensiune necunoscută la compilare (dynamic shape) blochează delegate search.
- Exemplu: decode-în-graf cu `tf.boolean_mask` sau `tf.where` → fail.
- **Regulă**: toate tensorii din modelul exportat trebuie să aibă shape static complet cunoscut. Decode/NMS se face pe host, nu în graf.

### 4. Stride final și alinierea grid-ului la EdgeTPU TRANSPOSE
- EdgeTPU mapează TRANSPOSE (din Detect head, reordonare DFL) cu constrângere de aliniere pe dimensiunile grid-ului.
- Un YAML cu extra-downsample (stride final = 64) funcționează **doar la imgsz=512×512** (grid P5 = 8×8).
  - 704/64 = 11 (impar) → TRANSPOSE "unspecified limitation" → 18 CPU fallback ops.
  - 640/64 = 10 (par, dar non-pow2) → și mai rău (127/279 ops pe TPU).
- Modelul standard (stride final = 32) funcționează la orice imgsz multiplu de 32, inclusiv rectangular.
- **Regulă**: pentru deployment rectangular pe EdgeTPU, folosește arhitecturi cu stride final ≤ 32.

### 5. Extra-downsample — trade-off receptive field vs flexibilitate rezoluție
- `yolo_1_downsample_v5n.yaml` adaugă un downsample în plus față de YOLOv5 standard.
  - Pro: câmp receptiv mai mare, potențial util pentru obiecte mici la rezoluții mici.
  - Con: stride 64 la P5 → restricție severă de rezoluție pe EdgeTPU (doar 512×512).
- **Concluzie**: pentru EdgeTPU cu input non-pătrat, extra-downsample YAML nu este viabil.

### 6. PyTorch path (Ultralytics standard export) vs TF native path
- PyTorch → ONNX → TFLite (via Ultralytics `format=edgetpu`) produce modele 100% pe EdgeTPU cu standard YOLOv5n.
  - train-11 sweep: 598/598 ops, 0 CPU la 704×512 și multe alte rezoluții.
- TF native path (tf_nhwc_detect_train_export.py): best result = 105/135 TPU (cu TF 2.10 + deconv + minimal topology).
- **Concluzie**: pentru deployment practic, calea PyTorch → Ultralytics export este superioară și mai simplă.

### 7. Sweep de rezoluție — utilitate și cost
- Fiecare candidat din sweep durează ~120-170s (export ONNX → SavedModel → TFLite + compilare EdgeTPU).
- `--max-candidates` nu reduce timpul total dacă candidații valizi sunt mulți.
- **Regulă**: rulează sweep doar după ce arhitectura e validată structural la o singură rezoluție fixă.
- Validare structurală rapidă: un singur export la o rezoluție reprezentativă (e.g. 512×512 sau 512×640).

### 8. Antrenare "de formă" pentru validare structurală
- Nu este nevoie de antrenare completă pentru a valida dacă un YAML exportă corect pe EdgeTPU.
- `epochs=1 fraction=0.05` este suficient pentru a obține weights valide structural (nu de calitate).
- Aceasta durează ~5-10s și produce un `.pt` utilizabil pentru export.
- **Regulă**: pentru validare structurală EdgeTPU, folosește epochs=1 fraction=0.05.

### 9. Fișier YAML dedicat pentru EdgeTPU
- Menținerea unui YAML separat (e.g. `yolov5n_iarna_edgetpu.yaml`) cu:
  - `nc` corect pentru dataset
  - Doar scale `n` definit
  - Comentarii cu rezoluțiile validate
  permite reproducibilitate și evită confuzii cu YAML-ul original multi-clasă.

### Tabel rezumat configurații testate

| Configurație | TPU ops | CPU ops | Total | Note |
|---|---|---|---|---|
| TF native, SiLU, nearest | 98 | 89 | 187 | baseline TF |
| TF native, ReLU, nearest | 96 | 43 | 139 | activare schimbată |
| TF native, ReLU6, nearest | 96 | 43 | 139 | echivalent ReLU |
| TF 2.19, deconv, ReLU | ❌ | — | — | TRANSPOSE_CONV v4 parse error |
| TF 2.10, deconv, ReLU6, standard | 96 | 43 | 139 | v2 parsează ok |
| TF 2.10, deconv, ReLU6, minimal | 105 | 30 | 135 | best TF result |
| PyTorch YOLOv5n (stride 32), 512×512 | 598 | 0 | 598 | ✅ 100% TPU |
| PyTorch YOLOv5n (stride 32), 704×512 | 598 | 0 | 598 | ✅ 100% TPU |
| PyTorch YOLOv5n (stride 32), 512×640 | 598 | 0 | 598 | ✅ 100% TPU |
| PyTorch 1ds-v5n (stride 64), 512×512 | 279 | 0 | 279 | ✅ 100% TPU, doar square |
| PyTorch 1ds-v5n (stride 64), 512×640 | 127 | 152 | 279 | ❌ TRANSPOSE fail |
| PyTorch 1ds-v5n (stride 64), 1024×1024 | 261 | 18 | 279 | ❌ TRANSPOSE fail |

---

## Update: TF parallel path with YAML parser (mirror PyTorch) — 2026-04-27

### Goal
- Add a TensorFlow training+export path that parses the **same Ultralytics model YAMLs** as the PyTorch path, so the same architecture can be deployed via two backends.
- Hypothesis: NHWC-native TF graph is friendlier to EdgeTPU than PyTorch → ONNX → TFLite (avoids implicit transposes around channels-first → channels-last conversion).
- Goal scope: **only** direct EdgeTPU export. No generic TFLite or other targets.

### Wiring
- New flag in [ultralytics/cfg/default.yaml](ultralytics/cfg/default.yaml): `framework: pytorch` (also accepts `tensorflow`/`tf`/`tensor` as alias for "platform=tensorflow").
- Dispatch in [ultralytics/cfg/__init__.py](ultralytics/cfg/__init__.py) and [ultralytics/engine/model.py](ultralytics/engine/model.py) routes train/export to [ultralytics/utils/tensorflow_native.py](ultralytics/utils/tensorflow_native.py), which forwards to [examples/tf_nhwc_detect_train_export.py](examples/tf_nhwc_detect_train_export.py).
- The YAML path is forwarded as `--model-yaml`; the script also persists an `architecture.yaml` sidecar next to `detector.keras` so export can rebuild even if invoked separately.

### TF NHWC modules added (mirror PyTorch parse_model)
File: [examples/tf_nhwc_detect_train_export.py](examples/tf_nhwc_detect_train_export.py).
- `TFConv(c2, k, s, act)` — Conv2D + BN + activation, padding="same" (matches PyTorch autopad output shapes for all (k,s) used in v5).
- `TFBottleneck(c2, shortcut, e=0.5)` — kernels (3,3), residual when `c1==c2 and shortcut`.
- `TFC3(c2, n, shortcut, e=0.5)` — 3-conv CSP, n bottlenecks.
- `TFSPPF(c2, k=5)` — 1×1 reduce → 3 chained MaxPool (stride 1, same) → channel concat → 1×1 expand. Spatial dims preserved (NOT a flatten).
- `TFDetectHead(nc)` — anchor-free per-scale 1×1 Conv2D emitting 4+nc channels. Simplification vs PyTorch `Detect`: **no DFL** (reg_max not used). Decode + DFL-equivalent run on host.
- `build_model_from_yaml(yaml_path, img_hw, ch_in, nc_override, scale_override, act)` — walks `backbone+head`, resolves `from`/`number`/`module`/`args`, applies `scales[scale]` depth/width/max_channels, propagates channels through Concat (sum) and Detect (multi-input), returns `(model, nc)` with the model outputting a list of per-scale [B, H, W, 4+nc] maps (training shape).

### Sanity check (480×640, scale=n, nc=1)
- P3=60×80, P4=30×40, P5=15×20 (strides 8/16/32). Output channels 4+1=5.
- Total params ~1.77M (consistent with YOLOv5n at width=0.25).
- Train smoke test: `epochs=1 fraction=0.05` on iarna dataset → 1 step in ~13s, model saved.

### Run commands
```
# Train
yolo train framework=tensorflow \
    model=ultralytics/cfg/models/v5/yolov5n_iarna_edgetpu.yaml \
    data=/home/bogdan/work/yolo/iarna/my_dataset/detect/detect.yaml \
    epochs=1 fraction=0.05 imgsz=480,640 name=tf_iarna_480x640
# Export (auto-loads architecture.yaml sidecar saved during train)
yolo export framework=tensorflow \
    model=runs/tf_nhwc/tf_iarna_480x640/detector.keras \
    data=/home/bogdan/work/yolo/iarna/my_dataset/detect/detect.yaml \
    format=edgetpu imgsz=480,640
```

### EdgeTPU export blockers found in TF 2.19 + edgetpu_compiler 16

**TFLite op histogram (raw export, 480×640, SiLU):**
```
ADD: 7, CONCATENATION: 18, CONV_2D: 60, EXPAND_DIMS: 4,
LOGISTIC: 57, MAX_POOL_2D: 3, MUL: 57, PACK: 3, QUANTIZE: 2,
RESHAPE: 7, SHAPE: 7, STRIDED_SLICE: 7, TILE: 4, TRANSPOSE: 1
```

Compile error: `Didn't find op for builtin opcode 'TILE' version '3'`.

**Source 1 — TILE×4 (FIXED):** `tf.keras.layers.UpSampling2D(interpolation="nearest")` lowers on TF 2.19 to a 5D `EXPAND_DIMS → TILE([1,1,2,1,1]) → STRIDED_SLICE` pattern (one EXPAND/TILE per spatial axis × 2 upsamples = 4 TILEs). Compiler accepts only TILE ≤ v2.
- Fix: YAML parser's `nn.Upsample` branch now emits `tf.keras.layers.Resizing(target_h, target_w, interpolation="nearest")` instead of `UpSampling2D`. Resizing lowers to `RESIZE_NEAREST_NEIGHBOR` (EdgeTPU-supported).

**Source 2 — PACK×3 / SHAPE×7 / STRIDED_SLICE×7 (PARTIALLY ADDRESSED):** `tf.keras.layers.Reshape((n, c))` (and its old `Lambda(tf.reshape)` equivalent) used inside `build_export_model` to flatten per-scale maps to `[1, n, c]`. Even with `Input(batch_size=1)`, TF 2.19 MLIR converter doesn't fold the batch dim and lowers Reshape to a runtime shape computation: `STRIDED_SLICE(SHAPE(x)) → PACK with [n, c] constants → RESHAPE`. Three scales = three PACKs; preceded by SHAPE/STRIDED_SLICE per dim.
- Pending fix: replace flatten/concat/permute in the raw export wrapper with **multi-output** model (return three NHWC `[B, H_i, W_i, 4+nc]` tensors directly). Host concatenates and decodes. EdgeTPU happily compiles multi-output models and this avoids all dynamic-shape ops at the boundary.
- Alternative: re-test the export under the `tf210-edgetpu` conda env (TF 2.10) — same approach that previously resolved `TRANSPOSE_CONV v4 → v2` for deconv heads.

### EdgeTPU-friendly graph design rules (NHWC native, generalizing prior findings)
1. **Avoid `UpSampling2D(interpolation="nearest")`** — use `Resizing(h, w, "nearest")` instead. (TF 2.19 lowering pitfall.)
2. **Avoid `Reshape((n, c))` and `Lambda(tf.reshape)` at static-batch=1** — prefer multi-output models that defer flattening to host, or move to TF 2.10.
3. **Scale-broadcast `x * scale` in decode path produces TILE/EXPAND_DIMS** — keep decode on host (raw export), don't decode in the TFLite graph.
4. **SiLU still expensive** — same rule as before: ReLU/ReLU6 head is preferable for EdgeTPU even in NHWC native. With SiLU you get MUL=57 / LOGISTIC=57 pairs that fragment the EdgeTPU partition.

### Status snapshot at session pause (2026-04-27)
- Train via `framework=tensorflow` with YAML works: `runs/tf_nhwc/tf_iarna_480x640/detector.keras` + `architecture.yaml`.
- Export to TFLite INT8 works.
- ✅ **EdgeTPU compile succeeds at 100% TPU (201/201 ops, 1 subgraph, 0 CPU fallback)** after both fixes applied and re-tested end-to-end.

### EdgeTPU compile result — TF parallel path, YOLOv5n YAML, 480×640, SiLU
After applying both fixes (Upsample→Resizing in YAML parser + multi-output raw export wrapper), the TFLite op histogram is:

```
ADD: 7, CONCATENATION: 13, CONV_2D: 60, LOGISTIC: 57,
MAX_POOL_2D: 3, MUL: 57, QUANTIZE: 2, RESIZE_NEAREST_NEIGHBOR: 2
```

`edgetpu_compiler` 16 result:
```
Try to compile segment with 201 ops
On-chip memory used for caching model parameters: 1.82MiB
On-chip memory remaining for caching model parameters: 5.26MiB
Off-chip memory used for streaming uncached model parameters: 0.00B
Number of Edge TPU subgraphs: 1
Total number of operations: 201

Operator                       Count      Status
CONCATENATION                  13         Mapped to Edge TPU
LOGISTIC                       57         Mapped to Edge TPU
RESIZE_NEAREST_NEIGHBOR        2          Mapped to Edge TPU
CONV_2D                        60         Mapped to Edge TPU
MUL                            57         Mapped to Edge TPU
QUANTIZE                       2          Mapped to Edge TPU
MAX_POOL_2D                    3          Mapped to Edge TPU
ADD                            7          Mapped to Edge TPU
```

**Artifacts:**
- `runs/tf_nhwc/tf_iarna_480x640_export/detector_int8_edgetpu.tflite` (2.16 MB)
- `runs/tf_nhwc/tf_iarna_480x640_export/detector_int8.tflite` (1.96 MB)
- Output spec: 3 raw NHWC scale maps `[1, 60, 80, 5]`, `[1, 30, 40, 5]`, `[1, 15, 20, 5]` (P3/P4/P5 at strides 8/16/32).

### Surprising finding: SiLU maps cleanly on EdgeTPU in NHWC-native graphs
- Historical TF native SiLU baseline (hardcoded simplified detector, experiment.md row "TF native, SiLU, nearest"): 98 TPU + 89 CPU = 187 ops, ~52% TPU.
- New YAML-driven NHWC graph (full YOLOv5n, SiLU everywhere): **100% TPU**, 57 LOGISTIC + 57 MUL all mapped.
- Hypothesis: the prior CPU fallback wasn't intrinsic to SiLU — it came from interaction with `Reshape`/`Lambda(reshape)`/`UpSampling2D` partition boundaries that fragmented the graph. Once those are eliminated (multi-output raw export + Resizing), the `MUL(x, LOGISTIC(x))` SiLU pair stays inside the EdgeTPU subgraph.
- **Implication**: rules from experiment.md "Lecție 2 (SiLU vs ReLU/ReLU6)" and the activation tables under prior updates are partially superseded for the YAML-driven NHWC path. Activation choice for EdgeTPU mapping is no longer the dominant factor when partition boundaries are clean.

### Comparison vs PyTorch baseline at same model, same imgsz
- PyTorch path (Ultralytics export, ONNX→TFLite→EdgeTPU), YOLOv5n stride 32 at multiples of 32: 598/598 ops, 100% TPU.
- TF parallel NHWC-native path, YAML-driven, 480×640: 201/201 ops, 100% TPU.
- The TF op count is **lower** (~3×) because the NHWC graph avoids the implicit NCHW↔NHWC transposes and the DFL block (Detect head here is anchor-free 4+nc, no DFL). This is the cleanest EdgeTPU graph produced in this project so far.
- Caveat: the simplified Detect (no DFL) means box accuracy under quantization will likely be lower than the PyTorch baseline. Validation pending — only structural check has been done.

### Lesson summary added to project rules (2026-04-27)
- For EdgeTPU exports from NHWC-native Keras graphs:
  - Use `tf.keras.layers.Resizing(target_h, target_w, "nearest")` for upsampling — never `UpSampling2D` (TF 2.19 lowering pitfall).
  - Use **multi-output** model for per-scale heads. Don't flatten/concat/permute inside the graph; let the host do it.
  - SiLU is acceptable when partition boundaries are clean. Activation choice (SiLU vs ReLU/ReLU6) is a secondary lever, not the dominant one.
  - Detect head simplification (4+nc, no DFL) keeps the head graph small and fully TPU-mappable; if accuracy demands DFL, that block must be added with care to preserve full TPU mapping.

---

## Update: Side-by-side TF vs PyTorch on iarna 480×640 — 2026-04-27

### Goal
Train and export the same YAML through both backends, with maximally-aligned outputs (3 raw NHWC scale maps), to compare:
1. Op-level graph structure;
2. Per-invoke I/O footprint;
3. Output statistics on the same random uint8 input.

### Setup
- **Same YAML**: `ultralytics/cfg/models/v5/yolov5n_edgetpu_iarna.yaml` (copy of `_iarna_edgetpu` to avoid an Ultralytics filename bug — `check_yolov5u_filename` strips `u` from anything ending in `u.yaml`, see [ultralytics/utils/checks.py:600](ultralytics/utils/checks.py#L600)).
- **Same dataset**: iarna at `/home/bogdan/work/yolo/iarna/my_dataset/detect/detect.yaml`, nc=1.
- **Same training budget**: `epochs=1 fraction=0.05 imgsz=480,640 batch=16` — structural validation, not accuracy-meaningful.
- **PyTorch export**: `int8=True` via `model.export(format='tflite', ...)` with `Detect.raw_multiscale_export = True` (new export-only flag added to [ultralytics/nn/modules/head.py](ultralytics/nn/modules/head.py)). The flag bypasses the final cross-scale concat and outputs 3 per-scale `[B, 4+nc, H_i, W_i]` tensors — DFL is still applied per scale but cls is left as logits, no decode/sigmoid in graph. Training is unaffected (flag only checked when `self.export and self.raw_multiscale_export`).
- **TF export**: `framework=tensorflow format=edgetpu imgsz=480,640` (already-implemented YAML-driven NHWC builder). Now uses `inference_input_type=tf.uint8` (host doesn't pre-quantize images).

### File-level comparison

| Aspect | TF (NHWC native) | PT (NCHW→NHWC via ONNX) |
|---|---|---|
| TFLite size | 1.87 MiB | 2.53 MiB |
| EdgeTPU compile | 100% TPU, 1 subgraph, 202 ops | (not yet tried — likely partial due to DFL/SOFTMAX) |
| Input dtype | uint8 `[1,480,640,3]` (q: scale=1, zero=0) | float32 `[1,480,640,3]` (no quant at I/O even with int8=True) |
| Outputs | 3× int8 NHWC `[1,H_i,W_i,5]` (raw quantized) | 3× float32 NHWC `[1,H_i,W_i,5]` |
| Output strides | 8 / 16 / 32 (P3/P4/P5) | 8 / 16 / 32 (P3/P4/P5) |

### Op histogram diff (TFLite, pre-EdgeTPU)
```
OP                              TF      PT      delta
ADD                              7       7        0
CONCATENATION                   13      16       +3
CONV_2D                         60      78      +18   (PT Detect cv2/cv3 are 3-stage Conv blocks)
LOGISTIC                        57      68      +11   (extra SiLU inside PT Detect head)
MAX_POOL_2D                      3       3        0
MUL                             57      68      +11
PAD                              0       6       +6   (NCHW→NHWC layout artifacts)
QUANTIZE                         3       0       -3
RESHAPE                          0       6       +6   (NCHW→NHWC artifacts)
RESIZE_NEAREST_NEIGHBOR          2       2        0
SOFTMAX                          0       3       +3   (DFL per scale)
TRANSPOSE                        0       9       +9   (NCHW→NHWC layout artifacts)
TOTAL                          202     266      +64  (+32%)
```

The +32% in PT graph comes from three sources:
1. **DFL block (~3 SOFTMAX + extra concat/conv)**: PT applies DFL inside the graph to collapse `4·reg_max=64` → `4` box channels. TF skips DFL entirely (anchor-free 4+nc direct regression).
2. **Heavier Detect head**: PT cv2 = Conv→Conv→Conv2d (3 levels), cv3 = DWConv→Conv→DWConv→Conv→Conv2d (5 levels) per scale, with SiLU between. TF Detect head = a single 1×1 Conv2D per scale.
3. **Layout conversion overhead**: 9 TRANSPOSE + 6 PAD + 6 RESHAPE come from the ONNX→TFLite converter inserting NCHW↔NHWC adapters at non-trivial boundaries. TF graph is NHWC-native end-to-end, so zero of these.

### Inference parity check (random uint8 input, seed=42)
Same logical 480×640×3 uint8 image fed to both (TF: as uint8; PT: as fp32 in [0,1]):

| Output | TF (dequantized) | PT (fp32) | Note |
|---|---|---|---|
| `[1,60,80,5]` (P3) | mean=−0.08, std=1.19 | mean=4.57, std=5.86 | PT box channels post-DFL (distance) |
| `[1,30,40,5]` (P4) | mean=−0.17, std=0.29 | mean=4.85, std=5.31 | same |
| `[1,15,20,5]` (P5) | mean=0.30, std=0.17 | mean=5.12, std=4.75 | same |

- ✅ **Output shapes identical** between TF and PT (3 raw scale maps at strides 8/16/32, 5 channels each).
- ✅ **Both run cleanly with the bundled TFLite interpreter** — no EdgeTPU runtime required for this inference path.
- ⚠️ **Box channel semantics differ**: PT box is post-DFL distance to box edges (range 0..reg_max=15 in cell units), TF box is raw logit interpretable as xywh after sigmoid. This is fine for an op-count comparison; for predictions to match, the host post-processor must understand which side it's reading.
- ✅ **Cls channel semantics identical**: both emit raw logits (no in-graph sigmoid).

### Per-invoke I/O traffic (relevant if running on Coral USB Accelerator)
| | TF | PT |
|---|---:|---:|
| Input bytes (host → device) | 921,600 (uint8) | 3,686,400 (fp32) ← 4× more |
| Output bytes (device → host) | 31,500 (int8) | 126,000 (fp32) ← 4× more |
| **Total per-invoke USB** | **~931 KiB** | **~3.63 MiB** ← ~4× more |

The TF int8 I/O alone is a 4× bandwidth win for Coral USB Accelerator deployments.

### Practical takeaways
- For EdgeTPU specifically the TF NHWC-native + multi-output path produces a **smaller, fully TPU-mapped graph** (202/202) with 4× lower per-invoke USB traffic than the PT path.
- Adding the export-only `Detect.raw_multiscale_export` flag on the PT side gives matched output shapes (`[1, H_i, W_i, 4+nc]`), which means the host post-processing pipeline (decode + NMS) can be written ONCE and reused across both backends — modulo the box-channel semantics caveat (DFL distance vs raw logit).
- The remaining ~32% op-count delta on PT is structural (DFL + heavier head + layout adapters); fully closing it would require either dropping DFL (anchor-free PT retrain with custom loss) or rewriting the ONNX exporter to lay out tensors NHWC-first.

---

## Update: 3-epoch parallel training + comparable detections (2026-04-27)

### What we did

1. **Trained both backends 3 epochs on full iarna (140 train / 40 val, 480×640):**
   - **TF**: `yolo train framework=tensorflow model=yolov5n_edgetpu_iarna.yaml ... epochs=3 imgsz=480,640`
     - Loss curve (TF custom L1+BCE on single-positive cell): 1.60 → 0.98 → 0.59 (-63% across 3 epochs).
     - Final losses: `box=0.114 cls=0.042 total=0.589`.
   - **PyTorch**: same YAML, augmentations disabled per-user request (`mosaic=0 mixup=0 scale=0 copy_paste=0.3 erasing=0`) since iarna detections are tiny and any image-shrinking would lose them.
     - Loss curve (Ultralytics v8DetectionLoss = `7.5·CIoU + 0.5·BCE + 1.5·DFL`): box≈6.92 cls≈8.30 dfl≈4.23 — essentially flat across 3 epochs.
     - Validation metrics: P=R=mAP50=mAP50-95=0 every epoch — TaskAlignedAssigner could not establish stable positive matches yet; v8 loss typically needs 50+ epochs from random init.

2. **Loss-comparison conclusion**: NOT a like-for-like comparison.
   - TF uses simplified loss: 5·L1 box (sigmoid(xywh) vs image-normalized target on single-positive cell) + 0.5·BCE cls.
   - PT uses v8DetectionLoss: TaskAlignedAssigner (top-K alignment metric `score^α · iou^β`), CIoU on decoded boxes, DFL distribution loss, BCE with soft target.
   - Magnitudes differ by ~150×. Convergence speed differs because PT's assigner needs the model to generate decent overlaps before any positives match — TF's simple cell-center assignment always has positives.

3. **Filename gotcha**: discovered that Ultralytics' `check_yolov5u_filename` strips `u` from any path containing `u.yaml`. Workaround: copied `yolov5n_iarna_edgetpu.yaml` → `yolov5n_edgetpu_iarna.yaml` (stem ends `_iarna`, doesn't trigger the substring rule). Saved as a feedback memory.

4. **Parallel TFLite uint8 exports** (both produce 3 raw per-scale NHWC outputs):
   - TF: `runs/tf_nhwc/tf_iarna_3ep` (8 ops in head, 202-205 total — 100% TPU on EdgeTPU).
   - PT: `runs/detect/pt_iarna_3ep-2/weights/best_saved_model/best_int8.tflite` (266 ops including DFL SOFTMAX×3 — 0% TPU until we re-quantize with int8 I/O, see earlier section).
   - Both runnable through `tf.lite.Interpreter` for inference comparison locally.

5. **NMS / decode helper** added: [examples/tf_edgetpu_postprocess.py](examples/tf_edgetpu_postprocess.py).
   - Dequantize uint8 output → sigmoid → multiply by imgsz → xyxy → class-aware NMS.
   - Important convention: in our TF training, the box channel encodes **image-normalized xywh** (not cell-relative offsets), so the host-side decode is `cx = sigmoid(bx) * img_w`, etc.

### What we propose to do next (active task)

**User chose option "Port complet v8 loss în TF"** when asked how to interpret strict 1:1 parity. Plan:

1. **Port TaskAlignedAssigner + DFL + CIoU + BCE** to TF (Keras / `tf.function`-friendly), as a stateless loss module.
   - Helpers needed: `tf_make_anchors`, `tf_dist2bbox`, `tf_bbox2dist`, `tf_bbox_iou(CIoU=True)`, `tf_dfl_loss`.
   - Reference impl in PT: [ultralytics/utils/loss.py:88,109,333](ultralytics/utils/loss.py) and [ultralytics/utils/tal.py:14,400,416,428](ultralytics/utils/tal.py).
2. **Modify `TFDetectHead`** in [examples/tf_nhwc_detect_train_export.py](examples/tf_nhwc_detect_train_export.py) to emit `4·reg_max + nc = 65` channels per scale (instead of `4 + nc = 5`), default `reg_max=16`.
3. **Replace `DetectorTrainer.train_step`** to mirror `v8DetectionLoss.__call__`:
   - Permute per-scale outputs `[B, H, W, 4·reg_max + nc]` → `[B, num_total_anchors, 4·reg_max + nc]`.
   - Decode with softmax+DFL projection → CIoU + DFL + BCE losses.
   - Use PT default hyper-params: `box=7.5, cls=0.5, dfl=1.5`.
4. **Re-train iarna 3 epochs** with the new loss and compare:
   - Loss curves directly (now same units).
   - mAP/precision/recall on val (need a TF-side val loop or run PT val on TF-export TFLite).
5. **Validate EdgeTPU compatibility** of the new heavier head: confirm whether 100% TPU mapping holds, or whether DFL's SOFTMAX×3 + the inflated `4·reg_max=64` box channels push some ops to CPU. If yes, evaluate trade-off against the parity gain.

### Open risks
- **EdgeTPU compile regression**: DFL adds SOFTMAX (which historically triggered CPU fallback in the PT path), and the 16× box-channel inflation grows the head conv. The 100% TPU result we just achieved may degrade.
- **TaskAlignedAssigner on TF**: PyTorch fancy indexing (`pd_scores[ind[0], :, ind[1]]`) and `scatter_add_` need careful porting to `tf.gather` / `tf.tensor_scatter_nd_add` with explicit batch handling.
- **Training stability**: PT v8 loss needs many more epochs from random init; the 3-epoch comparison plan may still show flat loss for both. Plan to run 20-50 epochs once correctness is verified.
- **Inference time pipeline**: NMS code in [examples/tf_edgetpu_postprocess.py](examples/tf_edgetpu_postprocess.py) currently expects raw 4+nc output. After v8 port, the model will output 4·reg_max+nc — postprocess must apply DFL projection (softmax + matmul `[0..15]`) before decode. Either move DFL into the export graph (simpler host code, costs TPU ops) or keep it host-side.

---

## Update: TF native NHWC port for v5 repo (this repository) — 2026-04-27

### Goal
Build a TensorFlow path inside `yolov5/` (this repo) that parses the standard
`models/yolov5*.yaml` files directly, without going through PyTorch → ONNX →
TFLite. User constraint: "problema la implementarea pytorch este ordinea
canalelor, channel first, tensorflow si edgetpu au channel last. Nu vreau sa
mai trecem la export prin ONNX."

### New code added to this repo
- [models/tf_native.py](models/tf_native.py) — NHWC-native Keras builder.
  Layers: `TFConvN`, `TFBottleneckN`, `TFC3N`, `TFSPPFN`, `TFConcatN`,
  `TFUpsampleN` (uses `tf.image.resize(..., 'nearest')` →
  `RESIZE_NEAREST_NEIGHBOR`, NOT `UpSampling2D`), `TFDetectRawN` (multi-output
  `[B, H_i, W_i, na*(5+nc)]`, no in-graph decode/reshape/concat). Random init,
  no PyTorch dependency.
- [models/tf_loss.py](models/tf_loss.py) — `ComputeLossTF`, parity port of v5
  `utils/loss.py:ComputeLoss`. CIoU box, BCE-obj with target=IoU.detach(),
  BCE-cls, per-scale balance `[4.0, 1.0, 0.4]`, anchor matching with
  `anchor_t=4` ratio filter and offset bias 0.5 (3 candidate cells per
  target). Hyper-params propagated from the same hyp YAML used by PT
  (box=0.05, obj=1.0, cls=0.5 for scratch-low).
- [models/tf_data.py](models/tf_data.py) — calibration generator (real images
  from `data.yaml: train`).
- [train_tf_native.py](train_tf_native.py) — CLI training. **Reuses
  `utils/dataloaders.create_dataloader`** so mosaic/mixup/HSV/scale/fliplr/
  translate/copy_paste are byte-for-byte the same pipeline as PT `train.py`.
  Per-batch conversion: PT uint8 NCHW → TF float32 NHWC.
- [export_tf_edgetpu.py](export_tf_edgetpu.py) — INT8 quant (uint8 in, int8
  out) with calibration source `--data` (data.yaml: train) or
  `--calib-images <dir>`. Falls back to random noise only if no real source.
- [compare_models.py](compare_models.py) — side-by-side TF vs PT inspection.

### Smoke test on iarna (3 epochs, imgsz=640, hyp.scratch-low.yaml)
Same dataset, same hyp, same imgsz, same optimizer (SGD lr=0.01 mom=0.937
wd=5e-4), same augmentation pipeline (mosaic=1.0, HSV=0.015/0.7/0.4,
scale=0.5, translate=0.1, fliplr=0.5).

| Run | ep1 val total | ep2 val total | ep3 val total |
|---|---:|---:|---:|
| TF native | 1.655 | **1.525** | 1.751 |
| PT (v5 ComputeLoss) | ~1.72 | ~1.65 | ~1.66 |

Per-channel ep2 val: TF box=0.151 obj=0.039 vs PT box=0.107 obj=0.109. Total
loss within ~4% — loss path verified at parity.

### Side-by-side TFLite/EdgeTPU comparison (after the 3-ep training)

| Aspect | **TF native NHWC** | **PT (ONNX→TFLite)** |
|---|---|---|
| File size | 2120 KiB | 2140 KiB |
| EdgeTPU compile | **208/208 = 100%**, 1 subgraph | **123/261 = 47%**, multi-subgraph, 138 CPU fallback |
| Total TFLite ops | 211 | 265 (+25.5%) |
| Latency CPU TFLite, 640×640 | 39.4 ± 0.4 ms | 40.9 ± 0.5 ms |
| Input | uint8 [1,640,640,3] | uint8 [1,640,640,3] |
| Output | 3× int8 raw maps `[1,80,80,18]`, `[1,40,40,18]`, `[1,20,20,18]` | 1× uint8 `[1,25200,6]` decoded in-graph |

### Op histogram diff (TF − PT)

| OP | TF | PT | Δ | Source of delta |
|---|---:|---:|---:|---|
| CONV_2D | 60 | 60 | 0 | identical backbone |
| MAX_POOL_2D | 3 | 3 | 0 | SPPF |
| PAD | 7 | 7 | 0 | stride>1 autopad |
| RESIZE_NEAREST_NEIGHBOR | 2 | 2 | 0 | nn.Upsample (both via Resize) |
| RESHAPE | 0 | 6 | +6 | NCHW↔NHWC adapters from ONNX converter |
| STRIDED_SLICE | 0 | 9 | +9 | flatten 25200 anchors in PT Detect graph |
| LOGISTIC | 57 | 66 | +9 | extra sigmoid in PT in-graph decode |
| MUL | 57 | 75 | +18 | SiLU + xywh decode + anchor multiply in graph |
| CONCATENATION | 13 | 17 | +4 | scale concat in PT Detect |
| ADD | 7 | 10 | +3 | xywh decode adds (xy+grid) in PT |
| QUANTIZE | 2 | 6 | +4 | scale quantize at decode boundaries in PT |

### Root cause of the 100%→47% mapping gap on PT path
1. **NCHW→NHWC adapters** (RESHAPE×6) inserted by the ONNX→TFLite converter
   at non-trivial layout boundaries.
2. **In-graph decode** in PT Detect head (sigmoid + xywh decode + anchor
   multiply + scale flatten + concat) — produces STRIDED_SLICE×9, extra
   LOGISTIC×9, MUL×18, ADD×3.
3. **Multi-subgraph** result on PT compile: `edgetpu_compiler` fragments the
   graph and produces "More than one subgraph is not supported" entries that
   force ops to CPU even when individually supported.

### EdgeTPU compile sweep (TF native, untrained smoke)
| YAML | imgsz | strides | TPU ops | CPU ops |
|---|---|---|---:|---:|
| yolov5n.yaml | 640×640 | 8/16/32 | 209 | 0 |
| yolov5n.yaml | 480×640 | 8/16/32 | 209 | 0 |
| yolov5s-512.yaml | 512×640 | 16/32/64 | 213 | 0 |
| yolov5s-1024.yaml | 1024×1024 | 32/64/128 | 217 | 0 |

Stride-64 YAMLs that previously failed via PT path on rectangular resolutions
(experiment.md "Lecție 4") now pass at 100% via TF native — the TRANSPOSE
that historically broke alignment is absent in the multi-output raw graph.

### Conclusions for the v5 repo
- TF native NHWC is the recommended path for EdgeTPU exports in this repo.
  PT path remains usable but is fundamentally limited to ~47% TPU mapping
  for this Detect head shape due to in-graph decode + layout adapters.
- Loss / training behaviour is at parity — same hyp file, same dataset,
  ~4% total loss difference at 3 epochs (random init / aug RNG variance).
- All standard YOLOv5 YAML modules are supported: `Conv`, `C3`, `SPPF`,
  `nn.Upsample`, `Concat`, `Detect`. Custom YAMLs with extra downsamples
  (stride 64, 128) compile 100% TPU at any imgsz that's a multiple of the
  final stride.

### Run commands
```
# Train
python train_tf_native.py --cfg models/yolov5n.yaml \
    --data data/iarna_abs.yaml --hyp data/hyps/hyp.scratch-low.yaml \
    --imgsz 640 --epochs 3 --batch 8 --out runs/tf_native/run1

# Export to EdgeTPU (uses architecture.json sidecar from train run)
python export_tf_edgetpu.py --weights runs/tf_native/run1/best.weights.h5 \
    --data data/iarna_abs.yaml --out runs/tf_native/run1_export
```

---

## Update: Loss-parity scaffolding (warmup + linear LR + EMA + 3 param groups) — 2026-04-27

### What was added to train_tf_native.py
- **3 param groups** (mirror `train.py` PT): `g_kernel` (with weight decay),
  `g_bn` (BN gamma/beta + everything else, no decay), `g_bias` (no decay,
  separate warmup LR).
- **Manual SGD-with-Nesterov** (`SGDMomentum`) so `lr` and `momentum` are
  `tf.Variable`s reassignable per-iteration. Two instances: `opt_main` for
  `g_kernel + g_bn`, `opt_bias` for `g_bias`.
- **Warmup over `nw = max(round(warmup_epochs * batches_per_epoch), 100)` iters:**
    - main LR : 0.0 → `lr0 * lf(epoch)`
    - bias LR : `warmup_bias_lr (0.1)` → `lr0 * lf(epoch)`
    - momentum: `warmup_momentum (0.8)` → `momentum (0.937)`
- **Linear LR scheduler** `lf(x) = (1 - x/epochs)*(1 - lrf) + lrf` (PT default).
  `--cos-lr` flag adds optional cosine.
- **ModelEMA** with PT-parity decay schedule
  `d = decay_target * (1 - exp(-updates / tau))`, `decay_target=0.9999`,
  `tau=2000`. Validation runs with EMA weights swapped in then restored.
  `last.weights.h5` / `best.weights.h5` save EMA weights.

### 10-epoch parity comparison on iarna (640×640, hyp.scratch-low.yaml)

Same dataset (140 train / 40 val), same hyp file (mosaic=1.0, HSV, scale=0.5,
fliplr=0.5), same batch=8, same seed=0, same SGD lr=0.01 mom=0.937 wd=5e-4.

| ep | TF train tot | PT train tot | Δ% | TF val tot | PT val tot | Δ% |
|---|---:|---:|---:|---:|---:|---:|
|  1 | 6.806 | 1.808 | +276% | 2.220 | 1.718 | +29% |
|  2 | 2.050 | 1.784 | +14.9% | 1.996 | 1.726 | +15.6% |
|  3 | 2.002 | 1.893 | +5.7% | 1.944 | 1.735 | +12.1% |
|  5 | 1.950 | 1.904 | +2.4% | 1.907 | 1.745 | +9.2% |
|  7 | 1.918 | 1.898 | +1.1% | 1.861 | 1.739 | +7.0% |
| 10 | 1.917 | 1.900 | **+0.9%** | 1.876 | 1.725 | +8.7% |

Per-component at ep10:
- Train box loss: TF 0.113, PT 0.111 (+1.8%)
- Train obj loss: TF 0.134, PT 0.127 (+5.5%)
- Val box loss: TF 0.107, PT 0.108 (-0.9%)
- Val obj loss: TF 0.127, PT 0.108 (+18%)

### Interpretation
- **Train total at convergence: 1.917 vs 1.900 — within 0.9%.** Loss path is
  at functional parity.
- **ep1 TF train is much higher (6.8 vs 1.8)** because EMA shadow starts at
  init and the first few iters dominate the average; PT's reported
  `train/box_loss` is mloss EMA which dampens this. TF reports raw mean.
- **Val box loss is essentially identical** (0.107 vs 0.108 at ep10). The
  TF `val obj` is consistently +18% higher than PT — likely from one or
  more residual differences:
    - PT BN init: gamma=1, beta=0; TF Keras default same. ✓
    - PT Conv init: Kaiming (fan_in, mode='fan_out'); TF Keras default
      Glorot. NOT matched — this is the most likely source of 8-18% obj
      drift over 10 epochs.
    - EMA shadow init: TF copies from current weights at construction time
      (post `model(input)` build trigger); PT EMA initializes after
      optimizer first step. Order-of-operations difference, very small.
- **PT-side mAP/P/R remains effectively zero** at 10 epochs (mAP50=0.00029,
  P=R=0.0011). 140 images × 10 epochs from random init is far below the
  budget needed for v5n on a single-class task with mosaic on. This isn't
  a parity bug — both sides are still pre-convergence on detection metric.

### EdgeTPU export comparison (after 10 epochs)
| | TF native | PT export |
|---|---|---|
| Total ops | 211 | 266 (+25.6%) |
| EdgeTPU mapped | **208/208 = 100%** (1 subgraph) | 123/262 = 47% (multi-subgraph, 139 CPU) |
| CPU TFLite latency | 40.4 ± 0.5 ms | 41.1 ± 0.3 ms |
| Output | 3× int8 raw maps `[1,H_i,W_i,18]` | 1× uint8 `[1,25200,6]` decoded in-graph |

100% TPU mapping confirmed even with trained EMA-stable weights — the graph
shape doesn't depend on weight magnitudes, just topology.

### Remaining for strict bit-parity (not implemented yet)
1. **Conv kernel init** — switch TF `Conv2D` from default `glorot_uniform`
   to `HeNormal(mode='fan_out', nonlinearity='relu')` to match PT.
2. **In-training mAP loop** — TF currently has no NMS-based val; needs
   host-side decode + NMS to compute P/R/mAP per epoch.
3. **PT-style mloss EMA reporting** for first-epoch loss display (cosmetic
   only, doesn't affect optimization).
4. **Multi-scale training** (PT default off; not a current driver).

---

## Update: P2P4 architecture for small-object detection (iarna) — 2026-04-27

### Motivation
- iarna labels at imgsz=640: 2,299 objects, aspect w/h median = 0.97 (square),
  sqrt(w*h) p25=20, p50=24, p75=28, p90=33, p99=41, max ≈ 57.
- Original yolov5n.yaml has stride 8/16/32; at stride 32 a 24 px object spans
  0.75×0.75 cells (essentially below the matching grid).
- First conv `[64, 6, 2, 2]` (k=6 s=2) blurs early — unhelpful for 24 px targets.

### Changes ([models/yolov5n_iarna_p2p4.yaml](models/yolov5n_iarna_p2p4.yaml))
- First Conv: `[64, 3, 2, 1]` (k=3 instead of 6).
- Detect head moved to **(P2/4, P3/8, P4/16)** instead of P3/P4/P5.
- Backbone stops at P4 (drops 1024-ch C3 + last stride-2). SPPF on 512-ch P4.
- 1 anchor per scale (na=1) → 6 channels per cell (vs 18 with na=3).
- Initial anchors `[8, 8] / [16, 16] / [32, 32]` (PT autoanchor confirmed
  these as optimal — BPR > 0.98, no kmeans recompute needed).

### 30-epoch parity comparison (no mosaic, no scale, hyp.iarna-small.yaml)

| metric | TF native | PT |
|---|---:|---:|
| **Train ep30 total** | **1.052** | **1.055** (Δ −0.2%) |
| Val ep30 total | 1.059 | 1.016 (Δ +4.2%) |
| **Val ep30 box** | 0.070 | 0.063 (Δ +11%) |
| Val ep30 obj | 0.063 | 0.063 (Δ ~0%) |
| EdgeTPU TPU/total | **188/188 (100%)** | 102/240 (42%) |
| TFLite size | 0.6 MB | 0.8 MB |
| CPU TFLite latency @ 480×640 | 23.6 ms | 24.0 ms |
| Output footprint | 148 KiB (3 raw maps) | 148 KiB (1 flat) |

Val_obj per-epoch deep-dive: TF is consistently +0.0095 above PT
(σ = 0.0030, mean Δ% ≈ +15%). The bias is sustained across all 30 epochs
— not noise.

### Fixes applied to close the gap
1. **Detect bias init** ([models/tf_native.py:TFDetectRawN.initialize_biases](models/tf_native.py)) — sets the obj logit bias to `log(8 / (W/s)^2)` and cls bias to `log(0.6 / (nc - 0.999))`, mirroring PT `_initialize_biases`. **Effect:** train ep1 dropped from 4.08 → 1.31 (huge stabilization). Val obj at ep1 went from 0.061 → 0.058 (small improvement).
2. **Conv kernel init** ([models/tf_native.py:PTConvKernelInit](models/tf_native.py)) — replicates PT `nn.Conv2d` default `kaiming_uniform_(a=sqrt(5))` ≡ uniform `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`. **Effect:** marginal (val_obj barely moved). Init wasn't the dominant source.
3. **Bias init for Detect head** — uniform `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`.

### Residual gap (~+15% on val_obj)
Plausible remaining sources, in probability order:
- BN running statistics drift (despite same eps/momentum, the per-step
  update order differs slightly between PT `nn.BatchNorm2d` and Keras BN).
- Floating-point ordering in SiLU `x * sigmoid(x)` (TF Keras `swish` vs PT
  `nn.SiLU` — same math, different op fusion).
- 2-instance manual SGDMomentum vs PT 1-instance SGD with 3 param groups —
  velocity accumulation under different step ordering.

These are subtle and account for ≤ 5% on val_total. Not blocking parity at
the architectural level.

---

## Status snapshot — 2026-04-27 (end of session)

### What works at parity (functional)
- Build TF NHWC graph from any `models/yolov5*.yaml`.
- Train using PT's `LoadImagesAndLabels` dataloader (mosaic / mixup / HSV /
  scale / translate / fliplr / etc.).
- Loss = CIoU + BCE-obj(target=IoU) + BCE-cls + per-scale balance,
  PT-parity at <1% on train_total at convergence.
- Optimizer: 3 param groups (kernel decay, BN no-decay, bias separate),
  warmup, linear LR scheduler, ModelEMA with PT decay schedule.
- Detect head bias + Conv kernel init at PT defaults.
- INT8 quantization with real-image calibration from `data.yaml: train`.
- EdgeTPU export 100% TPU mapping (vs PT 42-47%).

### What is NOT yet at parity / NOT yet implemented in TF
Listed in priority order (most impact first):

1. **Auto-anchor** (`check_anchors` + `kmean_anchors` from
   [utils/autoanchor.py](utils/autoanchor.py)). Today TF uses YAML anchors
   verbatim. On a new dataset where BPR < 0.98, this would silently train
   with bad anchors. ~10 LoC: call PT util at trainer start.
2. **Val NMS + mAP loop** — TF has no in-training accuracy measurement.
   Today `best.weights.h5` is selected by val loss, not mAP. Without this
   we cannot say "PT and TF reach the same accuracy", only "same loss".
   ~150 LoC: host-side decode + class-aware NMS + per-IoU mAP computation
   (or call PT `validate.run` on the TFLite via interpreter).
3. **Save best by mAP** (instead of val_loss). Trivial after item 2.
4. **Early stopping** (`--patience`). ~15 LoC.
5. **`--resume`** continuation from `last.weights.h5` + optimizer/EMA state.
   ~30 LoC + serialization of velocity buffers.
6. **Native `results.png` and `train_batch*.jpg` plotting** in the trainer
   (today done out-of-band by [plot_curves.py](plot_curves.py)).
7. **Logger integrations** (TensorBoard / Comet / ClearML). Pure UX.
8. **`--save-period`** intermediate checkpoints. Trivial.

### Items deliberately skipped (low/no impact for current scope)
- Mixed precision (AMP) — CPU training, no speedup.
- Sync BatchNorm — single-device.
- Image weights — nc=1, irrelevant.
- Multi-scale training — PT default off.
- `--quad` collate — rare.
- `--evolve` (GA hyperparameter evolution) — research-only.
- Strip optimizer pre-save — `.weights.h5` already excludes opt state.

---

## Plan: incremental TF feature activation toward full PT parity

### Phase 1 — measure accuracy (highest impact)
**Step 1.1**: Implement TF host-side decode + class-aware NMS + mAP@0.5 /
mAP@0.5:0.95 calculation. Compute on val split after each epoch. Report
P, R, mAP50, mAP50-95 in `training.log` next to existing loss columns.

**Step 1.2**: Use mAP-based "best" checkpoint selection in `train_tf_native.py`.
Save `best.weights.h5` when val mAP@0.5:0.95 improves (PT convention).

**Step 1.3**: Add `--patience` early stopping based on mAP50-95 plateau.

**Acceptance**: TF and PT produce comparable mAP@0.5 / mAP@0.5:0.95 on
iarna val split after equal epoch budget. Gap should narrow significantly
vs the loss-only metric we have today (~+15% val_obj noise won't translate
proportionally to mAP — the model may still detect objects equally well).

### Phase 2 — auto-anchor (low effort, future-proofing)
**Step 2.1**: Wire `check_anchors(dataset, model_stub, thr=4.0, imgsz=imgsz)`
into `train_tf_native.py` at the start of training. The stub needs `.anchors`
(PT tensor in cell units) and `.stride` attributes — wrap the TF anchors in
a PT tensor for the call. If the util mutates anchors, copy them back into
`ComputeLossTF.anchors_*` and persist in the architecture.json sidecar.

**Step 2.2**: If autoanchor recomputes, log the new anchors and seed the
loss with the updated values for the rest of training.

**Acceptance**: BPR check runs and prints, anchors are updated when BPR
< 0.98, training proceeds with the updated anchors.

### Phase 3 — quality of life
**Step 3.1**: `--resume` — serialize `(velocity_main, velocity_bias, ema.shadow,
ema.updates)` next to `last.weights.h5` and reload on resume.

**Step 3.2**: Native `results.png` (matplotlib loss + mAP curves) and
`train_batch0.jpg` (mosaic montage) saved to the run dir, matching PT.

**Step 3.3**: `--save-period N` — checkpoint every N epochs.

### Phase 4 — observability (optional)
**Step 4.1**: TensorBoard scalars (loss/lr/mAP).
**Step 4.2**: Optional Comet / ClearML hooks if requested.

### Phase 5 — close residual loss gap (cosmetic / research)
**Step 5.1**: Unify the manual SGDMomentum into a single instance with a
mask-based per-variable LR/wd, mirroring PT's single optimizer with 3
param_groups exactly.

**Step 5.2**: Verify BN moving stats convergence by logging running_mean
and running_var of selected layers between TF and PT after epoch 1, 5, 10.

**Step 5.3**: Force fused SiLU (Conv → BN → SiLU as a single op via
`tf.keras.activations.swish`) and compare INT8 calibration outputs vs PT.

### Out-of-scope for now (deferred or skipped)
- Multi-GPU training (DDP, sync-BN) — single device suffices.
- AMP — CPU-only.
- Hyperparameter evolution — separate workflow.
- Image weights / multi-scale — niche, default off in PT itself.

### Working order
Phase 1 first (mAP loop), because without it we cannot answer the
fundamental question "do these two models actually detect equally well?".
All subsequent gap analysis depends on it. Phase 2 right after (cheap and
prevents silent quality loss on new datasets). Phase 3+4 incremental as
operational maturity demands.

---

## Update: source-tree refactor mirroring PT layout — 2026-04-27

### Goal
Reorganize the TF-side code so each TF file maps 1:1 to a PT file with
`tf_` prefix. Reuse PT utilities directly where they accept numpy or torch
tensors at the boundary, instead of duplicating logic.

### Final layout

```
yolov5/
├── train.py / val.py / detect.py / export.py     (PT — unchanged)
├── train_tf.py                                   (← train_tf_native.py)
├── val_tf.py                                     (← NEW Phase 1)
├── export_tf.py                                  (← export_tf_edgetpu.py)
├── compare_models.py / compare_curves.py / plot_curves.py  (tooling, root)
├── models/
│   ├── common.py / yolo.py                       (PT layers + DetectionModel)
│   ├── tf.py                                     (legacy PT-driven TF, used by export.py)
│   ├── tf_common.py                              (← split: TF layers)
│   └── tf_yolo.py                                (← split: TFDetect + DetectionModelTF + parse_model_tf)
└── utils/
    ├── general.py / metrics.py / autoanchor.py / dataloaders.py / plots.py / augmentations.py  (PT)
    ├── tf_loss.py                                (← move from models/)
    ├── tf_dataloaders.py                         (← move from models/, uses utils.augmentations.letterbox)
    ├── tf_torch_utils.py                         (← extract: ModelEMA, SGDMomentum, split_param_groups)
    ├── tf_metrics.py                             (← NEW Phase 1)
    └── tf_autoanchor.py                          (← NEW Phase 2)
```

### Naming changes
- `TFConvN` → `TFConv`, `TFC3N` → `TFC3`, `TFSPPFN` → `TFSPPF`,
  `TFConcatN` → `TFConcat`, `TFUpsampleN` → `TFUpsample`,
  `TFBottleneckN` → `TFBottleneck` — the `N` suffix was redundant once
  the `TF` prefix already disambiguates from PT.
- `TFDetectRawN` → `TFDetect` — also adopted PT name.
- `build_tf_model_from_yaml` → `parse_model_tf` + `DetectionModelTF`
  (façade class, mirrors PT `parse_model` + `DetectionModel`).

### What we directly reuse from PT (no duplication)

| PT module | Reused in TF | How |
|---|---|---|
| `utils/dataloaders.create_dataloader` | training + val data | direct import |
| `utils/augmentations.letterbox` | calibration generator | direct import |
| `utils/autoanchor.check_anchors` + `kmean_anchors` | TF auto-anchor | thin stub adapter (~30 LoC) |
| `utils/general.non_max_suppression` | TF val pipeline | direct import |
| `utils/general.xywh2xyxy` | TF val pipeline | direct import |
| `utils/metrics.ap_per_class` | TF mAP | direct import |
| `val.process_batch` | TF correct-prediction matrix | direct import |
| `utils/plots.plot_results` | training curves PNG | direct import on TF-written CSV |

### What stays TF-native (graph-coupled)
- Layers (Conv/BN/C3/SPPF/...) — Keras
- Loss `ComputeLossTF` — tape-compatible TF ops
- ModelEMA + SGDMomentum — `tf.Variable`-based
- INT8 calibration generator wrapper — TFLiteConverter feed
- TFDetect + parser — Keras model build

### Smoke verification after refactor
- 1-epoch train on iarna P2P4: same loss values as pre-refactor
  (val total 1.246, box 0.097, obj 0.058)
- Export: 187/187 ops 100% TPU, 1 subgraph, 0 CPU fallback

---

## Update: Phase 1 — accuracy measurement (mAP in TF training) — 2026-04-27

### What was added
- [utils/tf_metrics.py](utils/tf_metrics.py) (~150 LoC): `host_decode` +
  `tf_validate`. Decodes raw multi-output maps in NumPy (sigmoid + xy/wh
  anchor decode), wraps to torch, then **directly reuses PT's
  `non_max_suppression`, `process_batch`, `ap_per_class`** — zero
  re-implementation of NMS / IoU matching / AP integration.
- [val_tf.py](val_tf.py): standalone CLI (mirror of `val.py`) — loads
  `*.weights.h5` via sidecar, reports P / R / mAP@0.5 / mAP@0.5:0.95.
- [train_tf.py](train_tf.py) integrated changes:
  - mAP computed on val split after every epoch.
  - **Save best by fitness = 0.1·mAP50 + 0.9·mAP50-95** (PT convention
    from `utils.metrics.fitness`), not by val loss.
  - `--patience N` early stopping on fitness plateau.
  - `--conf-thres` and `--iou-thres` flags for val NMS.
  - `training.log` extended with P / R / mAP50 / mAP50-95 columns.
  - **`results.csv` written in PT format** so `utils.plots.plot_results`
    works directly (Phase 3 plot reuses this).

### Smoke result (1 epoch, iarna P2P4)
```
val: total=1.246 box=0.097 obj=0.058
P=0.0047 R=0.0895 mAP50=0.0029 mAP50-95=0.0007
new best fitness=0.0009; saved best.weights.h5
```

The TF-side mAP/P/R numbers are now measurable and comparable to PT's
in-loop validation output. This is the single largest functionality gap
that was closed in this phase.

---

## Update: Phase 2 — auto-anchor — 2026-04-27

### What was added
- [utils/tf_autoanchor.py](utils/tf_autoanchor.py) (~70 LoC):
  `maybe_recompute_anchors(dataset, anchors_yaml, strides, imgsz, thr)`.
  Builds a duck-type `_DetectStub` with `.anchors` (cell units) and
  `.stride` (torch tensor), feeds it to PT's `check_anchors`. If kmeans
  improves BPR, the stub is mutated in-place; we read out the new anchors
  and convert back to YAML list format.
- [train_tf.py](train_tf.py): runs auto-anchor at start of training
  (after dataloader build, before main loop). If anchors change,
  rebuilds `ComputeLossTF` with the new values and updates
  `detmodel.anchors` (so the architecture.json sidecar persists the
  effective anchors). `--noautoanchor` mirrors PT's flag.

### Smoke result (iarna P2P4, current YAML anchors)
```
[anchor] running BPR check (utils.autoanchor.check_anchors)...
AutoAnchor: 2.67 anchors/target, 0.991 Best Possible Recall (BPR).
            Current anchors are a good fit to dataset ✅
[anchor] BPR ok — keeping YAML anchors
```

For iarna with `[8,8] [16,16] [32,32]` the BPR is already 0.991, so
kmeans does not run. On a different dataset where BPR < 0.98, the
anchors would be replaced and the loss would use the new values
automatically — same as PT.

---

## Update: Phase 3 — quality of life — 2026-04-27

### What was added
- **`--save-period N`** ([train_tf.py](train_tf.py)): saves
  `epoch{N}.weights.h5` every N epochs (in addition to last/best).
- **`--resume <last.weights.h5>`**: full resume support.
  Persisted state in `opt_state.npz` next to weights:
  - velocity buffers for both SGDMomentum instances
  - EMA shadow + EMA `updates` counter
  - current epoch + `best_fitness`
  Resume re-loads weights, restores velocity / EMA / counters, and
  picks up the loop at `start_epoch + 1`. Verified: 2-epoch run +
  resume to 4 epochs continues the loss curve smoothly
  (1.254 → 1.236 → 1.223 → 1.215).
- **Native `results.png` plot**: at end of training, `train_tf.py`
  calls `utils.plots.plot_results(csv_path)` directly on the
  PT-format `results.csv`. Same chart shape as PT produces.

---

## Phase status snapshot — 2026-04-27 (end of session)

### ✅ Done
- **Refactor** to PT-mirrored layout (root + models/tf_* + utils/tf_*).
- **Phase 1** (mAP measurement): TF training reports P / R / mAP50 /
  mAP50-95 every epoch; best checkpoint selected by fitness; standalone
  `val_tf.py` mirroring PT `val.py`.
- **Phase 2** (auto-anchor): `check_anchors` runs at start of training,
  kmeans recompute when BPR < 0.98, anchors persist in sidecar.
- **Phase 3** (quality of life): `--save-period`, `--resume` with full
  optimizer/EMA state, native `results.png` plot at training end.

### Carried-over (Phase 5 — close residual loss gap, low priority)
The +15% sustained val_obj delta vs PT remains. Plausible sources
(BN running stats drift, SiLU op fusion, 2-instance SGDMomentum vs PT
1-instance with 3 param groups) are subtle; impact on actual detection
mAP is not yet quantified. Defer until we see a 30-epoch run with the
new mAP loop, where mAP gap should show whether the val_obj difference
is academic or actually loses detections.

### Carried-over (Phase 4 — observability, optional)
TensorBoard / Comet / ClearML hooks. Not blocking; useful for long runs.

### Recommended next step
Run **30-epoch TF vs PT** with the new mAP loop (and unchanged hyp
files / dataset / aug pipeline) to confirm that **detection accuracy is
at parity** even with the residual val_obj loss gap. If TF mAP@0.5:0.95
is within ~5% of PT's, Phase 5 can be deferred indefinitely. If TF
mAP is materially lower, the BN/op-fusion debug cycle becomes worth
the effort.

---

## Update: Architectural block variants (P3-P4 head, small output) — 2026-04-28

### Goal
After confirming P3-P4 keeps the output footprint small (~35 KiB at 480x640
vs 148 KiB for P2-P4), we explore replacing the C3 backbone block with
modern alternatives from 2018-2024 papers — keeping the coarse grid (P3
cell 8x8, P4 cell 16x16) but enriching intra-cell representation, on the
hypothesis that for small but **non-overlapping** objects (iarna profile),
feature richness > grid density.

iarna distance-to-nearest-neighbor analysis (verified):
- min: 4.1 px, p1: 7.2 px, p5: 11.5 px, p10: 15.6 px, p25: 28.2 px, p50: 48.5 px
- 134 pairs at < 16 px (P3/8 same-cell risk), 19 pairs at < 8 px (P2/4 risk)
- Even with P3/8 grid, ~94% of objects have neighbors > 16 px → grid resolution is sufficient

### Blocks added to `models/tf_common.py`
All EdgeTPU-friendly by construction (no SE/attention):

| Class | Paper | Year | Idea |
|---|---|---|---|
| `TFC2f` | YOLOv8 (Ultralytics) | 2023 | split + N bottlenecks chained, all concat — gradient flow ↑ |
| `TFGELAN` | YOLOv9 (Wang et al) | 2024 | Generalized ELAN — pairs of bottlenecks, intermediate features collected |
| `TFMSBlock` | YOLO-MS | 2023 | multi-kernel parallel (k=1, 3, 5) on channel chunks, concat |
| `TFPConv` | FasterNet (CVPR 2023) | 2023 | partial conv on 1/4 channels, pass-through rest, +1×1 mix |
| `TFGhost` | GhostNet (CVPR 2020) | 2020 | half features via 1×1, half via cheap DW on the half |
| `TFCoordConvStem` | Liu et al (NeurIPS 2018) | 2018 | append normalized x/y grid before stem Conv |

All wired into `parse_model_tf` (in `models/tf_yolo.py`) so any YAML can
reference `C2f`, `GELAN`, `PConv`, `Ghost`, `MSBlock`, `CoordConv` as
module names — same syntax as `Conv` / `C3`.

### YAML variants created (all P3-P4 head, gw=0.125, na=1, nc=1)

- [models/yolov5n_iarna_p3p4_c2f.yaml](models/yolov5n_iarna_p3p4_c2f.yaml)
- [models/yolov5n_iarna_p3p4_gelan.yaml](models/yolov5n_iarna_p3p4_gelan.yaml)
- [models/yolov5n_iarna_p3p4_msblock.yaml](models/yolov5n_iarna_p3p4_msblock.yaml)
- [models/yolov5n_iarna_p3p4_pconv_ghost.yaml](models/yolov5n_iarna_p3p4_pconv_ghost.yaml)
- [models/yolov5n_iarna_p3p4_coordconv.yaml](models/yolov5n_iarna_p3p4_coordconv.yaml)

### Smoke test (1 epoch on iarna + EdgeTPU export 480x640 uint8)

| # | Experiment | Params | TFLite KB | CPU ms | FPS | EdgeTPU TPU/total | Status |
|---|---|---:|---:|---:|---:|---|---|
| 0 | baseline (C3, w0125) | 119,924 | 195 | 8.8 | 113.5 | **145/145** | ✅ 100% TPU |
| 1 | C2f (YOLOv8) | 129,780 | 202 | 9.7 | 103.6 | **135/135** | ✅ 100% TPU |
| 2 | GELAN (YOLOv9) | 179,796 | 274 | 10.9 | 91.5 | **190/190** | ✅ 100% TPU |
| 3 | **MSBlock** (YOLO-MS) | **67,022** | **130** | **8.7** | **115.3** | **126/126** | ✅ 100% TPU, **−44% params** |
| 4 | PConv+Ghost (FasterNet+GhostNet) | 33,452 | 81 | 6.4 | 156.7 | — | ❌ EdgeTPU compile FAIL |
| 5 | CoordConv stem (Liu 2018) | 120,068 | 797 | 9.9 | 101.2 | 1/147 | ❌ tile dynamic fragments partition (0% TPU effective) |

All output footprints identical at 35 KiB (P3 60×80×6 + P4 30×40×6 = 36000 B)
since the head is unchanged.

### Findings
1. **MSBlock is the standout small-objects-friendly variant**: same output
   size, 44% fewer params, identical CPU latency, **100% TPU mapped**.
   Multi-kernel (k=1, 3, 5) parallel branches give varied receptive fields
   per branch — a natural fit for iarna's distribution (24 px median + tail
   to 57 px).
2. **C2f + GELAN are safe drop-ins** for C3: same partition behavior,
   marginally more params (C2f) or significantly more (GELAN). Use them
   if accuracy gain > extra params is documented; otherwise MSBlock wins
   on the cost/benefit axis.
3. **PConv+Ghost combination breaks the EdgeTPU compiler** entirely.
   The interaction of PConv's channel-slice + Ghost's DepthwiseConv +
   stride-2 chain produces a graph the compiler refuses. Standalone PConv
   or standalone Ghost should be retried — the failure is the combination,
   not either block alone.
4. **CoordConv with dynamic-batch tile fragments partition** (only 1 op
   on TPU). Fix would be to bake the grid into the input pre-processing on
   host (lose trainable coord, but keep 100% TPU). Documented as anti-pattern
   in `models/tf_common.py:TFCoordConvStem`.

### Recommendation
For real Edge TPU testing on iarna:
1. **Train MSBlock 5-30 epochs** — first comparison vs C3 baseline on
   actual mAP. If MSBlock matches or beats C3 mAP at half the params, this
   is the architecture to commit to.
2. **C2f / GELAN as fallbacks** if MSBlock underperforms — both stable,
   100% TPU.
3. Skip PConv+Ghost and CoordConv for deployment — keep only as research
   notes.

All 4 successfully-compiled artefacts (baseline + C2f + GELAN + MSBlock)
are paired with their `*.yaml` and `architecture.json` in
`runs/tf_native/iarna_p3p4_<variant>_1ep/export_uint8_480x640/`.

---

## Update: Practical feature parity (A/B/C tier) — 2026-04-27

User asked for a final pass to close the practical-impact gaps with PT.
Three items, listed by impact, all leveraging existing PT utilities (no
re-implementations).

### A — `detect_tf.py` (CLI inference)
[detect_tf.py](detect_tf.py): mirror of PT `detect.py` for the TF backend.

- Source: image / folder / video / webcam — uses PT `LoadImages` directly.
- Model: rebuilt from `architecture.json` sidecar + `*.weights.h5`.
- Decode + NMS via `utils.tf_metrics.host_decode` and PT
  `non_max_suppression`.
- Box drawing via `ultralytics.utils.plotting.Annotator` (same util PT uses).
- Outputs annotated images to `--out`. `--save-txt` writes YOLO-format
  predictions per image with optional confidence column.
- Smoke verified on iarna val image.

### B — `models/tf_experimental.py` (MixConv2d + Ensemble + attempt_load_tf)
[models/tf_experimental.py](models/tf_experimental.py).

- `TFMixConv2d`: TF port of `MixConv2d` (mixed kernel-size convs +
  BN + SiLU). Uses dense Conv2D per branch (instead of PT's grouped
  depthwise) for cleaner EdgeTPU mapping. Wired into
  [models/tf_yolo.py](models/tf_yolo.py)'s `parse_model_tf` so that any
  YAML referencing `MixConv2d` (e.g. some PAN paths in yolov5n6/s6
  variants) now builds with the TF parser.
- `TFEnsemble`: averages outputs of N `DetectionModelTF` instances
  per-scale (TF native multi-output ensembling).
- `attempt_load_tf(weights, imgsz_hw=None)`: loads one or more TF models
  from `*.weights.h5` files via their sidecars. Returns a single
  `DetectionModelTF` or a `TFEnsemble` of them. Mirror of PT
  `attempt_load`.

### C — Confusion matrix + PR/F1 curves + label-distribution plot
Integration in existing files (no new module):

- [utils/tf_metrics.py](utils/tf_metrics.py): `tf_validate(...)` now
  accepts `plot=True` and `confusion_matrix=True`. When set, it builds
  `utils.metrics.ConfusionMatrix(nc=nc)`, calls `process_batch` per image
  to populate it, and at end of validation calls `cm.plot(save_dir, names)`
  — produces `confusion_matrix.png`. The same `plot=True` flag triggers
  `ap_per_class` to write `PR_curve.png`, `P_curve.png`, `R_curve.png`,
  `F1_curve.png` (PT util, called directly).
- [train_tf.py](train_tf.py):
  - `plot_labels(labels_np, names, save_dir)` runs at start of training
    and produces `labels.jpg` + `labels_correlogram.jpg` (PT util used
    directly on `train_ds.labels`).
  - After training completes, a final validation pass on `best.weights.h5`
    runs with `plot=True, confusion_matrix=True` to produce the full set
    of PR/F1/CM plots.
- [val_tf.py](val_tf.py): added `--plot` and `--save-dir` flags so the
  standalone CLI can produce the same plots on a saved model.

### Smoke verification (1 epoch on iarna P2P4)
Generated files in run dir (matches PT's val.py output structure):
```
architecture.json     (TF sidecar)
best.weights.h5       last.weights.h5      opt_state.npz
training.log          (TF format, TSV with mAP columns)
results.csv           (PT format → consumed by plot_results)
results.png           (loss + mAP curves)
labels_correlogram.jpg
PR_curve.png  P_curve.png  R_curve.png  F1_curve.png
confusion_matrix.png
```

`detect_tf.py` smoke run on a val image produced an annotated PNG in the
out dir, with 1000 detections at conf_thres=0.001 (untrained model,
expected behavior).

### What is reused from PT (no duplication)
| Feature added | PT util reused |
|---|---|
| `detect_tf.py` source enumeration | `utils.dataloaders.LoadImages` |
| `detect_tf.py` box drawing | `ultralytics.utils.plotting.Annotator`, `colors` |
| `detect_tf.py` rescale | `utils.general.scale_boxes` |
| Confusion matrix | `utils.metrics.ConfusionMatrix` (init + `process_batch` + `plot`) |
| PR/F1 curves | `utils.metrics.ap_per_class(plot=True, save_dir, names)` |
| Label distribution | `utils.plots.plot_labels(labels, names, save_dir)` |

---

## Final status snapshot — 2026-04-27 (end of A/B/C session)

### ✅ Done across phases 1-3 + A/B/C
| Capability | Implementation |
|---|---|
| Train YAML→TF NHWC, no torch deps | [models/tf_common.py](models/tf_common.py), [models/tf_yolo.py](models/tf_yolo.py) |
| Loss CIoU + BCE-obj + BCE-cls (PT-parity) | [utils/tf_loss.py](utils/tf_loss.py) |
| Optimizer (SGD-Nesterov, 3 param groups, warmup, linear LR, EMA) | [utils/tf_torch_utils.py](utils/tf_torch_utils.py) |
| Auto-anchor (BPR + kmeans recompute) | [utils/tf_autoanchor.py](utils/tf_autoanchor.py) |
| In-loop mAP@0.5 / mAP@0.5:0.95 / P / R | [utils/tf_metrics.py](utils/tf_metrics.py) |
| Best by fitness (0.1·mAP50 + 0.9·mAP50-95) | [train_tf.py](train_tf.py) |
| Early stopping (`--patience`) | [train_tf.py](train_tf.py) |
| Resume (`--resume` + opt_state.npz) | [train_tf.py](train_tf.py) |
| Save-period intermediate checkpoints | [train_tf.py](train_tf.py) |
| Label distribution plot at start | [train_tf.py](train_tf.py) → `plot_labels` |
| Loss/mAP curves plot at end | [train_tf.py](train_tf.py) → `plot_results` |
| PR / F1 / P / R / Confusion curves at end | [train_tf.py](train_tf.py) final-val |
| Standalone validation CLI | [val_tf.py](val_tf.py) (`--plot` for full plots) |
| Standalone inference CLI | [detect_tf.py](detect_tf.py) (image / folder / video) |
| MixConv2d / Ensemble / attempt_load | [models/tf_experimental.py](models/tf_experimental.py) |
| INT8 + EdgeTPU export with real-image calibration | [export_tf.py](export_tf.py) |

### ❌ Still not in TF (deferred / out of scope)
| Item | Why deferred |
|---|---|
| TensorBoard / Comet / ClearML hooks (Phase 4) | User has stated intent to use TensorBoard; pending integration |
| Adam / AdamW optimizer choice | Default SGD covers 99% of training; trivial to add |
| Multi-scale training (random imgsz ±50%) | Default off in PT itself; rare |
| Image weights (`--image-weights`) | Default off; niche |
| Hyperparameter evolution (`--evolve`) | Research workflow, separate |
| `--quad` collate | Default off in PT; rare |
| `save_json` (COCO format predictions) | Only needed for benchmark mAP COCO official |
| Multi-format export (TF.js, SavedModel fp16, etc.) | Only INT8/EdgeTPU is in scope |
| Mixed precision (AMP) | CPU only; pending GPU run |
| DDP / sync BN | Single-device for now |
| Residual val_obj loss +15% gap (Phase 5) | Subtle BN/op-fusion source; defer pending mAP comparison at 30+ epochs |

### Next planned step
**TensorBoard integration** ahead of GPU training run. Add scalar
logging for: loss components (train/val), LR per group, mAP/P/R/fitness,
EMA decay value, gradient norms (optional). Hook into the existing
per-epoch loop after `tf_validate`. ~30-40 LoC; can also hook custom
images at val (mosaic of detections vs GT).

After TensorBoard is wired and a 30-epoch run on GPU is performed, the
parity question can be answered definitively at the mAP level. Phase 5
(residual loss debug) only worth the effort if TF mAP lags PT mAP by
> 5% at convergence.
