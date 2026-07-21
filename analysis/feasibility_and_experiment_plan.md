# TinyYOLO — Feasibility Assessment & Experiment Plan

**Date:** 2026-07-22
**Context:** Written after the R1.4 box-decode fix. The prior real training run (`voc-q-320-seed42`) stalled at mAP@50 ≈ 0.0011 because the box head was decoded with `σ(pred) × imgsz` (no grid-cell anchoring). That is now fixed with a shared grid-anchored codec (`tinyYOLO/utils/boxcodec.py`). This document asks two questions: (1) is the core idea actually feasible, and (2) what experiments would establish it honestly.

---

## Part 1 — Is the core idea feasible?

### 1.1 What the core idea actually is

Four coupled claims:

1. A **sub-0.25M-parameter** anchor-free object detector that *learns usefully*.
2. A **dual-variant** design (SiLU+SE standard / ReLU6+ECA quantized) that is **INT8-native** with small accuracy loss.
3. **Multi-task** extensibility (det/seg/pose/cls/obb) over one ~0.08M shared Ghost backbone.
4. **Edge-deployable** latency on Jetson Nano / Raspberry Pi 4.

Each is assessed separately, because they have very different risk.

### 1.2 Claim-by-claim verdict

| Claim | Feasible? | Basis / caveat |
|---|---|---|
| 1. 0.22–0.23M detector that learns | **Yes** | YOLO-Fastest (~0.25–0.35M) and MCUNetV2 (0.74M) already learn usable detectors at this scale. With the decode fixed, TinyYOLO should move well off the ≈0 floor. Expect *modest* accuracy, not SOTA. |
| 2. INT8-native, small drop | **Yes, with realistic drop** | ReLU6 + ECA is a proven INT8-friendly recipe (MobileNetV2, ECA-Net). A **1–3%** mAP drop under QAT is realistic at this scale; the previously reported **0.7%** is optimistic and unverified. |
| 3. Multi-task over 0.08M backbone | **Runs; will be weak** | Forward passes and shapes are valid. But a 0.08M shared backbone is a hard capacity bottleneck; seg/pose/obb heads will be data-starved. Feasible to *demonstrate*, not to be competitive per-task. |
| 4. Edge latency | **Yes, but must be measured** | Sub-0.3 GFLOP models will run in real time on Jetson/RPi. The specific ms/FPS/power numbers must be **instrumented on real hardware**, not estimated. |

### 1.3 Honest accuracy expectations (0.22–0.23M, anchor-free)

These are the *plausible ranges* to target — set as hypotheses, not promises:

| Metric | Realistic range | Notes |
|---|---|---|
| VOC mAP@50 (COCO 101-pt) | **30–45%** | Depends heavily on resolution (416 > 320) and assignment (TAL helps). |
| COCO mAP@50-95 | **6–11%** | AP_S will floor around **1.5–3%** — small objects are the fundamental ceiling. |
| INT8 QAT drop | **1–3%** | Quantized variant should retain more than the standard variant. |
| Jetson Nano INT8 latency | **~25–40 ms** @416 | Instrument, don't estimate. |

The paper's headline numbers (VOC 41.2%, COCO 19.7% mAP@50, 0.7% INT8 drop, all as 5-seed mean±std) are **at the optimistic edge or beyond**, and — critically — **not backed by any artifact on disk**. Feasibility of the *idea* does not rescue those specific numbers; they must be earned by real runs.

### 1.4 Principal risks

1. **Capacity ceiling.** 0.22M params is a genuine wall. The idea is only competitive inside the *sub-0.3M niche* (vs YOLO-Fastest), not against NanoDet/PicoDet (~1M) or YOLOv8n (3.2M). Frame the contribution as niche + INT8 + multi-task, never as general SOTA.
2. **Assignment is currently naive.** `TALAssigner` is defined but **never called** — `DetectionLoss.forward` uses single-cell assignment. The paper credits TAL with "+7.8% mAP"; that gain is presently unrealized. Wiring TAL is the single highest-value next code change.
3. **Small-object AP.** Structural; expect ~2%. Don't over-promise AP_S.
4. **Reproducibility / integrity.** Every result table must be regenerated from actual runs with saved `config.json` + `metrics.json`. No number enters the manuscript without a matching artifact.
5. **Multi-scale assignment.** Each GT is currently assigned to its cell at *all three* scales identically. Scale-aware assignment (size→level) is a likely accuracy lever.

### 1.5 Overall verdict

**The core idea is feasible and worth pursuing — as a niche, INT8-first, multi-task tiny detector — with honest, modest accuracy targets.** It is *not* feasible as literally claimed in the current manuscript (those numbers are unverified and near/above the realistic ceiling). The decode fix removes the blocker that made the detector untrainable; the next gates are (a) a sanity overfit, (b) wiring TAL, (c) real multi-seed VOC/COCO runs.

---

## Part 2 — Experiment plan

Design principles: **gated stages** (cheap sanity checks before expensive runs), **every run writes an artifact**, **mean±std over seeds for any headline number**, **ablate one variable at a time against a fixed baseline**.

Baseline for all ablations unless noted: **TinyYOLO-quantized, VOC 2007+2012, 416×416, 100 epochs, 3 seeds {42, 123, 256}.**

### Stage 0 — Sanity (CPU/GPU, minutes) — GATE

| Test | Command / method | Pass criterion |
|---|---|---|
| S0.1 Overfit one batch | Train on a single batch of ~4 images, 300 steps | Total loss → near 0; predicted boxes visually match GTs |
| S0.2 Gradient flow | Backprop once, check grad norms per module | All modules receive non-zero, non-NaN grads |
| S0.3 Decode round-trip | Encode GT → `decode_boxes` → compare | Recovered cx,cy,w,h ≈ GT within tolerance |

**Gate:** if S0.1 does not overfit, stop — the localization path is still broken. This is the direct test that the R1.4 fix works.

### Stage 1 — COCO128 smoke (GPU, ~30 min) — GATE

- 100 epochs, quantized, 320×320.
- **Pass:** mAP@50 clearly above the broken-decode result and boxes localized (not sprayed). Confirms the pipeline end-to-end before spending GPU-hours.

### Stage 2 — VOC baseline (Tesla T4, ~2–3 h/run)

- Variants: standard + quantized.
- Resolutions: 320 and 416.
- Seeds: {42, 123, 256, 512, 1024} (5).
- Epochs: 200–300, cosine, 3-epoch warmup, mosaic (disable last 10%).
- Deliverable: **replaces the fabricated VOC table** with real mean±std. Report under both COCO-101pt and VOC-11pt interpolation.

### Stage 3 — COCO val2017 / full COCO (GPU, longer)

- standard + quantized, 416×416, 300 epochs, ≥3 seeds.
- Report mAP@50, mAP@50-95, **AP_S/AP_M/AP_L** (currently entirely unbacked).
- If full COCO (118K) is too costly, state clearly that results are on the reduced split.

### Stage 4 — Ablations (each vs Stage-2 VOC-q baseline, 100 ep, ≥3 seeds)

| # | Ablation | Variable | Why it matters |
|---|---|---|---|
| A1 | **Decode** | grid-anchored (new) vs `σ×imgsz` (old) | Quantifies the R1.4 fix; expected to be the single largest jump |
| A2 | **Assignment** | single-cell vs **TAL (k=10)** | Requires wiring `TALAssigner`; paper's claimed +7.8% |
| A3 | Attention | ECA vs SE vs none | Justifies the quantized-variant choice |
| A4 | Ghost vs standard conv | conv type | Params/accuracy trade |
| A5 | Activation | ReLU6 vs SiLU (+ measure INT8 drop) | Core dual-variant justification |
| A6 | Resolution | 224 / 320 / 416 / 640 | Accuracy–latency knee |
| A7 | Mosaic | on vs off | Augmentation value at tiny scale |
| A8 | Objectness | dedicated head vs max-class proxy | Already implemented; measure it |
| A9 | Loss weights | box 2.0 vs 5.0 vs 7.5 | Tiny-model tuning |
| A10 | Width multiplier | 0.5× / 1.0× / 1.5× | Capacity sweep |
| A11 | Scale assignment | all-levels vs size-aware | Likely FPN lever |

### Stage 5 — Quantization

- PTQ (MinMax, 500-image calib) vs QAT (10 ep, lr 1e-4).
- Backends: qnnpack (ARM) and fbgemm (x86).
- Report FP32 → INT8 mAP delta and on-disk model size for both variants.

### Stage 6 — Edge deployment (real hardware) — must be instrumented

- Jetson Nano (TensorRT) + Raspberry Pi 4 (TFLite), plus Tesla T4 reference.
- Median latency over ≥100 runs, batch=1, at FP32/FP16/INT8.
- **Power measured with a meter**, not estimated (current manuscript estimates it — flag or drop).

### Stage 7 — Multi-task (only claim what is run)

| Task | Dataset | Metric | Status to reach |
|---|---|---|---|
| Segmentation | COCO | Box mAP + Mask mAP | run before claiming |
| Pose | COCO person | Box mAP + Keypoint AP | run before claiming |
| Classification | ImageNet-1k | Top-1 | currently "pending" in §12 |
| OBB | DOTA | mAP | currently "pending" in §12 |

Reconcile with the manuscript: README claims all five tasks "validated"; report §12 says Cls/OBB pending. Pick one truth and make the tables match the artifacts.

### Domain-specific fast track (if scope must shrink)

If the target is one concrete edge use-case (e.g. **person detection on Raspberry Pi 4**), run only: Stage 0 → Stage 1 → Stage 2 (VOC person subset or a domain set) → A1, A2, A5, A6 → Stage 5 → Stage 6. This yields a defensible single-domain result in a fraction of the compute, and is the recommended path if full multi-dataset benchmarking is not resourced.

---

## Integrity checklist (before anything goes in the paper)

- [ ] Every table row has a matching `experiments/results/<name>/metrics.json`.
- [ ] All checkpoints trained with the R1.4 decode (old `best.pt`/`ema.pt` are invalid).
- [ ] Seeds, resolution, epochs in the text match the `config.json` on disk.
- [ ] Edge latency/power come from real hardware logs.
- [ ] Claims about TAL / mosaic / objectness are backed by the A2/A7/A8 ablations, not asserted.
