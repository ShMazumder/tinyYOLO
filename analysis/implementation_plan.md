# TinyYOLO — Implementation Plan

## Goal

Build a **modular, research-grade tinyYOLO framework** on top of Ultralytics that supports:
- **5 tasks**: Detection, Segmentation, Pose, Classification, OBB
- **2 architecture families**: Standard (FP32/FP16, with attention) and Quantized (INT8-safe, no attention)
- **5 input resolutions**: 160, 224, 320, 416, 640
- **Multiple deployment targets**: CPU, GPU, Mobile NPU, Edge TPU, Browser
- Full experiment tracking with reproducible configs

---

## User Decisions Summary

| Question | Decision |
|----------|----------|
| **Deploy Target** | All — architecture must be flexible enough for MCU→GPU |
| **Tasks** | All 5 — separate model files per task (tinyYOLO-det, -seg, -pose, -cls, -obb) |
| **Dataset** | Start with COCO (benchmark baseline), then extend to domain experiments |
| **Resolution** | All (160/224/320/416/640) — configurable, not hardcoded |
| **Framework** | Hybrid: Custom modules + Ultralytics training/export infrastructure |
| **Quantization** | Two families: Standard (with attention) + Quantized (INT8-safe) |

---

## Dataset Strategy (Recommendation)

> [!TIP]
> **Phase 1 — COCO128 (quick iteration):** 128 images, fast training loops for architecture validation
> **Phase 2 — COCO (full benchmark):** 118K train / 5K val — the gold standard for comparing against published YOLO results
> **Phase 3 — Domain experiments (optional):** VisDrone (aerial), DOTA (OBB), KITTI (driving), custom

COCO is ideal because:
- Direct comparison with every YOLO version's published mAP
- Teacher models (YOLO26s, YOLOv12s) are readily available for knowledge distillation
- Multi-scale objects test our architecture's full range
- 80 classes stress-test the classification head

---

## Proposed Project Structure

```
tinyYOLO/
├── analysis/                          # ✅ Already exists
│   └── YOLO_complete_analysis.md
│
├── tinyYOLO/                          # Python package (core)
│   ├── __init__.py
│   ├── modules/                       # Custom PyTorch modules
│   │   ├── __init__.py
│   │   ├── common.py                  # Shared: GhostConv, DWConv, SE, ECA, Concat
│   │   ├── backbone.py                # TinyBackbone (standard + quantized variants)
│   │   ├── neck.py                    # LitePAN, LiteFPN
│   │   └── heads.py                   # Detect, Segment, Pose, Classify, OBB heads
│   │
│   └── utils/                         # Utilities
│       ├── __init__.py
│       ├── registry.py                # Register modules with Ultralytics
│       ├── benchmark.py               # FLOPs, params, latency measurement
│       └── distill.py                 # Knowledge distillation helpers
│
├── configs/                           # YAML model definitions (Ultralytics format)
│   ├── standard/                      # Standard architecture (FP32/FP16, with attention)
│   │   ├── tinyYOLO-det.yaml
│   │   ├── tinyYOLO-seg.yaml
│   │   ├── tinyYOLO-pose.yaml
│   │   ├── tinyYOLO-cls.yaml
│   │   └── tinyYOLO-obb.yaml
│   │
│   └── quantized/                     # INT8-safe architecture (no attention, BN+ReLU only)
│       ├── tinyYOLO-det-q.yaml
│       ├── tinyYOLO-seg-q.yaml
│       ├── tinyYOLO-pose-q.yaml
│       ├── tinyYOLO-cls-q.yaml
│       └── tinyYOLO-obb-q.yaml
│
├── scripts/                           # Training & evaluation scripts
│   ├── train.py                       # Unified training entry point
│   ├── val.py                         # Validation & mAP evaluation
│   ├── export.py                      # Export to ONNX/TFLite/CoreML/TensorRT
│   ├── benchmark.py                   # Latency + FLOPs benchmarking
│   └── distill.py                     # Knowledge distillation training
│
├── experiments/                       # Experiment configs & results
│   ├── resolution_sweep.yaml          # Train across 160/224/320/416/640
│   ├── ablation_study.yaml            # Module-by-module ablation
│   └── results/                       # Auto-saved experiment outputs
│       └── .gitkeep
│
├── notebooks/                         # Jupyter notebooks for exploration
│   ├── 01_architecture_visualization.ipynb
│   ├── 02_det_experiments.ipynb
│   ├── 03_seg_experiments.ipynb
│   ├── 04_pose_experiments.ipynb
│   ├── 05_cls_experiments.ipynb
│   ├── 06_obb_experiments.ipynb
│   ├── 07_quantization_comparison.ipynb
│   └── 08_resolution_ablation.ipynb
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Proposed Changes

### Component 1: Core Modules

#### [NEW] [common.py](file:///Applications/XAMPP/xamppfiles/htdocs/tinyYOLO/tinyYOLO/modules/common.py)

Shared building blocks used by all variants:

| Module | Purpose | INT8-Safe? |
|--------|---------|-----------|
| `GhostConv` | Cheap feature maps via linear transforms (from GhostNet) | ✅ |
| `GhostBottleneck` | Ghost conv + depthwise separable + residual | ✅ |
| `DWConv` | Depthwise separable convolution | ✅ |
| `SEBlock` | Squeeze-and-Excitation channel attention | ✅ |
| `ECABlock` | Efficient Channel Attention (1D conv) | ✅ |
| `LightSpatialAttn` | Lightweight spatial attention (7×7 DWConv) | ❌ (Standard only) |
| `C3Ghost` | CSP bottleneck with Ghost convolutions | ✅ |
| `ConvBNReLU` | Standard Conv+BN+ReLU/SiLU block | ✅ |

---

#### [NEW] [backbone.py](file:///Applications/XAMPP/xamppfiles/htdocs/tinyYOLO/tinyYOLO/modules/backbone.py)

Two backbone variants:

**Standard Backbone** (~1.0M params):
```
Input → Stem(3→16, stride 2)
  → Stage1: GhostBottleneck(16→24, stride 2)
  → Stage2: GhostBottleneck(24→40, stride 2)    ← P3
  → Stage3: GhostBottleneck(40→80, stride 2) + LightSpatialAttn ← P4
  → Stage4: GhostBottleneck(80→160, stride 2) + SEBlock          ← P5
```

**Quantized Backbone** (~0.9M params):
```
Input → Stem(3→16, stride 2)
  → Stage1: GhostBottleneck(16→24, stride 2)
  → Stage2: GhostBottleneck(24→40, stride 2)    ← P3
  → Stage3: GhostBottleneck(40→80, stride 2) + ECABlock ← P4
  → Stage4: GhostBottleneck(80→160, stride 2) + ECABlock ← P5
```

Key differences:
- Standard uses `SiLU` activation + `LightSpatialAttn` → better accuracy
- Quantized uses `ReLU6` activation + `ECABlock` only → INT8-safe

---

#### [NEW] [neck.py](file:///Applications/XAMPP/xamppfiles/htdocs/tinyYOLO/tinyYOLO/modules/neck.py)

**LitePAN** — Lightweight PAN+FPN using depthwise separable convolutions:
```
P5 → Upsample+Concat(P4) → DWConv → F4
F4 → Upsample+Concat(P3) → DWConv → F3
F3 → DWConv(stride 2)+Concat(F4) → DWConv → N4
N4 → DWConv(stride 2)+Concat(P5) → DWConv → N5

Outputs: [F3, N4, N5]  (3 scales)
```

Estimated: ~0.2M params, ~0.3 GFLOPs at 320×320

---

#### [NEW] [heads.py](file:///Applications/XAMPP/xamppfiles/htdocs/tinyYOLO/tinyYOLO/modules/heads.py)

| Head | Output | Extra Params | Notes |
|------|--------|-------------|-------|
| `TinyDetect` | cls + bbox (no DFL, NMS-free) | ~0.15M | Decoupled, anchor-free, dual assignment |
| `TinySegment` | cls + bbox + proto-masks (32 protos) | ~0.25M | Extends TinyDetect + mask coefficients |
| `TinyPose` | cls + bbox + 17 keypoints | ~0.22M | COCO keypoint format |
| `TinyClassify` | class logits | ~0.05M | Global avg pool → FC |
| `TinyOBB` | cls + bbox + angle | ~0.17M | Oriented bounding box regression |

---

### Component 2: Model Configs (YAML)

#### [NEW] configs/standard/tinyYOLO-det.yaml

Ultralytics-compatible YAML defining the standard detection model. Example structure:

```yaml
# TinyYOLO Detection — Standard Architecture
nc: 80
scales:
  n: [0.50, 0.25, 256]    # depth, width, max_channels
  s: [0.75, 0.50, 512]

backbone:
  - [-1, 1, ConvBNAct, [16, 3, 2]]          # 0: stem
  - [-1, 1, GhostBottleneck, [24, 2]]        # 1: stage1
  - [-1, 2, GhostBottleneck, [40, 2]]        # 2: stage2 → P3
  - [-1, 3, GhostBottleneck, [80, 2]]        # 3: stage3 → P4
  - [-1, 1, LightSpatialAttn, [80]]          # 4: attention
  - [-1, 2, GhostBottleneck, [160, 2]]       # 5: stage4 → P5
  - [-1, 1, SEBlock, [160]]                  # 6: channel attention

neck:
  # ... LitePAN definition

head:
  - [TinyDetect, [nc]]
```

Similar YAMLs for seg, pose, cls, obb — each identical backbone/neck but different head.

#### [NEW] configs/quantized/tinyYOLO-det-q.yaml

Same structure but:
- `ReLU6` instead of `SiLU`
- `ECABlock` instead of `LightSpatialAttn`
- No `SEBlock` in critical paths
- Channel counts aligned to multiples of 8

---

### Component 3: Training Scripts

#### [NEW] [train.py](file:///Applications/XAMPP/xamppfiles/htdocs/tinyYOLO/scripts/train.py)

Unified entry point:
```python
# Usage examples:
# python scripts/train.py --config configs/standard/tinyYOLO-det.yaml --imgsz 320
# python scripts/train.py --config configs/quantized/tinyYOLO-seg-q.yaml --imgsz 416
# python scripts/train.py --config configs/standard/tinyYOLO-det.yaml --imgsz 160,224,320,416,640 --sweep
```

Features:
- Resolution as CLI argument (not hardcoded)
- `--sweep` mode for multi-resolution experiments
- Auto-logging to `experiments/results/`
- Knowledge distillation via `--teacher yolo26s.pt`

#### [NEW] [benchmark.py](file:///Applications/XAMPP/xamppfiles/htdocs/tinyYOLO/scripts/benchmark.py)

Measures for each model variant:
- Parameter count
- GFLOPs at each resolution
- Inference latency (CPU, GPU, TensorRT)
- Model file size (FP32, FP16, INT8)
- mAP@50-95

---

### Component 4: Experiment Notebooks

Each notebook is self-contained with:
1. Model loading & architecture visualization
2. Training on COCO128 (quick) or COCO (full)
3. Evaluation & comparison plots
4. Export & latency benchmarking

| Notebook | Purpose |
|----------|---------|
| `01_architecture_visualization` | Visualize all variants, count params/FLOPs, export diagrams |
| `02_det_experiments` | Train tinyYOLO-det (std + quantized) across resolutions |
| `03_seg_experiments` | Train tinyYOLO-seg variants |
| `04_pose_experiments` | Train tinyYOLO-pose variants |
| `05_cls_experiments` | Train tinyYOLO-cls variants |
| `06_obb_experiments` | Train tinyYOLO-obb variants |
| `07_quantization_comparison` | Compare FP32 vs FP16 vs INT8 across all tasks |
| `08_resolution_ablation` | Resolution sweep analysis with plots |

---

## Target Metrics (All at 320×320 on COCO val)

| Variant | Target Params | Target GFLOPs | Target mAP@50-95 | Baseline (YOLO26n@640) |
|---------|--------------|---------------|------------------|----------------------|
| tinyYOLO-det | ≤1.5M | ≤2.0 | ≥32% | 39.8% |
| tinyYOLO-det-q | ≤1.3M | ≤1.8 | ≥30% | 39.8% |
| tinyYOLO-seg | ≤1.8M | ≤2.5 | ≥28% (mask mAP) | — |
| tinyYOLO-pose | ≤1.7M | ≤2.3 | ≥45% (AP keypoint) | — |
| tinyYOLO-cls | ≤0.8M | ≤0.5 | ≥70% (top-1 ImageNet) | — |
| tinyYOLO-obb | ≤1.6M | ≤2.2 | ≥30% (DOTA mAP) | — |

> [!NOTE]
> At 320×320, we expect ~60-80% of the mAP compared to 640×640 input. These targets account for the resolution penalty.

---

## Experiment Matrix

The full experiment grid across all dimensions:

| Dimension | Values | Count |
|-----------|--------|-------|
| Task | det, seg, pose, cls, obb | 5 |
| Architecture | standard, quantized | 2 |
| Resolution | 160, 224, 320, 416, 640 | 5 |

**Total unique configs: 5 × 2 × 5 = 50 experiments**

> [!WARNING]
> Running all 50 on full COCO takes significant GPU time (~2–4 hrs each on a single GPU = 100–200 GPU-hours total). Recommended approach:
> 1. **Phase 1:** Validate all 50 on COCO128 (minutes each) → eliminate weak configs
> 2. **Phase 2:** Run top 15–20 configs on full COCO
> 3. **Phase 3:** Deep-dive ablation on top 5 configs

---

## Implementation Order

| Phase | Work | Files Created |
|-------|------|--------------|
| **Phase 1** | Project setup, dependencies, module registration | `setup.py`, `requirements.txt`, `__init__.py`, `registry.py` |
| **Phase 2** | Core modules (common → backbone → neck → heads) | `common.py`, `backbone.py`, `neck.py`, `heads.py` |
| **Phase 3** | YAML configs (all 10 model definitions) | `configs/standard/*.yaml`, `configs/quantized/*.yaml` |
| **Phase 4** | Training & benchmark scripts | `scripts/train.py`, `scripts/val.py`, `scripts/export.py`, `scripts/benchmark.py` |
| **Phase 5** | Notebooks (architecture viz → task experiments) | `notebooks/01-08_*.ipynb` |
| **Phase 6** | Run experiments, collect results, analyze | `experiments/results/*` |

---

## Verification Plan

### Automated Tests
1. **Module tests:** Instantiate each module, pass random tensor, verify output shape
2. **Config validation:** Load each YAML, build model, verify it compiles
3. **Forward pass:** Run inference on a single COCO image per variant
4. **Export test:** Export each model to ONNX, verify with `onnxruntime`
5. **Param/FLOP budget:** Assert each variant is within target budget

### Training Validation
1. **Smoke test:** Train 5 epochs on COCO128 for each config → verify loss decreases
2. **Benchmark:** Measure params, GFLOPs, latency for each variant at each resolution
3. **Full training:** Train on COCO, compare mAP against published YOLO nano/tiny results

---

## Open Questions

> [!IMPORTANT]
> **GPU availability:** Do you have access to a GPU for training? (COCO full training requires GPU — A100/V100/3090 recommended). If not, we can use Google Colab or focus on COCO128 experiments locally.

> [!IMPORTANT]
> **Experiment priority:** Would you like me to implement all phases sequentially (Phase 1→6), or focus on getting detection (tinyYOLO-det) working end-to-end first, then replicate for other tasks?
