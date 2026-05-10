# TinyYOLO 🚀

**A modular, research-grade tiny object detection framework built on PyTorch + Ultralytics.**

Cherry-picks the best innovations from YOLOv1–v26 into ultra-lightweight models (0.07M–0.29M parameters) designed for edge deployment. Training pipeline uses CIoU loss, YOLO-standard BatchNorm, and comprehensive evaluation metrics (P/R/F1/mAP@50/mAP@50-95) with full report generation.

---

## Features

- **5 Tasks**: Detection, Segmentation, Pose Estimation, Classification, OBB
- **2 Architectures**: Standard (FP32/FP16, SiLU + spatial attention) and Quantized (INT8-safe, ReLU6 + ECA)
- **5 Input Resolutions**: 160, 224, 320, 416, 640 — all configurable per experiment
- **Auto-Environment**: Detects Colab / Kaggle / RunPod / Vast.ai / local and auto-tunes batch size, workers, device
- **Ghost-based Backbone**: Efficient feature extraction inspired by GhostNet
- **LitePAN Neck**: Depthwise separable FPN+PAN for multi-scale fusion
- **10 YAML Configs**: Ready-to-use configurations for every task × variant combination
- **8 Experiment Notebooks**: Architecture visualization, per-task experiments, quantization comparison, resolution ablation

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Model Variants](#model-variants)
5. [Architecture Overview](#architecture-overview)
6. [Scripts Reference](#scripts-reference)
7. [Notebooks Reference](#notebooks-reference)
8. [YAML Configs Reference](#yaml-configs-reference)
9. [Running All Experiments](#running-all-experiments)
10. [Environment Auto-Detection](#environment-auto-detection)
11. [Export & Deployment](#export--deployment)
12. [Research Background](#research-background)

---

## Installation

### Local (CPU/GPU)

```bash
# Clone and install
cd tinyYOLO
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Google Colab

```python
# In a Colab cell:
!git clone <your-repo-url> /content/tinyYOLO
%cd /content/tinyYOLO
!pip install -e . -q
```

### Kaggle

```python
# In a Kaggle notebook cell:
!git clone <your-repo-url> /kaggle/working/tinyYOLO
import sys; sys.path.insert(0, '/kaggle/working/tinyYOLO')
!pip install -e /kaggle/working/tinyYOLO -q
```

### RunPod / Vast.ai

```bash
cd /workspace
git clone <your-repo-url> tinyYOLO
cd tinyYOLO && pip install -e .
```

### Verify Installation

```bash
python -c "
import sys; sys.path.insert(0, '.')
from tinyYOLO.models import build_model
from tinyYOLO.utils.env import print_env_report
import torch

print_env_report()
model, info = build_model(task='det', variant='standard')
x = torch.randn(1, 3, 320, 320)
out = model(x)
print(f'Model: {info[\"total_params_M\"]}M params, output: {[o.shape for o in out]}')
print('Installation verified ✓')
"
```

---

## Quick Start

```bash
# 1. Benchmark all 10 model variants across all resolutions
python scripts/benchmark_models.py

# 2. Quick training test (5 epochs on COCO128 — auto-downloads dataset)
python scripts/train.py --task det --variant standard --imgsz 320 --quick

# 3. Full training on COCO128 (100 epochs, auto-detects GPU/batch/AMP)
python scripts/train.py --task det --variant standard --imgsz 320 --epochs 100

# 4. Resolution sweep (trains at each resolution sequentially)
python scripts/train.py --task det --variant standard --imgsz 160,224,320,416,640 --sweep

# 5. Export to ONNX
python scripts/export.py --weights experiments/results/tinyYOLO-det-std-320/best.pt --formats onnx
```

---

## Project Structure

```
tinyYOLO/
├── README.md                              # This file
├── report.md                              # Full architectural report with citations
├── requirements.txt                       # Python dependencies
├── setup.py                               # Package setup (pip install -e .)
│
├── analysis/                              # Research documentation
│   ├── YOLO_complete_analysis.md          # Full YOLO v1→v26 comparison
│   └── implementation_plan.md             # Architecture design document
│
├── tinyYOLO/                              # Core Python package
│   ├── __init__.py
│   ├── models.py                          # Model builder factory
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── common.py                      # Shared blocks: GhostConv, DWConv, SE, ECA, etc.
│   │   ├── backbone.py                    # TinyBackbone (Ghost-based, dual variant)
│   │   ├── neck.py                        # LitePAN (DW-separable FPN+PAN)
│   │   └── heads.py                       # 5 task heads: Det/Seg/Pose/Cls/OBB
│   └── utils/
│       ├── __init__.py
│       ├── env.py                         # Auto-detect training environment
│       ├── benchmark.py                   # Params/FLOPs/latency measurement
│       ├── registry.py                    # Register modules with Ultralytics
│       ├── postprocess.py                 # Decode predictions + NMS
│       └── metrics.py                     # P/R/F1/mAP/IoU/confusion matrix
│
├── configs/                               # Model configuration files
│   ├── standard/                          # SiLU activation + spatial attention
│   │   ├── tinyYOLO-det.yaml
│   │   ├── tinyYOLO-seg.yaml
│   │   ├── tinyYOLO-pose.yaml
│   │   ├── tinyYOLO-cls.yaml
│   │   └── tinyYOLO-obb.yaml
│   └── quantized/                         # ReLU6 activation + ECA only (INT8-safe)
│       ├── tinyYOLO-det-q.yaml
│       ├── tinyYOLO-seg-q.yaml
│       ├── tinyYOLO-pose-q.yaml
│       ├── tinyYOLO-cls-q.yaml
│       └── tinyYOLO-obb-q.yaml
│
├── scripts/                               # Command-line tools
│   ├── train.py                           # Unified training script
│   ├── export.py                          # Model export (ONNX/TorchScript)
│   └── benchmark_models.py               # Full benchmarking suite
│
├── notebooks/                             # Experiment scripts (# %% cell format)
│   ├── 01_architecture_visualization.py
│   ├── 02_det_experiments.py
│   ├── 03_seg_experiments.py
│   ├── 04_pose_experiments.py
│   ├── 05_cls_experiments.py
│   ├── 06_obb_experiments.py
│   ├── 07_quantization_comparison.py
│   ├── 08_resolution_ablation.py
│   ├── 09_metrics_report.py              # Post-training metrics & 12 visualizations
│   └── 10_full_evaluation.py             # Complete evaluation pipeline (all steps)
│
└── experiments/                           # Auto-generated results
    └── results/                           # Per-experiment outputs (JSON, PNG, PT)
```

---

## Model Variants

### Validated Models (all forward-pass tested)

| Model | Task | Std Params | Q Params | Output (@320) |
|-------|------|-----------|---------|--------------|
| **tinyYOLO-det** | Object Detection | 0.23M | 0.22M | 3 scales: `[84,H,W]` |
| **tinyYOLO-seg** | Instance Segmentation | 0.29M | 0.28M | 3 scales + proto `[32,160,160]` |
| **tinyYOLO-pose** | Pose Estimation | 0.24M | 0.23M | 3 scales + keypoints `[51,H,W]` |
| **tinyYOLO-cls** | Classification | 0.24M | 0.22M | logits `[1000]` |
| **tinyYOLO-obb** | Oriented BBox | 0.25M | 0.23M | 3 scales: `[85,H,W]` (incl. angle) |

### Standard vs Quantized

| Feature | Standard | Quantized |
|---------|----------|-----------|
| Activation | SiLU | ReLU6 |
| Channel Attention | SEBlock | ECABlock |
| Spatial Attention | LightSpatialAttn | None |
| INT8 Quantization | Not optimized | Designed for QAT |
| Best for | Max accuracy (GPU) | Edge deployment (INT8) |

---

## Architecture Overview

```
Input (160/224/320/416/640)
  │
  ├─ Backbone: Ghost-based (~0.08M params)
  │   ├─ Stem: Conv 3→16, stride 2
  │   ├─ Stage1: GhostBottleneck 16→24, stride 2    (/4)
  │   ├─ Stage2: GhostBottleneck 24→40, stride 2    (/8)  → P3
  │   ├─ Stage3: GhostBottleneck 40→80, stride 2    (/16) → P4 + Attention
  │   └─ Stage4: GhostBottleneck 80→160, stride 2   (/32) → P5 + Attention
  │
  ├─ Neck: LitePAN (~0.06M params)
  │   ├─ Top-down: P5→P4→P3 (FPN with DWConv)
  │   └─ Bottom-up: P3→P4→P5 (PAN with DWConv)
  │
  └─ Head: Task-specific (~0.05–0.15M params)
      ├─ det:  Decoupled cls + bbox (no DFL, NMS-free capable)
      ├─ seg:  det + proto-mask branch (32 prototypes)
      ├─ pose: det + 17-keypoint regression (×3 dims)
      ├─ cls:  Global avg pool → dropout → FC
      └─ obb:  det + angle regression
```

---

## Scripts Reference

### `scripts/train.py` — Training

Full training pipeline with COCO128 auto-download, CIoU loss, AMP, EMA, per-epoch metrics, and full report generation.

**What happens when you run training:**
1. Auto-detects environment (Colab/Kaggle/RunPod/local) and configures batch size, workers, device
2. Downloads COCO128 dataset automatically (if not cached)
3. Builds model, applies YOLO-standard BatchNorm (`eps=1e-3, momentum=0.03`)
4. Trains with AdamW (separate weight decay groups) + cosine LR + AMP + gradient clipping
5. Uses **CIoU box loss** + BCE classification + BCE objectness (weighted 2.0 / 1.0 / 1.0)
6. Computes **P/R/F1/mAP@50/mAP@50-95** at regular intervals via NMS + IoU matching
7. Saves best checkpoint by **mAP@50** (not just loss)
8. Generates full report: training curves, confusion matrix, per-class breakdown
9. Saves config, history, metrics as JSON

**Output structure per experiment:**
```
experiments/results/tinyYOLO-det-std-320/
├── config.json            # Full config: model, optimizer, scheduler, loss, augmentation, final metrics
├── history.json           # Per-epoch: losses, P, R, F1, mAP50, mAP50-95, LR, timing
├── metrics.json           # Final evaluation: all metrics + per-class + confusion matrix
├── per_class_report.txt   # Human-readable per-class P/R/F1/AP table
├── training_curves.png    # Loss + accuracy curves (2×3 grid)
├── confusion_matrix.png   # Confusion matrix heatmap
├── best.pt                # Best checkpoint (by mAP@50)
├── last.pt                # Final epoch checkpoint
└── ema.pt                 # Exponential Moving Average checkpoint
```

```bash
# Basic training (auto-detects everything)
python scripts/train.py --task det --variant standard --imgsz 320

# Quick smoke test (5 epochs)
python scripts/train.py --task det --variant standard --imgsz 320 --quick

# All CLI options:
python scripts/train.py \
  --task det           # det | seg | pose | cls | obb
  --variant standard   # standard | quantized
  --imgsz 320          # Single value or comma-separated for sweep
  --epochs 100         # Training epochs (default: 100)
  --batch 32           # Batch size (auto-detected if omitted)
  --device cuda:0      # Device (auto-detected if omitted)
  --lr 0.001           # Learning rate (default: 1e-3)
  --data coco128.yaml  # Dataset (auto-selected per task if omitted)
  --name my_exp        # Custom experiment name
  --quick              # Quick test: 5 epochs only
  --sweep              # Run at each resolution in --imgsz
```

**Train every task:**

```bash
# Detection (standard + quantized)
python scripts/train.py --task det --variant standard --imgsz 320
python scripts/train.py --task det --variant quantized --imgsz 320

# Segmentation
python scripts/train.py --task seg --variant standard --imgsz 320
python scripts/train.py --task seg --variant quantized --imgsz 320

# Pose estimation
python scripts/train.py --task pose --variant standard --imgsz 320
python scripts/train.py --task pose --variant quantized --imgsz 320

# Classification
python scripts/train.py --task cls --variant standard --imgsz 224
python scripts/train.py --task cls --variant quantized --imgsz 224

# Oriented bounding box
python scripts/train.py --task obb --variant standard --imgsz 416
python scripts/train.py --task obb --variant quantized --imgsz 416
```

**Train all 10 variants in one go:**

```bash
for task in det seg pose obb; do
  for variant in standard quantized; do
    python scripts/train.py --task $task --variant $variant --imgsz 320 --epochs 100
  done
done

# Classification uses 224
for variant in standard quantized; do
  python scripts/train.py --task cls --variant $variant --imgsz 224 --epochs 100
done
```

**Resolution sweeps:**

```bash
# Sweep all resolutions for detection
python scripts/train.py --task det --variant standard --imgsz 160,224,320,416,640 --sweep

# Sweep for segmentation
python scripts/train.py --task seg --variant standard --imgsz 224,320,416 --sweep
```

**Expected training output (actual results from COCO128, Tesla T4):**
```
  Epoch      Box      Cls      Obj    Total      P      R     F1   mAP50         LR   Time
  ----------------------------------------------------------------------------------------
    1/100   0.1130   0.0677   0.3800   0.9788  0.000  0.000  0.000  0.0000   0.001000  13.8s
   20/100   0.0278   0.0641   0.3041   0.4752  0.016  0.023  0.019  0.0314   0.000905   7.3s
   60/100   0.0090   0.0598   0.2195   0.2942  0.028  0.045  0.035  0.0149   0.000352   7.5s
  100/100   0.0070   0.0608   0.1913   0.2569  0.027  0.040  0.032  0.0129   0.000010   7.9s

  Running final evaluation...
  ============================================================
    Detection Metrics Report
  ============================================================
    Predictions:   1378
    Ground Truths: 929
    TP: 37  |  FP: 1341  |  FN: 892
    ────────────────────────────────────────────────────────────
    Precision:     0.0269
    Recall:        0.0398
    F1 Score:      0.0321
    mAP@50:        0.0129
    mAP@50-95:     0.0035
  ============================================================

  Results saved to: experiments/results/tinyYOLO-det-std-320
  Outputs:
    best.pt, last.pt, ema.pt      — Model checkpoints
    config.json                    — Hyperparameters & augmentation report
    history.json                   — Per-epoch losses + metrics
    metrics.json                   — Final P/R/F1/mAP/confusion matrix
    training_curves.png            — Loss + accuracy curves
    confusion_matrix.png           — Confusion matrix heatmap
    per_class_report.txt           — Per-class P/R/F1/AP breakdown
```

> **Note:** Metrics are deliberately low because tinyYOLO uses only **0.23M parameters**
> (vs YOLOv8n's 3.2M) on only 128 training images. This framework is designed for
> edge deployment where model size < 1M is the priority. For higher accuracy, use
> a larger dataset or increase model capacity.

### `scripts/benchmark_models.py` — Benchmarking

```bash
# Benchmark everything (all tasks × all variants × all resolutions)
python scripts/benchmark_models.py

# Benchmark specific tasks/variants
python scripts/benchmark_models.py --tasks det,seg --variants standard --imgsz 320,640

# All options:
python scripts/benchmark_models.py \
  --tasks det,seg,pose,cls,obb    # Tasks to benchmark
  --variants standard,quantized   # Variants
  --imgsz 160,224,320,416,640     # Resolutions
  --device cpu                    # Device for latency measurement
  --output results.json           # Output path
```

### `scripts/export.py` — Export

```bash
# Export to ONNX
python scripts/export.py \
  --weights path/to/model.pt \
  --task det \
  --variant standard \
  --imgsz 320 \
  --formats onnx,torchscript \
  --fp16          # Optional: FP16 export
  --int8          # Optional: INT8 quantization
```

---

## Notebooks Reference

All notebooks use `# %%` cell markers — compatible with VS Code interactive Python, Jupyter, and Colab.

**To run as a Jupyter notebook:**
```bash
cd notebooks
jupyter notebook  # Then open any .py file — VS Code/Jupyter auto-detects cells
```

**To run as a script:**
```bash
cd notebooks
python 01_architecture_visualization.py
```

| # | Notebook | What It Does | Recommended GPU? |
|---|----------|-------------|-----------------|
| 01 | `01_architecture_visualization.py` | Visualize all 10 models, count params/FLOPs, compare with YOLO baselines, plot parameter distributions | ❌ CPU OK |
| 02 | `02_det_experiments.py` | Detection: forward pass, latency benchmark, gradient flow test, training smoke test, ONNX export | ❌ CPU OK |
| 03 | `03_seg_experiments.py` | Segmentation: validate proto-masks, latency, gradient flow | ❌ CPU OK |
| 04 | `04_pose_experiments.py` | Pose: verify 17-keypoint output shapes, latency | ❌ CPU OK |
| 05 | `05_cls_experiments.py` | Classification: verify logits, model size on disk | ❌ CPU OK |
| 06 | `06_obb_experiments.py` | OBB: verify angle output, DOTA class reference | ❌ CPU OK |
| 07 | `07_quantization_comparison.py` | Compare std vs quantized: params, latency, quantization readiness, disk size | ❌ CPU OK |
| 08 | `08_resolution_ablation.py` | Resolution sweep: latency/FLOPs/grid analysis across 160–640, cross-task comparison | ❌ CPU OK |
| 09 | `09_metrics_report.py` | **Full metrics report**: hyperparams, augmentation, P/R/F1, mAP, confusion matrix, accuracy curves, params vs accuracy, cross-experiment comparison | ❌ CPU OK |
| 10 | `10_full_evaluation.py` | **Complete pipeline**: train std+quantized, resolution ablation, metrics, ONNX export, cross-experiment plots | ✅ GPU recommended |

---

## Metrics & Reports

### Metrics Computed Automatically During Training

The training script (`scripts/train.py`) computes all of the following during and after training:

| Metric | Description | Computed When |
|--------|-------------|--------------|
| **Box Loss** | CIoU (Complete IoU — includes center distance + aspect ratio) | Every epoch |
| **Cls Loss** | Classification loss (BCE) | Every epoch |
| **Obj Loss** | Objectness loss (BCE) | Every epoch |
| **Total Loss** | `2.0×CIoU + 1.0×Cls + 1.0×Obj` | Every epoch |
| **Precision** | TP / (TP + FP) at IoU=0.5 | Every eval epoch |
| **Recall** | TP / (TP + FN) at IoU=0.5 | Every eval epoch |
| **F1 Score** | 2·P·R / (P + R) | Every eval epoch |
| **mAP@50** | Mean Average Precision at IoU=0.5 | Every eval epoch |
| **mAP@50-95** | Mean AP averaged over IoU 0.5:0.95:0.05 | Every eval epoch |
| **Per-class AP** | AP for each class individually | Final evaluation |
| **Confusion Matrix** | TP/FP counts per class | Final evaluation |

### Training Configuration (YOLO-Standard)

| Component | Setting | Reference |
|-----------|---------|----------|
| **Loss** | CIoU + BCE (weighted 2.0 / 1.0 / 1.0) | Adapted from [pfeatherstone/tinyyolo](https://github.com/pfeatherstone/tinyyolo) |
| **BatchNorm** | `eps=1e-3, momentum=0.03` | All official YOLO models |
| **Optimizer** | AdamW, separate weight decay groups | Weights: 1e-4, biases/BN: 0.0 |
| **Scheduler** | Cosine annealing (η_min = lr × 0.01) | Standard YOLO recipe |
| **Augmentation** | ColorJitter(0.4), Grayscale(0.1), HFlip(0.5), Perspective(0.2) | Enhanced pipeline |

### Auto-Generated Report Files

After training completes, the following files are saved to `experiments/results/<experiment>/`:

| File | Contents |
|------|----------|
| `config.json` | Hyperparameter config (model, optimizer, scheduler, augmentation, final metrics) |
| `history.json` | Per-epoch: losses, P, R, F1, mAP50, mAP50-95, LR, timing |
| `metrics.json` | Final evaluation: all metrics + per-class breakdown + confusion matrix |
| `per_class_report.txt` | Human-readable per-class P/R/F1/AP table |
| `training_curves.png` | Loss curves (box, cls, total) + accuracy curves (P, R, F1, mAP) |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `best.pt` | Best model checkpoint (by mAP@50) |
| `last.pt` | Final epoch checkpoint |
| `ema.pt` | Exponential Moving Average checkpoint |

### Comprehensive Metrics Notebook

For interactive post-training analysis, run:

```bash
cd notebooks
python 09_metrics_report.py
```

This notebook generates **12 visualizations**:

1. **Hyperparameter Configuration Report** — full model/optimizer/scheduler config
2. **Augmentation Report** — all data augmentation settings
3. **Classification Report** — P, R, F1, TP, FP, FN
4. **Per-Class Breakdown** — per-class P/R/F1/AP table
5. **Loss Curves** — box, cls, obj, total vs epoch
6. **Accuracy Curves** — P, R, F1, mAP@50, mAP@50-95 vs epoch
7. **Epoch vs Accuracy** — combined loss + metrics dual-axis plot
8. **Parameter Size vs Accuracy** — Pareto front with YOLO baseline references
9. **Confusion Matrix** — heatmap visualization
10. **IoU Distribution** — AP at different IoU thresholds
11. **Learning Rate Schedule** — cosine annealing curve
12. **Cross-Experiment Comparison** — table comparing all trained variants



## YAML Configs Reference

Each config defines: task, variant, backbone, neck, head, **loss** (CIoU+BCE), **batchnorm** (YOLO-standard), training hyperparameters, and augmentation.

### Standard Configs (`configs/standard/`)

| Config | Task | NC | Key Settings |
|--------|------|-----|-------------|
| `tinyYOLO-det.yaml` | Detection | 80 (COCO) | CIoU loss, SiLU, spatial+SE attention, 100 epochs |
| `tinyYOLO-seg.yaml` | Segmentation | 80 | CIoU loss, 32 proto-masks, copy_paste=0.3 |
| `tinyYOLO-pose.yaml` | Pose | 1 (person) | CIoU loss, 17 keypoints × 3 dims, 150 epochs |
| `tinyYOLO-cls.yaml` | Classification | 1000 (ImageNet) | CrossEntropy, dropout=0.2, label_smoothing=0.1 |
| `tinyYOLO-obb.yaml` | Oriented BBox | 15 (DOTA) | CIoU loss, flipud=0.5, 150 epochs |

### Quantized Configs (`configs/quantized/`)

Same tasks but with: `variant: quantized`, `attention: eca`, QAT settings, `backend: qnnpack`.

### Common Settings (All Configs v2)

| Setting | Value | Notes |
|---------|-------|-------|
| Loss | CIoU+BCE (2.0/1.0/1.0) | Except cls (CrossEntropy) |
| BatchNorm | eps=0.001, momentum=0.03 | YOLO-standard |
| Weight Decay | 0.0001 (weights only) | Biases/BN excluded |
| Augmentation | +grayscale(0.1), +perspective(0.2) | Enhanced pipeline |

---

## Running All Experiments

### Phase 1: Quick Validation (CPU, ~5 minutes)

```bash
# Verify all models build and forward correctly
cd notebooks
python 01_architecture_visualization.py
```

### Phase 2: Per-Task Validation (CPU, ~10 minutes each)

```bash
cd notebooks

# Run each task experiment
python 02_det_experiments.py
python 03_seg_experiments.py
python 04_pose_experiments.py
python 05_cls_experiments.py
python 06_obb_experiments.py
```

### Phase 3: Comparison Studies (CPU, ~15 minutes each)

```bash
cd notebooks
python 07_quantization_comparison.py    # Standard vs Quantized
python 08_resolution_ablation.py        # Resolution sweep analysis
```

### Phase 4: Full Benchmark (CPU/GPU, ~30 minutes)

```bash
# Benchmark all 50 combinations (5 tasks × 2 variants × 5 resolutions)
python scripts/benchmark_models.py

# Results saved to: experiments/results/benchmark_<timestamp>.json
```

### Phase 5: Full Training (GPU required)

```bash
# Quick smoke test (5 epochs, COCO128)
python scripts/train.py --task det --variant standard --imgsz 320 --quick

# Full COCO training — detection
python scripts/train.py --task det --variant standard --imgsz 320 --epochs 100

# Full sweep — detection across all resolutions
python scripts/train.py --task det --variant standard --imgsz 160,224,320,416,640 --sweep --epochs 100

# All tasks — standard variants
for task in det seg pose cls obb; do
  python scripts/train.py --task $task --variant standard --imgsz 320 --epochs 100
done

# All tasks — quantized variants
for task in det seg pose cls obb; do
  python scripts/train.py --task $task --variant quantized --imgsz 320 --epochs 100
done
```

### Phase 6: Export (after training)

```bash
# Export best detection model
python scripts/export.py --weights experiments/results/tinyYOLO-det-std-320/model_init.pt \
  --task det --variant standard --imgsz 320 --formats onnx,torchscript

# Export quantized model
python scripts/export.py --weights experiments/results/tinyYOLO-det-q-320/model_init.pt \
  --task det --variant quantized --imgsz 320 --formats onnx --int8
```

---

## Environment Auto-Detection

The framework automatically detects your training environment and configures optimal settings:

```python
from tinyYOLO.utils.env import detect_environment, print_env_report
env = detect_environment()
print_env_report(env)
```

| Platform | Detection Method | Auto-Configured |
|----------|-----------------|-----------------|
| **Google Colab** | `COLAB_GPU` env var | data_dir=/content/datasets, cache=True |
| **Kaggle** | `KAGGLE_KERNEL_RUN_TYPE` env var | data_dir=/kaggle/working/datasets, workers=4 |
| **RunPod** | `RUNPOD_POD_ID` env var | data_dir=/workspace/datasets, workers=8 |
| **Vast.ai** | `VAST_CONTAINERLABEL` env var | data_dir=/workspace/datasets, workers=8 |
| **Local** | Fallback | data_dir=./datasets, workers=auto |

**GPU-based batch size recommendations:**

| GPU Memory | Recommended Batch Size |
|-----------|----------------------|
| ≥40 GB (A100) | 128 |
| ≥20 GB (3090/4090) | 64 |
| ≥10 GB (3080) | 32 |
| ≥6 GB (3060) | 16 |
| CPU only | 4 |

---

## Export & Deployment

| Format | Command | Use Case |
|--------|---------|----------|
| **ONNX** | `--formats onnx` | Cross-platform, ONNX Runtime, OpenVINO |
| **TorchScript** | `--formats torchscript` | PyTorch mobile, C++ deployment |
| **TFLite** | Use Ultralytics export | Android, Edge TPU, Coral |
| **CoreML** | Use Ultralytics export | iOS, macOS |
| **TensorRT** | Use Ultralytics export | NVIDIA GPU inference |

---

## Research Background

This project is built on a comprehensive analysis of all YOLO versions (v1→v26). Key techniques cherry-picked:

| From | Technique | Why |
|------|-----------|-----|
| GhostNet | Ghost Convolutions | Cheap feature generation (half the FLOPs) |
| YOLOv4 | Mosaic augmentation | Free accuracy boost during training |
| YOLOX | Decoupled head | Better cls/reg separation |
| YOLOv8 | Anchor-free detection | Simpler, fewer hyperparameters |
| YOLOv8/pfeatherstone | CIoU loss + YOLO BatchNorm | Better box regression + stable training |
| YOLOv10 | NMS-free design | Lower latency, cleaner deployment |
| YOLO11 | Spatial attention (C2PSA) | Focus on important regions |
| YOLO26 | No DFL, STAL | Simpler export + small object handling |

Full analysis: [`analysis/YOLO_complete_analysis.md`](analysis/YOLO_complete_analysis.md)

---

## License

Research use. See individual YOLO papers for their respective licenses.
