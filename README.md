# TinyYOLO 🚀

**A modular, research-grade tiny object detection framework built on PyTorch + Ultralytics.**

Cherry-picks the best innovations from YOLOv1–v26 into ultra-lightweight models (0.07M–0.29M parameters) designed for edge deployment.

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
python scripts/export.py --weights experiments/results/tinyYOLO-det-std-320/model_init.pt --formats onnx
```

---

## Project Structure

```
tinyYOLO/
├── README.md                              # This file
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
│       └── registry.py                    # Register modules with Ultralytics
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
│   └── 08_resolution_ablation.py
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

Full training pipeline with COCO128 auto-download, real detection loss, AMP, EMA, and checkpoint saving.

**What happens when you run training:**
1. Auto-detects environment (Colab/Kaggle/RunPod/local) and configures batch size, workers, device
2. Downloads COCO128 dataset automatically (if not cached)
3. Builds model, loads images + YOLO-format labels
4. Trains with AdamW + cosine LR schedule + AMP (on GPU) + gradient clipping
5. Logs per-epoch losses: box, classification, objectness, total
6. Saves checkpoints: `best.pt`, `last.pt`, `ema.pt`
7. Plots training curves as PNG
8. Saves full config + history as JSON

**Output structure per experiment:**
```
experiments/results/tinyYOLO-det-std-320/
├── config.json          # Full experiment configuration
├── history.json         # Per-epoch loss + LR + timing
├── best.pt              # Best checkpoint (lowest total loss)
├── last.pt              # Final epoch checkpoint
├── ema.pt               # Exponential Moving Average checkpoint
└── training_curves.png  # Loss plots (box, cls, total)
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

**Expected training output:**
```
  Epoch        Box        Cls        Obj      Total         LR     Time
  ----------------------------------------------------------------
    1/100     0.0842     0.0156     0.6932     0.7924   0.001000    2.3s
    2/100     0.0791     0.0134     0.6891     0.7612   0.000998    2.1s
    3/100     0.0723     0.0118     0.6834     0.7298   0.000995    2.1s
    ...
```

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

---

## YAML Configs Reference

Each config defines: task, variant, backbone channels/depths, neck settings, head type, training hyperparameters, and augmentation.

### Standard Configs (`configs/standard/`)

| Config | Task | NC | Key Settings |
|--------|------|-----|-------------|
| `tinyYOLO-det.yaml` | Detection | 80 (COCO) | SiLU, spatial attention, 100 epochs |
| `tinyYOLO-seg.yaml` | Segmentation | 80 | 32 proto-masks, copy_paste=0.3 |
| `tinyYOLO-pose.yaml` | Pose | 1 (person) | 17 keypoints × 3 dims, 150 epochs |
| `tinyYOLO-cls.yaml` | Classification | 1000 (ImageNet) | dropout=0.2, label_smoothing=0.1 |
| `tinyYOLO-obb.yaml` | Oriented BBox | 15 (DOTA) | flipud=0.5, 150 epochs |

### Quantized Configs (`configs/quantized/`)

Same tasks but with: `variant: quantized`, `attention: eca`, QAT settings, `backend: qnnpack`.

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
| YOLOv10 | NMS-free design | Lower latency, cleaner deployment |
| YOLO11 | Spatial attention (C2PSA) | Focus on important regions |
| YOLO26 | No DFL, STAL | Simpler export + small object handling |

Full analysis: [`analysis/YOLO_complete_analysis.md`](analysis/YOLO_complete_analysis.md)

---

## License

Research use. See individual YOLO papers for their respective licenses.
