# TinyYOLO 🚀

**A modular, research-grade tiny object detection framework built on PyTorch + Ultralytics.**

## Features

- **5 Tasks**: Detection, Segmentation, Pose, Classification, OBB
- **2 Architectures**: Standard (FP32/FP16) and Quantized (INT8-safe)
- **5 Resolutions**: 160, 224, 320, 416, 640 — all configurable
- **Auto-Environment**: Detects Colab/Kaggle/RunPod/local and optimizes settings
- **Ghost-based Backbone**: Efficient feature extraction with <1.5M parameters

## Quick Start

```bash
# Install
pip install -e .

# Benchmark all models
python scripts/benchmark_models.py

# Train detection model
python scripts/train.py --task det --variant standard --imgsz 320

# Train with resolution sweep
python scripts/train.py --task det --variant standard --imgsz 160,224,320,416,640 --sweep

# Export to ONNX
python scripts/export.py --weights path/to/model.pt --formats onnx
```

## Project Structure

```
tinyYOLO/
├── analysis/          # YOLO version analysis & implementation plan
├── tinyYOLO/          # Core Python package
│   ├── modules/       # Backbone, Neck, Heads, Common blocks
│   ├── utils/         # Environment detection, benchmarking
│   └── models.py      # Model builder
├── configs/           # YAML model configs (standard + quantized)
├── scripts/           # Training, export, benchmark scripts
├── notebooks/         # Experiment notebooks
└── experiments/       # Results and logs
```

## Model Variants

| Model | Task | Params | Target GFLOPs (@320) |
|-------|------|--------|---------------------|
| tinyYOLO-det | Detection | ≤1.5M | ≤2.0 |
| tinyYOLO-seg | Segmentation | ≤1.8M | ≤2.5 |
| tinyYOLO-pose | Pose | ≤1.7M | ≤2.3 |
| tinyYOLO-cls | Classification | ≤0.8M | ≤0.5 |
| tinyYOLO-obb | Oriented BBox | ≤1.6M | ≤2.2 |
