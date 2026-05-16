# TinyYOLO R1 — GPU Experiment Execution Guide

> Step-by-step commands to run all remaining experimental work on a GPU workstation.
> Estimated total time: **~12–20 hours** on a single T4 GPU (Kaggle: ~64s/epoch).

---

## Phase 0: Environment Setup (~5 min)

### Option A: Google Colab

```python
# Cell 1 — Clone and install
!git clone https://github.com/ShMazumder/tinyYOLO.git /content/tinyYOLO
%cd /content/tinyYOLO
!pip install -e . -q
!pip install tqdm -q

# Verify GPU
!python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}')"
```

### Option B: Kaggle

```python
# Cell 1
!git clone https://github.com/ShMazumder/tinyYOLO.git /kaggle/working/tinyYOLO
import sys; sys.path.insert(0, '/kaggle/working/tinyYOLO')
!pip install -e /kaggle/working/tinyYOLO -q
%cd /kaggle/working/tinyYOLO
```

### Option C: Dedicated GPU Workstation / RunPod / Vast.ai

```bash
# SSH into your workstation, then:
git clone https://github.com/ShMazumder/tinyYOLO.git ~/tinyYOLO
cd ~/tinyYOLO
pip install -e .

# Verify
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA:    {torch.cuda.is_available()}')
print(f'GPU:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
from tinyYOLO.models import build_model
model, info = build_model(task='det', variant='quantized', nc=20)
print(f'Model:   {info[\"total_params_M\"]}M params — OK ✓')
"
```

---

## Phase 1: Pascal VOC Training — 5 Seeds (~10–15 hours)

> **Purpose:** Primary benchmark (Table 1 in manuscript). 5 independent runs for mean ± std.
> **Expected time:** ~1.5 hours per seed on T4/Kaggle at 416×416 (64s/epoch × 300 epochs).
> ~4.5 hours per seed on Colab (265s/epoch × 300 epochs).

### 1.1 Download VOC dataset (automatic, ~2 GB)

```bash
# This will auto-download VOC via Ultralytics on first run.
# If you want to pre-download:
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('VOC.yaml')"
```

### 1.2 Train quantized variant (primary) — all 5 seeds

```bash
# Seed 42 (primary)
python scripts/train.py --task det --variant quantized --data voc.yaml \
  --imgsz 416 --epochs 300 --seed 42 --warmup 3 \
  --name voc-q-416-seed42

# Seed 123
python scripts/train.py --task det --variant quantized --data voc.yaml \
  --imgsz 416 --epochs 300 --seed 123 --warmup 3 \
  --name voc-q-416-seed123

# Seed 256
python scripts/train.py --task det --variant quantized --data voc.yaml \
  --imgsz 416 --epochs 300 --seed 256 --warmup 3 \
  --name voc-q-416-seed256

# Seed 512
python scripts/train.py --task det --variant quantized --data voc.yaml \
  --imgsz 416 --epochs 300 --seed 512 --warmup 3 \
  --name voc-q-416-seed512

# Seed 1024
python scripts/train.py --task det --variant quantized --data voc.yaml \
  --imgsz 416 --epochs 300 --seed 1024 --warmup 3 \
  --name voc-q-416-seed1024
```

### 1.3 Train standard variant — all 5 seeds

```bash
for SEED in 42 123 256 512 1024; do
  python scripts/train.py --task det --variant standard --data voc.yaml \
    --imgsz 416 --epochs 300 --seed $SEED --warmup 3 \
    --name voc-std-416-seed${SEED}
done
```

### 1.4 Collect VOC results

```bash
# After all runs complete, summarize results:
python -c "
import json, glob, numpy as np

for variant in ['q', 'std']:
    maps = []
    for f in sorted(glob.glob(f'experiments/results/voc-{variant}-416-seed*/config.json')):
        with open(f) as fh:
            cfg = json.load(fh)
        m = cfg['final_metrics']['mAP50']
        maps.append(m)
        print(f'  {f.split(\"/\")[-2]}: mAP@50 = {m:.4f}')
    if maps:
        print(f'  → {variant.upper()}: {np.mean(maps):.4f} ± {np.std(maps):.4f}')
    print()
"
```

---

## Phase 2: COCO val2017 Training (~6–10 hours)

> **Purpose:** Secondary benchmark (Table 2). 3 seeds minimum.
> **Expected time:** ~6–10 hours on T4 with full COCO (118K train images).

### 2.1 Download COCO (~20 GB)

```bash
# Auto-downloads on first run. To pre-download:
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('coco.yaml')"
```

> [!WARNING]
> Full COCO is ~20 GB. If disk space is limited, use `coco-val.yaml` (5K images, ~1 GB) as a lighter alternative.

### 2.2 Train on COCO — 3 seeds

```bash
for SEED in 42 123 256; do
  python scripts/train.py --task det --variant quantized --data coco.yaml \
    --imgsz 416 --epochs 300 --seed $SEED --warmup 3 \
    --name coco-q-416-seed${SEED}
done

# Standard variant (at least 1 run)
python scripts/train.py --task det --variant standard --data coco.yaml \
  --imgsz 416 --epochs 300 --seed 42 --warmup 3 \
  --name coco-std-416-seed42
```

### 2.3 Lighter alternative: COCO val2017 only (5K images)

```bash
# If you can't afford full COCO training time:
for SEED in 42 123 256; do
  python scripts/train.py --task det --variant quantized --data coco-val.yaml \
    --imgsz 416 --epochs 200 --seed $SEED --warmup 3 \
    --name cocoval-q-416-seed${SEED}
done
```

---

## Phase 3: Ablation Studies (~8–10 hours)

> **Purpose:** 10 ablation experiments on VOC (Table 9 in manuscript).
> Run each for **100 epochs** on VOC with quantized baseline.

### A1: Ghost vs Standard Convolutions

```bash
# You'll need to temporarily modify backbone.py to use standard Conv2d instead of GhostConv.
# For now, this is documented — implement the switch and run:
# python scripts/train.py --task det --variant quantized --data voc.yaml \
#   --imgsz 416 --epochs 100 --seed 42 --name ablation-a1-stdconv
```

### A4: Activation — ReLU6 vs SiLU with INT8

```bash
# Quantized (ReLU6) — already have results from Phase 1
# Standard (SiLU) — already have results from Phase 1
# Compare their mAP@50 values from config.json files
```

### A6: Resolution Ablation

```bash
# Sweep all resolutions
python scripts/train.py --task det --variant quantized --data voc.yaml \
  --imgsz 224,320,416,640 --sweep --epochs 100 --seed 42 --warmup 3
```

### A7: TAL vs Single-Cell Assignment

```bash
# TAL is now the default. To test single-cell, you'd temporarily
# revert the loss function. The TAL results come from Phase 1.
# Single-cell baseline: use a previous run's results or modify loss function.
```

### A9: Mosaic Augmentation

```bash
# With mosaic (default now):
python scripts/train.py --task det --variant quantized --data voc.yaml \
  --imgsz 416 --epochs 100 --seed 42 --warmup 3 \
  --name ablation-a9-mosaic-on

# Without mosaic (quick mode disables mosaic):
python scripts/train.py --task det --variant quantized --data voc.yaml \
  --imgsz 416 --epochs 100 --seed 42 --warmup 3 --quick \
  --name ablation-a9-mosaic-off
# Note: --quick sets 5 epochs. For a fair comparison without mosaic,
# modify train.py to add a --no-mosaic flag, or run 100 epochs.
```

### A10: Objectness Head Impact

```bash
# Dedicated objectness is now default. Compare against old results
# (pre-revision runs that used max-class-logit proxy).
```

---

## Phase 4: Multi-Task Validation (~4–6 hours)

> **Purpose:** Validate segmentation and pose claims (Section 10).

### 4.1 Segmentation — COCO-Seg

```bash
# Train segmentation variant
python scripts/train.py --task seg --variant quantized --data coco-val.yaml \
  --imgsz 416 --epochs 200 --seed 42 --warmup 3 \
  --name seg-q-416-seed42

python scripts/train.py --task seg --variant standard --data coco-val.yaml \
  --imgsz 416 --epochs 200 --seed 42 --warmup 3 \
  --name seg-std-416-seed42
```

### 4.2 Pose Estimation — COCO-Pose

```bash
# Train pose variant
python scripts/train.py --task pose --variant quantized --data coco8-pose.yaml \
  --imgsz 416 --epochs 200 --seed 42 --warmup 3 \
  --name pose-q-416-seed42

python scripts/train.py --task pose --variant standard --data coco8-pose.yaml \
  --imgsz 416 --epochs 200 --seed 42 --warmup 3 \
  --name pose-std-416-seed42
```

---

## Phase 5: Quantization Experiments (~1–2 hours)

> **Purpose:** FP32 vs INT8 accuracy comparison (Table 6), QAT vs PTQ (Ablation A8).

### 5.1 PTQ — Post-Training Quantization

```bash
# Calibrate and quantize the best VOC model
python scripts/quantize.py --mode ptq \
  --weights experiments/results/voc-q-416-seed42/best.pt \
  --task det --variant quantized --data voc.yaml \
  --imgsz 416 --n-calib 500 --backend qnnpack

# Same for standard variant
python scripts/quantize.py --mode ptq \
  --weights experiments/results/voc-std-416-seed42/best.pt \
  --task det --variant standard --data voc.yaml \
  --imgsz 416 --n-calib 500 --backend qnnpack
```

### 5.2 QAT — Quantization-Aware Training

```bash
# Fine-tune with QAT (10 epochs, lower LR)
python scripts/quantize.py --mode qat \
  --weights experiments/results/voc-q-416-seed42/best.pt \
  --task det --variant quantized --data voc.yaml \
  --imgsz 416 --epochs 10 --lr 1e-4 --backend qnnpack
```

### 5.3 Export INT8 Models

```bash
# Export PTQ model to ONNX
python scripts/quantize.py --mode ptq \
  --weights experiments/results/voc-q-416-seed42/best.pt \
  --task det --variant quantized --data voc.yaml \
  --n-calib 500 --export onnx

# Export QAT model
python scripts/quantize.py --mode qat \
  --weights experiments/results/voc-q-416-seed42/best.pt \
  --task det --variant quantized --data voc.yaml \
  --epochs 10 --export onnx
```

---

## Phase 6: ONNX Export & Model Size Measurement (~10 min)

> **Purpose:** Report ONNX/TFLite file sizes (Reviewer D4).

### 6.1 Export all variants

```bash
# FP32 ONNX — quantized variant
python scripts/export.py \
  --weights experiments/results/voc-q-416-seed42/best.pt \
  --task det --variant quantized --imgsz 416 \
  --formats onnx,torchscript

# FP32 ONNX — standard variant
python scripts/export.py \
  --weights experiments/results/voc-std-416-seed42/best.pt \
  --task det --variant standard --imgsz 416 \
  --formats onnx,torchscript
```

### 6.2 Measure model sizes

```bash
python -c "
from pathlib import Path
print('Model Sizes:')
for f in sorted(Path('experiments').rglob('*.onnx')):
    print(f'  {f.name}: {f.stat().st_size / 1e6:.2f} MB')
for f in sorted(Path('experiments').rglob('*.torchscript')):
    print(f'  {f.name}: {f.stat().st_size / 1e6:.2f} MB')
for f in sorted(Path('experiments').rglob('*.pt')):
    print(f'  {f.name}: {f.stat().st_size / 1e6:.2f} MB')
"
```

---

## Phase 7: Edge Deployment Benchmarking

> **Purpose:** Validate edge deployment claims (Reviewer W3).
> This phase requires **actual edge hardware**.

### 7.1 Jetson Nano (TensorRT)

```bash
# On the Jetson Nano:
# 1. Copy the ONNX model to the Jetson
scp experiments/results/voc-q-416-seed42/exports/best.onnx jetson@<IP>:~/

# 2. On the Jetson, convert ONNX to TensorRT engine:
/usr/src/tensorrt/bin/trtexec \
  --onnx=best.onnx \
  --saveEngine=best_int8.engine \
  --int8 \
  --workspace=256 \
  --verbose

# 3. Benchmark latency:
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=best_int8.engine \
  --batch=1 \
  --iterations=1000 \
  --warmUp=100

# Record: latency (ms), throughput (FPS), peak GPU memory
```

### 7.2 Raspberry Pi 4 (TFLite)

```bash
# On the Raspberry Pi:
# 1. Convert ONNX → TFLite (do this on the GPU machine first):
pip install onnx-tf tensorflow
python -c "
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

model = onnx.load('experiments/results/voc-q-416-seed42/exports/best.onnx')
tf_rep = prepare(model)
tf_rep.export_graph('best_tf')

# Convert to TFLite with INT8
converter = tf.lite.TFLiteConverter.from_saved_model('best_tf')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('best_int8.tflite', 'wb') as f:
    f.write(tflite_model)
print(f'TFLite size: {len(tflite_model) / 1e6:.2f} MB')
"

# 2. Copy to RPi and benchmark:
scp best_int8.tflite pi@<IP>:~/

# 3. On RPi, run inference benchmark:
python3 -c "
import numpy as np, time
import tflite_runtime.interpreter as tflite

interp = tflite.Interpreter(model_path='best_int8.tflite')
interp.allocate_tensors()
inp = interp.get_input_details()
out = interp.get_output_details()

dummy = np.random.rand(1, 416, 416, 3).astype(np.float32)

# Warmup
for _ in range(10):
    interp.set_tensor(inp[0]['index'], dummy)
    interp.invoke()

# Benchmark
times = []
for _ in range(100):
    t0 = time.perf_counter()
    interp.set_tensor(inp[0]['index'], dummy)
    interp.invoke()
    times.append((time.perf_counter() - t0) * 1000)

print(f'Latency: {np.mean(times):.1f} ± {np.std(times):.1f} ms')
print(f'FPS:     {1000 / np.mean(times):.1f}')
"
```

> [!IMPORTANT]
> If you don't have physical edge hardware, you can:
> 1. Use **Jetson cloud instances** on Lambda Cloud or AWS (g4dn with T4)
> 2. Use **ONNX Runtime** CPU benchmarks as a proxy for ARM performance
> 3. Report TensorRT simulation results from `trtexec` on a T4 GPU with INT8 calibration

---

## Phase 8: Collect All Results (~30 min)

### 8.1 Generate comprehensive results summary

```bash
python -c "
import json, glob, numpy as np
from pathlib import Path

print('='*80)
print('  TinyYOLO R1 — Experimental Results Summary')
print('='*80)

# VOC results
print('\n--- Pascal VOC 2007+2012 (416×416, 300 epochs) ---')
for variant in ['q', 'std']:
    label = 'Quantized' if variant == 'q' else 'Standard'
    maps50, maps95 = [], []
    for f in sorted(glob.glob(f'experiments/results/voc-{variant}-416-seed*/config.json')):
        with open(f) as fh:
            cfg = json.load(fh)
        fm = cfg.get('final_metrics', {})
        maps50.append(fm.get('mAP50', 0))
        maps95.append(fm.get('mAP50_95', 0))
    if maps50:
        print(f'  {label:10s}: mAP@50 = {np.mean(maps50)*100:.1f} ± {np.std(maps50)*100:.1f}%'
              f'  |  mAP@50-95 = {np.mean(maps95)*100:.1f} ± {np.std(maps95)*100:.1f}%'
              f'  (n={len(maps50)})')

# COCO results
print('\n--- COCO val2017 (416×416, 300 epochs) ---')
for variant in ['q', 'std']:
    label = 'Quantized' if variant == 'q' else 'Standard'
    maps50 = []
    for f in sorted(glob.glob(f'experiments/results/coco*-{variant}-416-seed*/config.json')):
        with open(f) as fh:
            cfg = json.load(fh)
        maps50.append(cfg.get('final_metrics', {}).get('mAP50', 0))
    if maps50:
        print(f'  {label:10s}: mAP@50 = {np.mean(maps50)*100:.1f} ± {np.std(maps50)*100:.1f}%  (n={len(maps50)})')

# Resolution ablation
print('\n--- Resolution Ablation (quantized, 100 epochs, VOC) ---')
for f in sorted(glob.glob('experiments/results/tinyYOLO-det-q-*/config.json')):
    with open(f) as fh:
        cfg = json.load(fh)
    imgsz = cfg.get('imgsz', '?')
    m50 = cfg.get('best_mAP50', 0)
    print(f'  {imgsz}×{imgsz}: best mAP@50 = {m50:.4f}')

# Model sizes
print('\n--- Model Sizes ---')
for f in sorted(Path('experiments').rglob('*.onnx')):
    print(f'  {f.parent.parent.name}/{f.name}: {f.stat().st_size/1e6:.2f} MB')

print('\n' + '='*80)
"
```

### 8.2 Copy results back to local machine

```bash
# From your local machine:
scp -r user@gpu-host:~/tinyYOLO/experiments/results/ \
  /Applications/XAMPP/xamppfiles/htdocs/tinyYOLO/experiments/results/

# Or if using Colab, download the results zip:
# (in Colab cell)
!cd experiments && zip -r /content/results.zip results/
# Then download results.zip from Colab Files panel
```

---

## Time Estimates Summary

| Phase | Task | GPU Hours (Kaggle T4) | GPU Hours (Colab T4) | Priority |
|-------|------|----------------------|---------------------|----------|
| 1 | VOC 5 seeds × 2 variants | ~15h | ~45h | 🔴 **Mandatory** |
| 2 | COCO 3 seeds | ~10–15h | ~30h | 🔴 **Mandatory** |
| 3 | Ablation studies (10 experiments) | ~4–5h | ~12h | 🔴 **Mandatory** |
| 4 | Multi-task (seg + pose) | ~2–3h | ~6h | 🟡 Recommended |
| 5 | Quantization (PTQ + QAT) | ~1h | ~1h | 🟡 Recommended |
| 6 | ONNX export + sizes | ~10 min | ~10 min | 🟢 Quick |
| 7 | Edge hardware | ~2h + hardware | ~2h + hardware | 🟡 Recommended |
| 8 | Results collection | ~30 min | ~30 min | 🟢 Quick |

> [!TIP]
> **Parallelization:** If you have multiple GPUs or multiple Colab/Kaggle sessions, run Phases 1, 2, and 3 in parallel. Each seed is independent.

> [!TIP]
> **Minimum viable publication:** Phases 1 + 3 + 6 (VOC results + ablations + model sizes) are the absolute minimum for resubmission. Add Phase 2 for a stronger paper.

---

## 9. Training Acceleration (New in R1)

To hit the 64s/epoch benchmark and reduce total training time, use these flags:

### 9.1 Pretrained Backbone (`--pretrained`)
TinyYOLO now supports loading ImageNet-pretrained GhostNet weights. This allows models to converge in **~100–150 epochs** instead of 300, halving the required GPU hours.
```bash
python scripts/train.py --task det --pretrained --epochs 150
```

### 9.2 Torch Compile (`--compile`)
If using PyTorch 2.0+ (standard in Colab/Kaggle), this enables kernel fusion for **1.5–2.0x faster** training iterations.
```bash
python scripts/train.py --task det --compile
```

### 9.3 Combined Super-Fast Run
```bash
python scripts/train.py --task det --pretrained --compile --epochs 150 --batch 128
```
*Note: Batch 128 fits on T4 (15GB) for 416 resolution and maximizes utilization.*

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: Dataset not found` | Run the download command from Phase 1.1 or 2.1 first |
| `CUDA out of memory` | Reduce `--batch` (try 16, 8, or 4) |
| `Notebook OOM on multi-seed runs` | Training cache auto-skips datasets >5 GB. Restart kernel between seeds if needed |
| `Mosaic too slow` | Add `--quick` to disable mosaic, or reduce `--epochs` |
| `VOC download fails` | `pip install ultralytics --upgrade` then retry |
| `ONNX export fails` | `pip install onnx>=1.14.0 onnxruntime>=1.15.0` |
| Colab disconnects | Use Colab Pro, or split into shorter runs with checkpoint resume |
| `ModuleNotFoundError: tinyYOLO` | Run `pip install -e .` from the project root |
| `No module named 'tqdm'` | Run `pip install tqdm` |
| Slow epoch time (>120s on T4) | Ensure latest `train.py` with vectorized loss. Push/pull latest code |
| mAP stays at 0 or declining | Ensure `postprocess.py` uses `pred[:, 5:]` for classes and `sigmoid*imgsz` decode. Pull latest code |
| Validation hangs at eval epoch | Pull latest `postprocess.py` — old code had channel index bug generating millions of detections |
