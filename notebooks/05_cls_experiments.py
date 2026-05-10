# %% [markdown]
# # 05 — TinyYOLO Classification Experiments
# Validate and benchmark tinyYOLO-cls — the lightest variant.

# %% Setup
import sys, json
sys.path.insert(0, '..')
import torch
from pathlib import Path
from tinyYOLO.models import build_model
from tinyYOLO.utils.env import detect_environment, print_env_report
from tinyYOLO.utils.benchmark import count_parameters, measure_latency

env = detect_environment()
print_env_report(env)
RESULTS_DIR = Path('../experiments/results/cls')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %% Build Models
model_std, info_std = build_model(task='cls', variant='standard', nc=1000)
model_q, info_q = build_model(task='cls', variant='quantized', nc=1000)
print(f"Cls-std: {info_std['total_params_M']}M | Cls-q: {info_q['total_params_M']}M")

# %% Forward Pass — Verify Classification Output
for imgsz in [160, 224, 320]:
    x = torch.randn(1, 3, imgsz, imgsz)
    with torch.no_grad():
        logits = model_std(x)
    probs = torch.softmax(logits, dim=1)
    top5 = torch.topk(probs, 5)
    print(f"  {imgsz}x{imgsz}: logits={logits.shape}, top5_conf={top5.values[0].tolist()}")

print("\n✓ Classification outputs verified")

# %% Latency — Classification is fastest (no neck, no multi-scale)
results = []
for name, model in [('std', model_std), ('q', model_q)]:
    for imgsz in [160, 224, 320]:
        lat = measure_latency(model, imgsz, 'cpu', warmup=10, runs=50)
        results.append({'variant': name, 'imgsz': imgsz, **lat})
        print(f"  cls-{name} @{imgsz}: {lat['mean_ms']:.1f}ms ({lat['fps']:.0f} FPS)")

with open(RESULTS_DIR / 'latency_benchmark.json', 'w') as f:
    json.dump(results, f, indent=2)

# %% Model Size Comparison
import os, tempfile

for name, model in [('std', model_std), ('q', model_q)]:
    tmp = Path(RESULTS_DIR) / f'tinyYOLO-cls-{name}.pt'
    torch.save(model.state_dict(), tmp)
    size_kb = tmp.stat().st_size / 1024
    print(f"  cls-{name}: {size_kb:.0f} KB on disk")

# %% [markdown]
# ## Summary
# - Classification is the lightest: ~0.09M (std) / ~0.07M (q)
# - No neck needed — backbone P5 → global pool → FC
# - Extremely fast: suitable for MCU classification tasks
# - Model size: <400 KB — fits on most edge devices
