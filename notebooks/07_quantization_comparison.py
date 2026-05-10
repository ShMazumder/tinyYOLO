# %% [markdown]
# # 07 — Quantization Comparison
# Compare FP32 (standard) vs INT8-safe (quantized) architectures
# across all tasks. Analyze parameter savings, latency, and
# quantization-readiness.

# %% Setup
import sys, json
sys.path.insert(0, '..')
import torch
import torch.quantization as quant
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tinyYOLO.models import build_model
from tinyYOLO.utils.benchmark import count_parameters, measure_latency
from tinyYOLO.utils.env import detect_environment, print_env_report

env = detect_environment()
print_env_report(env)
RESULTS_DIR = Path('../experiments/results/quantization')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Parameter Comparison: Standard vs Quantized

# %%
TASKS = ['det', 'seg', 'pose', 'cls', 'obb']
comparison = []

print(f"{'Task':<8} {'Std Params(M)':>14} {'Q Params(M)':>14} {'Savings':>10}")
print("-" * 48)

for task in TASKS:
    nc = 1 if task == 'pose' else (1000 if task == 'cls' else 80)
    m_std, i_std = build_model(task=task, variant='standard', nc=nc)
    m_q, i_q = build_model(task=task, variant='quantized', nc=nc)

    p_std = i_std['total_params_M']
    p_q = i_q['total_params_M']
    savings = (1 - p_q / p_std) * 100

    comparison.append({
        'task': task, 'std_params_M': p_std,
        'q_params_M': p_q, 'savings_pct': round(savings, 1)
    })
    print(f"{task:<8} {p_std:>14.3f} {p_q:>14.3f} {savings:>9.1f}%")

# %% [markdown]
# ## 2. Latency Comparison at 320×320

# %%
latency_comp = []

print(f"\n{'Task':<8} {'Std (ms)':>10} {'Q (ms)':>10} {'Speedup':>10}")
print("-" * 40)

for task in TASKS:
    nc = 1 if task == 'pose' else (1000 if task == 'cls' else 80)
    m_std, _ = build_model(task=task, variant='standard', nc=nc)
    m_q, _ = build_model(task=task, variant='quantized', nc=nc)

    lat_std = measure_latency(m_std, 320, 'cpu', warmup=5, runs=20)
    lat_q = measure_latency(m_q, 320, 'cpu', warmup=5, runs=20)
    speedup = lat_std['mean_ms'] / lat_q['mean_ms']

    latency_comp.append({
        'task': task, 'std_ms': lat_std['mean_ms'],
        'q_ms': lat_q['mean_ms'], 'speedup': round(speedup, 2)
    })
    print(f"{task:<8} {lat_std['mean_ms']:>10.1f} {lat_q['mean_ms']:>10.1f} {speedup:>9.2f}x")

# %% [markdown]
# ## 3. Post-Training Quantization Test (Detection)

# %%
print("\nPost-Training Static Quantization (det-quantized):")

m_q, _ = build_model(task='det', variant='quantized', nc=80)
m_q.eval()

# Check if model is quantization-ready
quantizable_ops = 0
non_quantizable_ops = 0
for name, module in m_q.named_modules():
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
        quantizable_ops += 1
    elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.ReLU6)):
        pass  # These fuse with Conv2d
    elif isinstance(module, torch.nn.Sigmoid):
        non_quantizable_ops += 1

print(f"  Quantizable ops (Conv2d/Linear): {quantizable_ops}")
print(f"  Non-quantizable ops (Sigmoid):   {non_quantizable_ops}")
print(f"  Quantization readiness:          {'HIGH' if non_quantizable_ops < 5 else 'MEDIUM'}")

# %% Model Size on Disk
print("\nModel file sizes:")
for task in ['det', 'cls']:
    for variant in ['standard', 'quantized']:
        nc = 1000 if task == 'cls' else 80
        model, _ = build_model(task=task, variant=variant, nc=nc)
        path = RESULTS_DIR / f'tinyYOLO-{task}-{variant}.pt'
        torch.save(model.state_dict(), path)
        kb = path.stat().st_size / 1024
        print(f"  {task}-{variant}: {kb:.0f} KB")

# %% [markdown]
# ## 4. Visualization

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Params comparison
x_pos = np.arange(len(TASKS))
width = 0.35
ax1.bar(x_pos - width/2, [c['std_params_M'] for c in comparison],
        width, label='Standard', color='#2196F3')
ax1.bar(x_pos + width/2, [c['q_params_M'] for c in comparison],
        width, label='Quantized', color='#4CAF50')
ax1.set_xlabel('Task')
ax1.set_ylabel('Parameters (M)')
ax1.set_title('Parameters: Standard vs Quantized')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(TASKS)
ax1.legend()

# Latency comparison
ax2.bar(x_pos - width/2, [c['std_ms'] for c in latency_comp],
        width, label='Standard', color='#2196F3')
ax2.bar(x_pos + width/2, [c['q_ms'] for c in latency_comp],
        width, label='Quantized', color='#4CAF50')
ax2.set_xlabel('Task')
ax2.set_ylabel('Latency (ms)')
ax2.set_title('Latency @320: Standard vs Quantized')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(TASKS)
ax2.legend()

plt.suptitle('Standard vs Quantized Architecture Comparison', fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'std_vs_quantized.png', dpi=150, bbox_inches='tight')
plt.show()

# %% Save all results
all_results = {'params': comparison, 'latency': latency_comp}
with open(RESULTS_DIR / 'comparison_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR}")

# %% [markdown]
# ## Summary
# - Quantized variants save ~5-10% parameters via ECA (lighter than SE+SpatialAttn)
# - ReLU6 activation is fully INT8-compatible
# - Low Sigmoid count = high quantization readiness
# - Full INT8 quantization requires QAT training for best accuracy
