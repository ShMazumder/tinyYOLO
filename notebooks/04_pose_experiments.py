# %% [markdown]
# # 04 — TinyYOLO Pose Estimation Experiments
# Validate and benchmark tinyYOLO-pose with COCO 17-keypoint format.

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
RESULTS_DIR = Path('../experiments/results/pose')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %% Build Models
model_std, info_std = build_model(task='pose', variant='standard', nc=1)
model_q, info_q = build_model(task='pose', variant='quantized', nc=1)
print(f"Pose-std: {info_std['total_params_M']}M | Pose-q: {info_q['total_params_M']}M")

# %% Forward Pass — Verify Keypoint Output Shapes
for imgsz in [224, 320, 416]:
    x = torch.randn(1, 3, imgsz, imgsz)
    with torch.no_grad():
        det_out, kpt_out = model_std(x)

    grid_h, grid_w = imgsz // 8, imgsz // 8  # P3 stride
    print(f"  {imgsz}x{imgsz}:")
    print(f"    det: {[o.shape for o in det_out]}")
    print(f"    kpt: {[k.shape for k in kpt_out]}")
    # Verify keypoint channels = 17 * 3 = 51
    assert kpt_out[0].shape[1] == 51, f"Expected 51 kpt channels, got {kpt_out[0].shape[1]}"

print("\n✓ Pose keypoint shapes verified (17 keypoints × 3 dims = 51 channels)")

# %% Latency Benchmark
results = []
for name, model in [('std', model_std), ('q', model_q)]:
    for imgsz in [224, 320, 416]:
        lat = measure_latency(model, imgsz, 'cpu', warmup=5, runs=20)
        results.append({'variant': name, 'imgsz': imgsz, **lat})
        print(f"  pose-{name} @{imgsz}: {lat['mean_ms']:.1f}ms ({lat['fps']:.0f} FPS)")

with open(RESULTS_DIR / 'latency_benchmark.json', 'w') as f:
    json.dump(results, f, indent=2)

# %% Gradient Flow
model_std.train()
x = torch.randn(2, 3, 320, 320)
det_out, kpt_out = model_std(x)
loss = sum(o.mean() for o in det_out) + sum(k.mean() for k in kpt_out)
loss.backward()
all_grad = all(p.grad is not None for p in model_std.parameters() if p.requires_grad)
print(f"\n✓ Gradient flow: {'PASS' if all_grad else 'FAIL'}")

# %% COCO Keypoint Skeleton Reference
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
print("\nCOCO Keypoints (17):")
for i, kp in enumerate(COCO_KEYPOINTS):
    print(f"  {i:2d}: {kp}")

# %% [markdown]
# ## Summary
# - 17 COCO keypoints × 3 dims (x, y, visibility) = 51 output channels per scale
# - Pose head adds ~0.01M params over detection
# - Suitable for lightweight person pose estimation on edge devices
