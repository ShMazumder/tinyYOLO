# %% [markdown]
# # Stage 6 — Edge / latency benchmark
#
# Two parts:
# 1. **Portable latency** (this host CPU/GPU) via `tinyYOLO.utils.benchmark` —
#    runs anywhere, good for relative comparisons and the resolution knee.
# 2. **Real edge hardware** (Jetson Nano / RPi 4) — must be run ON the device
#    with the vendor runtime. This notebook prints the exact steps; it cannot
#    fake those numbers. The manuscript's Jetson/RPi table must come from here.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root
from _utils import print_table
import torch
from tinyYOLO.models import build_model
from tinyYOLO.utils.benchmark import measure_latency   # (name may differ — see benchmark.py)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %% [markdown]
# ## Part 1 — Portable latency sweep (this host)

# %%
rows = []
for variant in ["standard", "quantized"]:
    model, _ = build_model(task="det", variant=variant)
    model = model.to(DEVICE).eval()
    for imgsz in [160, 224, 320, 416, 640]:
        try:
            lat = measure_latency(model, imgsz, DEVICE, warmup=5, runs=50)
            rows.append([variant, imgsz, f"{lat['mean_ms']:.1f}", f"{lat.get('fps', 0):.0f}"])
        except Exception as e:
            rows.append([variant, imgsz, "ERR", str(e)[:30]])
print(f"  host device: {DEVICE}")
print_table(rows, ["variant", "imgsz", "ms", "FPS"])

# %% [markdown]
# ## Part 2 — Real hardware protocol (run on device)
#
# **Do not estimate. Measure.** Steps:
#
# 1. Export INT8: `python scripts/quantize.py --mode qat --weights <best.pt> --data voc.yaml --export onnx`
# 2. **Jetson Nano (TensorRT):**
#    ```bash
#    /usr/src/tensorrt/bin/trtexec --onnx=model_int8.onnx --int8 --iterations=200 --avgRuns=200
#    ```
#    Record median latency; FPS = 1000 / median_ms.
# 3. **Raspberry Pi 4 (TFLite):** convert to INT8 `.tflite`, then benchmark:
#    ```bash
#    ./benchmark_model --graph=model_int8.tflite --num_runs=200 --use_xnnpack=true
#    ```
# 4. **Power:** measure with an inline USB power meter under sustained inference.
#    If you cannot instrument power, DROP the power column — do not estimate it.
#
# Fill and log:

# %%
print_table(
    [["Tesla T4",     "TensorRT", "<fp32>", "<fp16>", "<int8>", "<fps>"],
     ["Jetson Nano",  "TensorRT", "<fp32>", "<fp16>", "<int8>", "<fps>"],
     ["Raspberry Pi 4","TFLite",  "<fp32>", "—",      "<int8>", "<fps>"]],
    ["platform", "runtime", "FP32 ms", "FP16 ms", "INT8 ms", "INT8 FPS"],
)
