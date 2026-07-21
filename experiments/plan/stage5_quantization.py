# %% [markdown]
# # Stage 5 — Quantization (PTQ vs QAT)
#
# Drives `scripts/quantize.py` on a trained quantized checkpoint. Reports the
# FP32 -> INT8 mAP delta and on-disk size for PTQ and QAT, on both backends.
#
# Prereq: a trained checkpoint from Stage 2, e.g.
#   experiments/results/voc_quantized_416_seed42/best.pt

# %%
import sys, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import REPO_ROOT, QUANTIZE, print_table

DRY_RUN = False
WEIGHTS = REPO_ROOT / "experiments" / "results" / "voc_quantized_416_seed42" / "best.pt"
DATA = "voc.yaml"
IMGSZ = 416


def run_quant(mode, backend, extra=None):
    cmd = [sys.executable, str(QUANTIZE), "--mode", mode,
           "--weights", str(WEIGHTS), "--data", DATA,
           "--variant", "quantized", "--imgsz", str(IMGSZ),
           "--backend", backend]
    if mode == "ptq":
        cmd += ["--n-calib", "500"]
    if mode == "qat":
        cmd += ["--epochs", "10", "--lr", "1e-4"]
    if extra:
        cmd += extra
    print("  $", " ".join(cmd))
    if not DRY_RUN:
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

# %%
if not WEIGHTS.exists() and not DRY_RUN:
    print(f"  MISSING checkpoint: {WEIGHTS}\n  Run Stage 2 first (or point WEIGHTS at an existing best.pt).")
else:
    for backend in ["qnnpack", "fbgemm"]:      # ARM / x86
        run_quant("ptq", backend)
        run_quant("qat", backend)

# %% [markdown]
# ## Record the results
# quantize.py prints/saves INT8 mAP + model size. Fill this table and log it in
# CHANGELOG.md. Expected realistic INT8 drop at 0.22M: **1-3%** (not 0.7%).

# %%
print_table(
    [["PTQ", "qnnpack", "<fill>", "<fill>", "<fill>"],
     ["QAT", "qnnpack", "<fill>", "<fill>", "<fill>"],
     ["PTQ", "fbgemm",  "<fill>", "<fill>", "<fill>"],
     ["QAT", "fbgemm",  "<fill>", "<fill>", "<fill>"]],
    ["mode", "backend", "FP32 mAP@50", "INT8 mAP@50", "size (MB)"],
)
