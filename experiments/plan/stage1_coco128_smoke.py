# %% [markdown]
# # Stage 1 — COCO128 smoke test (GATE)
#
# End-to-end pipeline check on the tiny 128-image COCO subset (auto-downloads).
# ~30 min on a T4. Confirms the full train->eval->artifact path works and that
# boxes localize after the R1.4 fix, before spending real GPU-hours on Stage 2.
#
# **Pass:** mAP@50 clearly > 0 and predictions are not sprayed (FP not in the
# hundreds-of-thousands). The broken-decode VOC run had 644k predictions / mAP 0.001.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import run_train, load_metrics, print_table

DRY_RUN = False   # set True to just print commands

# %%
# coco128 has only 128 images -> at the auto batch (128) that's ONE gradient step
# per epoch (~100 steps total), far too few to train confidences off the init
# floor. Force a small batch so the smoke test does ~8 steps/epoch (~800 total).
name = "s1_coco128_q_320"
run_train(name, task="det", variant="quantized", imgsz=320, epochs=100,
          seed=42, data="coco128.yaml", extra=["--batch", "16"], dry_run=DRY_RUN)

# %%
m = load_metrics(name)
if m:
    print_table(
        [[f"{m['mAP50']:.4f}", f"{m['mAP50_95']:.4f}", f"{m['precision']:.4f}",
          f"{m['recall']:.4f}", m.get("n_predictions", "?"), m.get("fp", "?")]],
        ["mAP50", "mAP50-95", "P", "R", "n_pred", "FP"],
    )
    gate = m["mAP50"] > 0.05 and (m.get("n_predictions", 1e9) < 50000)
    print("  STAGE 1 PASS" if gate else "  STAGE 1 FAIL — inspect boxes/decode before Stage 2")
else:
    print("  no metrics.json — run with DRY_RUN=False on a GPU box")
