# %% [markdown]
# # Stage 3 — COCO baseline
#
# Real COCO numbers to replace the unbacked COCO table (incl. AP_S/AP_M/AP_L,
# which currently have no artifact at all).
#
# `coco-val.yaml` (5K) is the affordable middle ground; `coco.yaml` (118K) is the
# gold standard but needs 6-10 h/run. Use >=3 seeds for reported numbers.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import run_train, summarize, print_table, SEEDS_3

DRY_RUN = False
EPOCHS = 300
DATA = "coco-val.yaml"      # switch to "coco.yaml" for the full benchmark
RES = 416
VARIANTS = ["quantized", "standard"]
SEEDS = SEEDS_3

# %%
for variant in VARIANTS:
    for seed in SEEDS:
        name = f"coco_{variant}_{RES}_seed{seed}"
        run_train(name, task="det", variant=variant, imgsz=RES, epochs=EPOCHS,
                  seed=seed, data=DATA, dry_run=DRY_RUN)

# %%
rows = []
for variant in VARIANTS:
    names = [f"coco_{variant}_{RES}_seed{s}" for s in SEEDS]
    stats, n = summarize(names)
    if stats:
        rows.append([
            variant, n,
            f"{stats['mAP50'][0]*100:.1f} ± {stats['mAP50'][1]*100:.1f}",
            f"{stats['mAP50_95'][0]*100:.1f} ± {stats['mAP50_95'][1]*100:.1f}",
        ])
print_table(rows, ["variant", "n_seeds", "mAP@50 (%)", "mAP@50-95 (%)"])
print("\n  NOTE: AP_S/AP_M/AP_L require size-bucketed eval — extend metrics.py if the")
print("  current DetectionMetrics does not already emit them, then re-tabulate.")
