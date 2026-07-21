# %% [markdown]
# # Stage 2 — VOC baseline (headline result)
#
# The real VOC benchmark that **replaces the fabricated table** in report.md.
# Standard + quantized, at 320 and 416, over 5 seeds, reported as mean ± std.
#
# Cost: ~2-3 h per run on a Tesla T4 -> the full grid is large. Trim `VARIANTS`,
# `RES`, or `SEEDS` to fit your compute; keep >=3 seeds for any number you report.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import run_train, summarize, print_table, SEEDS_5

DRY_RUN = False
EPOCHS = 300
DATA = "voc.yaml"

VARIANTS = ["quantized", "standard"]
RES = [416, 320]
SEEDS = SEEDS_5

# %% [markdown]
# ## Launch all runs
# Each run writes experiments/results/voc_<variant>_<res>_seed<seed>/.

# %%
for variant in VARIANTS:
    for res in RES:
        for seed in SEEDS:
            name = f"voc_{variant}_{res}_seed{seed}"
            run_train(name, task="det", variant=variant, imgsz=res, epochs=EPOCHS,
                      seed=seed, data=DATA, dry_run=DRY_RUN)

# %% [markdown]
# ## Aggregate mean ± std

# %%
rows = []
for variant in VARIANTS:
    for res in RES:
        names = [f"voc_{variant}_{res}_seed{s}" for s in SEEDS]
        stats, n = summarize(names)
        if stats:
            rows.append([
                variant, res, n,
                f"{stats['mAP50'][0]*100:.1f} ± {stats['mAP50'][1]*100:.1f}",
                f"{stats['mAP50_95'][0]*100:.1f} ± {stats['mAP50_95'][1]*100:.1f}",
            ])
print_table(rows, ["variant", "imgsz", "n_seeds", "mAP@50 (%)", "mAP@50-95 (%)"])
print("\n  -> paste these numbers (not the old ones) into report.md §7.1 / README VOC table")
