# %% [markdown]
# # Stage 4 — Ablations
#
# Each ablation changes ONE variable vs a fixed baseline and reports delta mAP.
# Baseline: quantized, VOC, 416, 100 epochs, seeds {42,123,256}.
#
# Two kinds:
# - **CLI-driven** (A3 attention, A6 resolution, A7 mosaic) — fully automated below.
# - **Code-toggle** (A1, A2, A4, A5, A8, A9, A10, A11) — need a one-line source
#   change per arm; each is listed with the exact file/knob so a run is trivial.

# %%
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import run_train, summarize, print_table, SEEDS_3

DRY_RUN = False
EPOCHS = 100
DATA = "voc.yaml"
RES = 416
SEEDS = SEEDS_3


def run_arm(tag, variant="quantized", imgsz=RES, extra=None):
    names = []
    for s in SEEDS:
        name = f"abl_{tag}_seed{s}"
        run_train(name, task="det", variant=variant, imgsz=imgsz, epochs=EPOCHS,
                  seed=s, data=DATA, extra=extra, dry_run=DRY_RUN)
        names.append(name)
    return names


def report(label, arms):
    rows = []
    for arm_label, names in arms:
        stats, n = summarize(names)
        if stats:
            rows.append([arm_label, n, f"{stats['mAP50'][0]*100:.1f} ± {stats['mAP50'][1]*100:.1f}"])
    print(f"\n  == {label} ==")
    print_table(rows, ["arm", "n", "mAP@50 (%)"])

# %% [markdown]
# ## A1 — Box decode: grid-anchored (R1.4) vs legacy `sigmoid*imgsz` (the bug)
# Automated via the `TINYYOLO_LEGACY_DECODE` env flag — no code edit. Expect the
# legacy arm to collapse toward mAP≈0, quantifying the R1.4 fix. Short (40 ep) is
# enough to see the gap.

# %%
def run_arm_env(tag, env_flag=None, imgsz=RES, epochs=40):
    names = []
    prev = os.environ.get("TINYYOLO_LEGACY_DECODE")
    if env_flag:
        os.environ["TINYYOLO_LEGACY_DECODE"] = "1"
    else:
        os.environ.pop("TINYYOLO_LEGACY_DECODE", None)
    try:
        for s in SEEDS_3:
            name = f"abl_{tag}_seed{s}"
            run_train(name, task="det", variant="quantized", imgsz=imgsz,
                      epochs=epochs, seed=s, data=DATA, dry_run=DRY_RUN)
            names.append(name)
    finally:
        if prev is None:
            os.environ.pop("TINYYOLO_LEGACY_DECODE", None)
        else:
            os.environ["TINYYOLO_LEGACY_DECODE"] = prev
    return names

report("A1 decode", [
    ("grid-anchored (R1.4)", run_arm_env("A1_grid", env_flag=False)),
    ("legacy sigmoid*imgsz", run_arm_env("A1_legacy", env_flag=True)),
])

# %% [markdown]
# ## A3 — Attention: ECA vs SE vs None  (CLI: --attention)

# %%
report("A3 attention", [
    ("eca",  run_arm("A3_eca",  extra=["--attention", "eca"])),
    ("se",   run_arm("A3_se",   extra=["--attention", "se"])),
    ("none", run_arm("A3_none", extra=["--attention", "none"])),
])

# %% [markdown]
# ## A6 — Resolution: 224 / 320 / 416 / 640  (CLI: --imgsz)

# %%
report("A6 resolution", [
    (str(r), run_arm(f"A6_{r}", imgsz=r)) for r in [224, 320, 416, 640]
])

# %% [markdown]
# ## A7 — Mosaic on vs off  (CLI: --close-mosaic 0 disables from epoch 0)

# %%
report("A7 mosaic", [
    ("mosaic_on",  run_arm("A7_on")),
    ("mosaic_off", run_arm("A7_off", extra=["--close-mosaic", "0"])),
])

# %% [markdown]
# ## Code-toggle ablations (run after editing the noted knob)
#
# | # | Ablation | File & knob | Arms |
# |---|---|---|---|
# | A2 | **Assignment** | TAL is now the default (R1.4). For the single-cell arm, set `DetectionLoss.assigner.topk = 1` or restore single-cell assignment | single-cell vs TAL(k=10) |
# | A4 | Ghost vs standard conv | `tinyYOLO/modules/backbone.py` conv type | ghost vs conv |
# | A5 | Activation ReLU6 vs SiLU (+INT8 drop) | `configs/*/*.yaml` act, or compare variants | relu6 vs silu |
# | A8 | Objectness head vs max-class proxy | `tinyYOLO/modules/heads.py` obj branch | dedicated vs proxy |
# | A9 | Loss weights | `DetectionLoss.__init__` box_weight | 2.0 vs 5.0 vs 7.5 |
# | A10 | Width multiplier | backbone channel list `[16,24,40,80,160]` | 0.5x / 1.0x / 1.5x |
# | A11 | Scale assignment | loss GT->level mapping | all-levels vs size-aware |
#
# For each: make the edit, set a distinct `--name`/tag, run `run_arm`, and log the
# result in CHANGELOG.md. **A1 and A2 are the highest priority** — A1 proves the
# fix, A2 unlocks the paper's claimed +7.8%.

print("\n  CLI ablations launched. Code-toggle ablations: see table above.")
