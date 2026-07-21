# Experiment Plan — Sanity Gate + Thin Driver

> **Read this first.** The **canonical manuscript-reproduction suite is
> `experiments/01_*.ipynb … 07_*.ipynb`** (maps to Tables I–IX, has skip-if-done
> and `RUN_VARIANTS` session-splitting). Those notebooks already train via
> `!python scripts/train.py …`, so the R1.4 decode fix and the TAL wiring apply to
> them automatically — just re-run them (previous results were produced pre-fix
> and are invalid). Use `01–07` for all headline tables.
>
> This `plan/` folder is a lighter, programmatic complement. Its **unique value is
> `stage0_sanity.py`** — a fast decode/gradient/overfit gate the `.ipynb` suite
> lacks. `stage1`–`stage7` here overlap `01–07`; prefer the `.ipynb` versions for
> anything that goes in the paper. Keep these as scriptable drivers / aggregators.

Runnable `# %%` cell scripts (VS Code / Jupyter / `python file.py`). Every stage
drives the **real** training path (`scripts/train.py`) via subprocess and reads
back `experiments/results/<name>/` artifacts — no reimplemented training or
metrics. Shared helpers live in `_utils.py`.

## Mapping to the canonical suite

| plan/ script | Canonical equivalent | Use which |
|---|---|---|
| `stage0_sanity.py` | _(none — unique)_ | **this** — run before everything |
| `stage1_coco128_smoke.py` | quick smoke via `07` | either |
| `stage2_voc_baseline.py` | `01_voc_benchmark.ipynb` | **01** for the paper |
| `stage3_coco_baseline.py` | `02_coco_benchmark.ipynb` | **02** |
| `stage4_ablations.py` | `04_ablations.ipynb`, `03_cross_ablation.ipynb` | **03/04** |
| `stage5_quantization.py` | `06_quantization.ipynb` | **06** |
| `stage6_edge_benchmark.py` | `07_profile_export.ipynb` | **07** + real HW |
| `stage7_multitask.py` | `05_multitask.ipynb` | **05** |

## Order (gates first)

| File | Stage | Hardware | Time | Gate? |
|---|---|---|---|---|
| `stage0_sanity.py` | Overfit one batch, decode round-trip, grad flow | CPU | seconds | **HARD GATE** |
| `stage1_coco128_smoke.py` | Full pipeline on COCO128 | GPU | ~30 min | **GATE** |
| `stage2_voc_baseline.py` | VOC 5-seed headline (replaces fabricated table) | T4 | ~2-3 h/run | — |
| `stage3_coco_baseline.py` | COCO mAP + AP_S/M/L | GPU | hours | — |
| `stage4_ablations.py` | A1-A11 (A3/A6/A7 automated; rest = 1-line toggles) | GPU | varies | — |
| `stage5_quantization.py` | PTQ vs QAT, qnnpack/fbgemm | GPU/CPU | ~1 h | — |
| `stage6_edge_benchmark.py` | Host latency + real Jetson/RPi protocol | edge HW | varies | — |
| `stage7_multitask.py` | seg/pose/cls/obb per-task metrics | GPU | hours | — |

## How to run

```bash
cd experiments/plan
python stage0_sanity.py          # must PASS before anything else
python stage1_coco128_smoke.py   # must PASS before Stage 2
python stage2_voc_baseline.py    # edit VARIANTS/RES/SEEDS to fit compute
```

Each stage has a `DRY_RUN` flag at the top — set `True` to print the exact
commands without launching training.

## Rules of the road

1. **Start at Stage 0.** If it fails, the localization path is still broken — fix
   before spending GPU time.
2. **No number in the paper without an artifact.** Every reported value must trace
   to a `metrics.json` produced by these runs.
3. **>=3 seeds** for any headline number; report mean ± std.
4. **Log every run in `../../CHANGELOG.md`.**
5. Highest-value code work before big runs: wire `TALAssigner` (ablation A2) and
   run the decode A/B (A1) to quantify the R1.4 fix.
