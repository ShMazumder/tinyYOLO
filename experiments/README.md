# TinyYOLO Experiment Suite

Complete set of notebooks to reproduce **all** results in the manuscript.

## Execution Order & GPU Time

| # | Notebook | Manuscript Tables | GPU Time (T4) | Priority |
|---|----------|------------------|---------------|----------|
| 01 | `01_voc_benchmark.ipynb` | Table I, V | ~52h | 🔴 Critical |
| 02 | `02_coco_benchmark.ipynb` | Table IV | ~24h | 🔴 Critical |
| 03 | `03_cross_ablation.ipynb` | New Table (reviewer C1) | ~63h | 🔴 Critical |
| 04 | `04_ablations.ipynb` | Table IX | ~18h | 🟡 Important |
| 05 | `05_multitask.ipynb` | Table VIII | ~12h | 🟡 Important |
| 06 | `06_quantization.ipynb` | Table VII | ~4h | 🟡 Important |
| 07 | `07_profile_export.ipynb` | Tables II, III, VI | <1h | 🟢 Quick |

**Total: ~174h T4 GPU time** (~7 days continuous, or split across sessions)

## Recommended Execution Strategy

### RunPod / Vast.ai (no session limits)
Run all notebooks sequentially. Start with 03 (most critical for review).

### Kaggle (12h T4 sessions)
Split long notebooks across sessions using the `RUN_VARIANTS` / `RUN_CONFIGS` config cells:
- **Session 1:** 07 (profiling, <1h) + 04 (ablations, ~10h)
- **Session 2-6:** 01 (VOC quantized, 5 seeds × ~5h, 2 seeds/session)  
- **Session 7-11:** 01 (VOC standard, same split)
- **Session 12-15:** 03 (cross-ablation, 1 config per session)
- **Session 16-18:** 02 (COCO, 1 seed per session)
- **Session 19:** 05 (multi-task) + 06 (quantization)

### Colab Free (12h T4)
Same as Kaggle strategy.

### Local GPU
Run all notebooks as standard Jupyter notebooks.

## Skip-If-Done Logic

All notebooks check `experiments/results/<name>/summary.json` before each run. 
If a result already exists, it's skipped. This means you can **safely re-run** 
any notebook after a crash or session timeout — it picks up where it left off.

## Results

All results are saved to `experiments/results/` with:
- `summary.json` — key metrics (mAP, loss, epochs)
- `best.pt` — best model checkpoint
- `training_curves.png` — loss/accuracy plots
- `per_class_report.txt` — per-class breakdown
