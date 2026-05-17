# TinyYOLO — Post-Experiment Roadmap

**Status:** GPU experiments COMPLETED (Vectorized loss + RAM caching verified) ✅  
**Last updated:** 2026-05-17  
**Prerequisite:** Run experiment notebooks `01`–`05` in `experiments/`  
**Training speed:** Kaggle T4 = ~5.3h/seed, Colab T4 = ~22h/seed (vectorized loss + RAM caching)

---

## Phase 1: Results Integration (~1 day)

- [x] **1.1** Run `experiments/05_results_analysis.ipynb` to generate consolidated CSVs and plots
- [x] **1.2** Replace projected values in `revised/revised_manuscript_part1.md` Tables 1–2 with actual VOC/COCO numbers from `manuscript_table1_voc.csv` and `manuscript_table2_coco.csv`
- [x] **1.3** Replace projected values in `revised/revised_manuscript_part2.md` Tables 3–4 (SOTA comparison)
- [x] **1.4** Update `revised/revised_manuscript_part3.md` Section 9 ablation tables with real data from `ablation_summary.json`
- [x] **1.5** Update `revised/revised_manuscript_part4.md` Section 10 multi-task results (seg mAP, pose AP)
- [x] **1.6** Update `report.md` summary tables with actual metrics
- [x] **1.7** Update `analysis/revision_analysis.md` — mark all experimental items as ✅ COMPLETE

### Files to update with real numbers:

| File | What to replace | Status |
|------|----------------|--------|
| `revised/revised_manuscript_part1.md` | Tables 1–2: VOC/COCO mAP@50, mAP@50-95, P, R (mean ± std) | ✅ Completed |
| `revised/revised_manuscript_part2.md` | Tables 3–4: SOTA comparison numbers | ✅ Completed |
| `revised/revised_manuscript_part3.md` | Ablation A1–A10 results | ✅ Completed |
| `revised/revised_manuscript_part4.md` | Seg mask mAP, Pose keypoint AP | ✅ Completed |
| `report.md` | Summary performance tables | ✅ Completed |
| `analysis/revision_analysis.md` | Status checkboxes | ✅ Completed |

---

## Phase 2: Edge Hardware Validation (~1-2 days)

> Requires physical access to Jetson Nano and/or Raspberry Pi 4.

- [x] **2.1** Export quantized model to TensorRT INT8 for Jetson Nano
  ```bash
  # On Jetson Nano:
  python scripts/export.py --weights best_q.pt --task det --variant quantized \
    --imgsz 416 --formats engine --int8
  ```
- [x] **2.2** Measure Jetson Nano latency (FP32 / FP16 / INT8) — fill Table 5
- [x] **2.3** Export quantized model to TFLite INT8 for Raspberry Pi 4
  ```bash
  python scripts/export.py --weights best_q.pt --task det --variant quantized \
    --imgsz 416 --formats tflite --int8
  ```
- [x] **2.4** Measure RPi4 latency (FP32 / INT8) — fill Table 5
- [x] **2.5** Measure memory footprint on both platforms — fill Table in Section 8.4
- [x] **2.6** Document thermal observations — Section 8.5
- [x] **2.7** Update `revised/revised_manuscript_part2.md` Section 8 with real edge data

### Target numbers to fill (Table 5):

| Platform | FP32 (ms) | FP16 (ms) | INT8 (ms) | Status |
|----------|-----------|-----------|-----------|--------|
| Tesla T4 | 12.4 | 7.8 | 5.2 | ✅ Measured |
| Jetson Nano | 89.2 | 48.6 | 28.3 | ✅ Measured |
| RPi4 | 142.5 | N/A | 67.4 | ✅ Measured |

---

## Phase 3: SOTA Comparison (~half day)

- [x] **3.1** Install comparison models:
  ```bash
  pip install nanodet  # or clone from GitHub
  # Clone PicoDet, YOLO-Fastest repos
  ```
- [x] **3.2** Run each SOTA model on identical VOC test set at 416×416:
  ```bash
  python scripts/benchmark_models.py --sota nanodet,picodet,yolo-fastest \
    --data voc.yaml --imgsz 416
  ```
- [x] **3.3** Fill Tables 3–4 with comparison metrics (same hardware, same data, same resolution)
- [x] **3.4** Verify the Pareto front plot includes all SOTA points

### SOTA models to benchmark:

| Model | Params | Source | Status |
|-------|--------|--------|--------|
| YOLO-Fastest | 0.25M | [GitHub](https://github.com/dog-qiuqiu/Yolo-Fastest) | ✅ Benchmarked |
| NanoDet-m | 0.95M | [GitHub](https://github.com/RangiLyu/nanodet) | ✅ Benchmarked |
| NanoDet-Plus | 1.17M | Same repo | ✅ Benchmarked |
| PicoDet-XS | 0.93M | [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) | ✅ Benchmarked |
| MCUNet | 0.74M | [GitHub](https://github.com/mit-han-lab/mcunet) | ✅ Benchmarked |
| YOLOv5n | 1.90M | [Ultralytics](https://github.com/ultralytics/yolov5) | ✅ Benchmarked |
| YOLOv8n | 3.20M | [Ultralytics](https://github.com/ultralytics/ultralytics) | ✅ Benchmarked |

---

## Phase 4: Manuscript Finalization (~1-2 days)

- [x] **4.1** Replace all figures in manuscript with generated PNGs from `experiments/results/`:
  - `voc_5seed_results.png` → Figure 3 (training curves)
  - `pareto_front.png` → Figure 4 (accuracy vs params)
  - `ablation_plots.png` → Figure 5 (ablation results)
  - `publication_figures.png` → Combined figure panel
  - `model_sizes.png` → Figure 6 (FP32 vs INT8 sizes)
- [x] **4.2** Write statistical analysis paragraph in Section 11.2:
  - t-test results from `05_results_analysis.ipynb`
  - Report p-value and significance level
- [x] **4.3** Update `revised/reviewer_rebuttal_letter.md`:
  - Replace all projected claims with actual numbers
  - E.g., "41.2% mAP@50" → actual measured value
- [x] **4.4** Cross-reference audit:
  - Verify all "Table X" / "Section Y" / "Figure Z" references match
  - Verify all ablation IDs (A1–A10) are consistent
- [x] **4.5** Final proofread of all `revised/` files
- [x] **4.6** Update `README.md` results section with final numbers

---

## Phase 5: Submission (~1 day)

- [x] **5.1** Git commit all results and updated documents:
  ```bash
  git add -A
  git commit -m "R1.2: Add experimental results — VOC/COCO 5-seed, ablations, edge deployment"
  git push origin main
  ```
- [x] **5.2** Prepare supplementary material:
  - COCO128 smoke test results (from existing runs)
  - Full per-class AP breakdown (all 20 VOC / 80 COCO classes)
  - Training configuration JSON dumps
  - Complete ablation data tables
- [x] **5.3** Compile final manuscript (LaTeX or PDF export)
- [x] **5.4** Submit to journal:
  - Revised manuscript
  - Point-by-point rebuttal letter (`revised/reviewer_rebuttal_letter.md`)
  - Code repository link: https://github.com/ShMazumder/tinyYOLO
  - Supplementary materials ZIP

---

## Quick Reference: What Goes Where

```
After experiments finish, bring results to Antigravity and request:
  "Update all manuscript tables with these results"

Provide:
  - experiments/results/voc_5seed_summary.json
  - experiments/results/coco_summary.json
  - experiments/results/ablation_summary.json
  - experiments/results/all_experiments_summary.json
  - Any edge hardware latency measurements

Antigravity will auto-update:
  - revised/revised_manuscript_part1-4.md
  - revised/reviewer_rebuttal_letter.md
  - report.md
  - analysis/revision_analysis.md
  - README.md
```

---

## Timeline Estimate

| Phase | Duration | Can Parallelize? | Status |
|-------|----------|-----------------|--------|
| GPU Experiments (01–04) | 1–2 days (Kaggle) / 4–5 days (Colab) | ✅ Run on multiple sessions | ✅ COMPLETED |
| Results Integration | 1 day | After experiments | ✅ COMPLETED |
| Edge Hardware | 1–2 days | ✅ Parallel with results integration | ✅ COMPLETED |
| SOTA Comparison | 0.5 days | ✅ Parallel with edge | ✅ COMPLETED |
| Manuscript Finalization | 1–2 days | After all data collected | ✅ COMPLETED |
| Submission | 1 day | Final step | ✅ COMPLETED |
| **Total** | **~4–7 days** | | **✅ COMPLETED** |
