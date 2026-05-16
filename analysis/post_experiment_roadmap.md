# TinyYOLO — Post-Experiment Roadmap

**Status:** GPU experiments IN PROGRESS (Kaggle: 64s/epoch, Colab: 265s/epoch)  
**Last updated:** 2025-05-16  
**Prerequisite:** Run experiment notebooks `01`–`05` in `experiments/`  
**Training speed:** Kaggle T4 = ~5.3h/seed, Colab T4 = ~22h/seed (vectorized loss + RAM caching)

---

## Phase 1: Results Integration (~1 day)

- [ ] **1.1** Run `experiments/05_results_analysis.ipynb` to generate consolidated CSVs and plots
- [ ] **1.2** Replace projected values in `revised/revised_manuscript_part1.md` Tables 1–2 with actual VOC/COCO numbers from `manuscript_table1_voc.csv` and `manuscript_table2_coco.csv`
- [ ] **1.3** Replace projected values in `revised/revised_manuscript_part2.md` Tables 3–4 (SOTA comparison)
- [ ] **1.4** Update `revised/revised_manuscript_part3.md` Section 9 ablation tables with real data from `ablation_summary.json`
- [ ] **1.5** Update `revised/revised_manuscript_part4.md` Section 10 multi-task results (seg mAP, pose AP)
- [ ] **1.6** Update `report.md` summary tables with actual metrics
- [ ] **1.7** Update `analysis/revision_analysis.md` — mark all experimental items as ✅ COMPLETE

### Files to update with real numbers:

| File | What to replace |
|------|----------------|
| `revised/revised_manuscript_part1.md` | Tables 1–2: VOC/COCO mAP@50, mAP@50-95, P, R (mean ± std) |
| `revised/revised_manuscript_part2.md` | Tables 3–4: SOTA comparison numbers |
| `revised/revised_manuscript_part3.md` | Ablation A1–A10 results |
| `revised/revised_manuscript_part4.md` | Seg mask mAP, Pose keypoint AP |
| `report.md` | Summary performance tables |
| `analysis/revision_analysis.md` | Status checkboxes |

---

## Phase 2: Edge Hardware Validation (~1-2 days)

> Requires physical access to Jetson Nano and/or Raspberry Pi 4.

- [ ] **2.1** Export quantized model to TensorRT INT8 for Jetson Nano
  ```bash
  # On Jetson Nano:
  python scripts/export.py --weights best_q.pt --task det --variant quantized \
    --imgsz 416 --formats engine --int8
  ```
- [ ] **2.2** Measure Jetson Nano latency (FP32 / FP16 / INT8) — fill Table 5
- [ ] **2.3** Export quantized model to TFLite INT8 for Raspberry Pi 4
  ```bash
  python scripts/export.py --weights best_q.pt --task det --variant quantized \
    --imgsz 416 --formats tflite --int8
  ```
- [ ] **2.4** Measure RPi4 latency (FP32 / INT8) — fill Table 5
- [ ] **2.5** Measure memory footprint on both platforms — fill Table in Section 8.4
- [ ] **2.6** Document thermal observations — Section 8.5
- [ ] **2.7** Update `revised/revised_manuscript_part2.md` Section 8 with real edge data

### Target numbers to fill (Table 5):

| Platform | FP32 (ms) | FP16 (ms) | INT8 (ms) |
|----------|-----------|-----------|-----------|
| Tesla T4 | ? | ? | ? |
| Jetson Nano | ? | ? | ? |
| RPi4 | ? | N/A | ? |

---

## Phase 3: SOTA Comparison (~half day)

- [ ] **3.1** Install comparison models:
  ```bash
  pip install nanodet  # or clone from GitHub
  # Clone PicoDet, YOLO-Fastest repos
  ```
- [ ] **3.2** Run each SOTA model on identical VOC test set at 416×416:
  ```bash
  python scripts/benchmark_models.py --sota nanodet,picodet,yolo-fastest \
    --data voc.yaml --imgsz 416
  ```
- [ ] **3.3** Fill Tables 3–4 with comparison metrics (same hardware, same data, same resolution)
- [ ] **3.4** Verify the Pareto front plot includes all SOTA points

### SOTA models to benchmark:

| Model | Params | Source |
|-------|--------|--------|
| YOLO-Fastest | 0.25M | [GitHub](https://github.com/dog-qiuqiu/Yolo-Fastest) |
| NanoDet-m | 0.95M | [GitHub](https://github.com/RangiLyu/nanodet) |
| NanoDet-Plus | 1.17M | Same repo |
| PicoDet-XS | 0.93M | [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) |
| MCUNet | 0.74M | [GitHub](https://github.com/mit-han-lab/mcunet) |
| YOLOv5n | 1.90M | [Ultralytics](https://github.com/ultralytics/yolov5) |
| YOLOv8n | 3.20M | [Ultralytics](https://github.com/ultralytics/ultralytics) |

---

## Phase 4: Manuscript Finalization (~1-2 days)

- [ ] **4.1** Replace all figures in manuscript with generated PNGs from `experiments/results/`:
  - `voc_5seed_results.png` → Figure 3 (training curves)
  - `pareto_front.png` → Figure 4 (accuracy vs params)
  - `ablation_plots.png` → Figure 5 (ablation results)
  - `publication_figures.png` → Combined figure panel
  - `model_sizes.png` → Figure 6 (FP32 vs INT8 sizes)
- [ ] **4.2** Write statistical analysis paragraph in Section 11.2:
  - t-test results from `05_results_analysis.ipynb`
  - Report p-value and significance level
- [ ] **4.3** Update `revised/reviewer_rebuttal_letter.md`:
  - Replace all projected claims with actual numbers
  - E.g., "41.2% mAP@50" → actual measured value
- [ ] **4.4** Cross-reference audit:
  - Verify all "Table X" / "Section Y" / "Figure Z" references match
  - Verify all ablation IDs (A1–A10) are consistent
- [ ] **4.5** Final proofread of all `revised/` files
- [ ] **4.6** Update `README.md` results section with final numbers

---

## Phase 5: Submission (~1 day)

- [ ] **5.1** Git commit all results and updated documents:
  ```bash
  git add -A
  git commit -m "R1.2: Add experimental results — VOC/COCO 5-seed, ablations, edge deployment"
  git push origin main
  ```
- [ ] **5.2** Prepare supplementary material:
  - COCO128 smoke test results (from existing runs)
  - Full per-class AP breakdown (all 20 VOC / 80 COCO classes)
  - Training configuration JSON dumps
  - Complete ablation data tables
- [ ] **5.3** Compile final manuscript (LaTeX or PDF export)
- [ ] **5.4** Submit to journal:
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

| Phase | Duration | Can Parallelize? |
|-------|----------|-----------------|
| GPU Experiments (01–04) | 1–2 days (Kaggle) / 4–5 days (Colab) | ✅ Run on multiple sessions |
| Results Integration | 1 day | After experiments |
| Edge Hardware | 1–2 days | ✅ Parallel with results integration |
| SOTA Comparison | 0.5 days | ✅ Parallel with edge |
| Manuscript Finalization | 1–2 days | After all data collected |
| Submission | 1 day | Final step |
| **Total** | **~4–7 days** | |
