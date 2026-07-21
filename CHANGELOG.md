# Changelog

All notable changes to TinyYOLO are recorded here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/). **This file is the authoritative
change record going forward — update it in the same commit as any code/doc change.**

> **Git provenance note.** The full upstream commit history lives in the packed
> git objects but was not traversable in the environment where this file was
> created (no `git` binary available offline; the local reflog `.git/logs/HEAD`
> contained only `clone` + one `commit: up` by Shazzad Hossain Mazumder). Entries
> for R0–R1.3 below are therefore **reconstructed from the maintained Revision
> History in `README.md`**, not from `git log`. To regenerate a precise history
> once a git client is available:
> ```bash
> git log --pretty='%h %ad %an %s' --date=short > analysis/git_history.txt
> ```

---

## [Unreleased]

### Docs — repo-wide consistency & integrity pass (R1.4)
- **Reviewed all package code** (`common/backbone/neck/models/heads`, loss, decode, metrics) as
  an ML/CV/YOLO review. Architecture is sound; the only substantive defects were the box decode
  (fixed) and the un-wired TAL (fixed).
- **Retracted the false "P1b box-decode fix"** narrative in `revised/code_fixes_and_readiness.md`:
  documented that changing the decode to `sigmoid×imgsz` (removing grid anchoring) was the
  root-cause bug, not a fix. Downgraded the readiness matrix from "all RESOLVED / Q1-ready" to
  honest per-item status; overall = **not submission-ready** until results regenerated.
- **Struck all unbacked result numbers → `TBD (rerun)`** across `report.md`, `README.md`, the four
  `revised/revised_manuscript_part*.md`, and `revised/reviewer_rebuttal_letter.md` (VOC 41.2%,
  COCO 19.7%, INT8 0.7%, Jetson/RPi latency, ablation deltas +7.8/+4.3/+2.6/+1.6, multi-task
  metrics, 62.8% win claim). Added retraction banners to each manuscript part and the rebuttal.
- **Fixed bug-reintroducing guidance:** `analysis/gpu_experiment_guide.md` and
  `analysis/revision_analysis.md` previously *instructed* the broken `sigmoid×imgsz` decode —
  corrected to the R1.4 grid-anchored codec. `analysis/post_experiment_roadmap.md` edge
  latencies re-marked NOT measured.
- **Added R1.4 addendum** to `review/final_audit_report.md` (the audit assumed 41.2% was real; the
  true issue was mAP≈0). Updated `presentation/README.md` "Q1-ready" copy. Left
  `review/peer_review.md` as historical reviewer input.

### Fixed
- **cls/obj loss normalization (YOLOX-style) — fixes "0 predictions at eval".** cls and obj
  losses were `mean`-reduced, diluting the <1% positive signal ~100× (cls gave the true class
  only 1/nc of the gradient; obj positives were swamped by negatives). Confidences never rose
  above the init floor, so `obj×cls < val-conf` everywhere → 0 predictions, mAP 0 (boxes were
  fine — box loss trained to 0.73). Now both are **summed and normalized by N_pos** like the box
  loss; objectness uses `pos_weight` = 1.0 under the new normalization. _Re-run stage1: expect
  cls loss to actually drop and predictions/mAP > 0._
- **Scale-aware label assignment** (`DetectionLoss._select_level_gts`, `scripts/train.py`).
  Second bug found via the COCO128 smoke run: TAL assigned every GT at all 3 scales, flooding
  the coarse P5 grid (~70 positives / 100 cells at 320px). With `pos_weight=4` this made the
  objectness loss enormous (~7–11) and unlearnable, so confidences never rose above the init
  floor → **0 predictions at eval, mAP 0** (box loss trained fine — decode was OK). Fix routes
  each GT to only the FPN level matching its size (small→P3, medium→P4, large→P5), standard
  FCOS/FPN practice. Box loss had already confirmed the decode fix (0.99→0.73); this addresses
  the objectness/confidence collapse. _Re-run stage1 to verify predictions > 0 and mAP > 0._

### Added
- **A1 decode ablation switch** — `TINYYOLO_LEGACY_DECODE=1` reverts `boxcodec` to the broken
  pre-R1.4 `sigmoid` decode (affects both loss and inference since they share the codec). Lets
  `stage4_ablations.py` run grid-anchored-vs-legacy as an automated A/B with no code edits.
  Never enable for a real run.

### Diagnosed
- **"0 predictions" root cause = under-training, not a bug.** `diagnose_confidence.py` on the
  trained `best.pt` showed objectness stuck at the init floor (obj max 0.014 ≈ σ(-4.6)) and cls
  near floor — confidences never left initialization. Cause: coco128 has 128 images, so at the
  auto batch (128) the run does **1 gradient step/epoch ≈ 100 steps total** — far too few (stage0
  overfit needed 300 steps). Pipeline is correct; the smoke test was starved of steps. Fixes:
  `stage1` now forces `--batch 16` (~800 steps); `obj_pos_weight` 1.0→2.0 to climb faster. Real
  training (VOC, ~12.9k steps/run) is unaffected.

### Verified
- **TAL + decode confirmed together** — `stage0_sanity.py` with TAL wired: total loss 5.13→0.31
  (94% drop), box 0.99→0.14. Localization learns; TAL gives denser positives than single-cell.
- **Decode fix confirmed on real execution** via `experiments/plan/stage0_sanity.py`:
  decode round-trip exact (err 0.0), 284 grad tensors none-zero/no-NaN, overfit-one-
  batch total loss 6.63→1.14 and box 2.81→0.49 by step 100 (old broken decode floored
  box ≈2.19). Localization now learns.

### Notes
- Confirmed the existing suite `experiments/01_*.ipynb … 07_*.ipynb` (and `archive/`)
  train exclusively via `!python scripts/train.py`, so the R1.4 decode fix + TAL wiring
  apply to them with no per-notebook edits. That suite is the canonical
  manuscript-reproduction path; **all prior on-disk results predate the fix and must
  be regenerated.** `experiments/plan/` is now positioned as a sanity gate + thin driver
  (see its README), not a parallel suite.

### Changed
- **Wired `TALAssigner` into `DetectionLoss.forward`** (`scripts/train.py`). The
  loss now uses Task-Aligned top-k (k=10) multi-positive assignment via the
  alignment metric `t = s^0.5 · u^6.0`, replacing naive single-cell assignment.
  Boxes are decoded with the shared grid-anchored codec (`boxcodec.decode_grid`)
  for both assignment (detached) and CIoU (grad path); objectness targets are set
  at all positive cells. Added `DetectionLoss._ciou_vec` helper and a `topk` arg.
  This realizes the previously dead-code TAL path (manuscript's claimed +7.8% mAP).
  _Not yet run/measured — verify with Stage 0 + ablation A2 (TAL vs single-cell)._

---

## [R1.4] — 2026-07-22 — Critical box-decode fix

### Fixed
- **Box localization was broken; mAP collapsed to ≈0.** Both the training loss
  (`scripts/train.py::DetectionLoss.forward`) and inference decode
  (`tinyYOLO/utils/postprocess.py::decode_predictions`) parametrized boxes as
  `cx = σ(pred) × imgsz` with **no grid-cell anchoring**. A translation-equivariant
  convolutional head cannot regress an absolute image-space center from identical
  local features, so the real VOC run (`voc-q-320-seed42`) stalled at
  **mAP@50 ≈ 0.0011**. Replaced with a grid-anchored, anchor-free decode.

### Added
- `tinyYOLO/utils/boxcodec.py` — single canonical box codec shared by loss and
  inference (`decode_boxes`, `decode_grid`, `make_grid`). Parametrization:
  `cx=(gi+2σ(tx)−0.5)/W`, `cy=(gj+2σ(ty)−0.5)/H`, `w=exp(tw)/W`, `h=exp(th)/H`.
- `analysis/feasibility_and_experiment_plan.md` — feasibility assessment + staged
  experiment plan (Stages 0–7, ablations A1–A11, integrity checklist).
- `experiments/plan/` — ready-to-run experiment notebooks (see that folder's README).
- `CHANGELOG.md` — this file.

### Changed
- `tinyYOLO/utils/postprocess.py` — `decode_predictions` now calls
  `boxcodec.decode_grid`; removed the misleading "must NOT use grid-offset" comment.
- `scripts/train.py` — `DetectionLoss.forward` now decodes positive-cell boxes via
  `boxcodec.decode_boxes`; imports `boxcodec` at module top.
- `report.md` (§4.2) and `README.md` — box-decode description updated to the
  grid-anchored codec; added R1.4 row to the Revision History.

### Known issues / follow-ups (not yet done)
- ~~`TALAssigner` defined but never called~~ — **now wired** (see [Unreleased]).
- **Existing checkpoints (`best.pt`, `ema.pt`, `last.pt`) are invalid** — they were
  trained with the broken decode. All detection results must be regenerated.
- Result tables in `report.md` / `README.md` are **not backed by artifacts on disk**
  (only `voc-q-320-seed42` and a COCO128 toy run exist). Must be re-earned.

---

## [R1.3] — 2026-05-25 — AP math correctness _(reconstructed from README)_

### Fixed
- Overhauled Average Precision math in `tinyYOLO/utils/metrics.py`. Replaced linear
  interpolation (`np.interp`) with 101-point step-function interpolation (COCO style),
  removing low-recall AP over-estimation and boundary underestimation.
- Class-averaged mAP now skips categories with zero ground-truth annotations
  (`N_gt = 0`), matching pycocotools/COCOeval.

## [R1.2] — 2026-05-19 — Evaluation memory + metric correctness _(reconstructed)_

### Fixed
- Added a Per-Image Class-Aware Matching Engine in `metrics.py` (bounds eval memory
  ~24 KB), resolving a global coordinate-leakage bug that matched boxes across images.
- Corrected class-averaging that divided only by "active" (AP>0) classes (≈5× mAP
  inflation).
### Changed
- Conservative RAM caching (cache only <1.5 GB datasets fitting in 20% free RAM).
- Colab workers set to 2.

## [R1.1] — 2025-05-16 — Pending code fixes completed _(reconstructed)_

### Added
- `TALAssigner` class, LR warmup in training loop, `MosaicDataset` wrapper,
  QAT/PTQ pipeline (`scripts/quantize.py`), ONNX export docs.
  _(Note: `TALAssigner` added but not wired into the loss — see R1.4 follow-ups.)_

## [R1] — 2025-05-15 — Major peer-review revision _(reconstructed)_

### Added / Changed
- Head activation fix (configurable `act`), dedicated objectness head, seed control,
  warmup, train/val splits, VOC/COCO evaluation scaffolding, edge-deployment section,
  SOTA comparison tables, 10 ablation write-ups.

## [R0] — 2025-05-09 — Initial submission _(reconstructed)_

- First version. COCO128 evaluation only.
