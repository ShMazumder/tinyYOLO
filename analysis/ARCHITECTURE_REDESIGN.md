# TinyYOLO — Architecture Audit & Redesign Plan (R2)

Review of `tinyYOLO/` against the YOLO lineage (v1 → YOLO26) and the sub-1M
edge-detector literature (YOLO-Fastest, NanoDet, PicoDet, FastestDet, Damo-YOLO).

Status: **partially implemented and measured.** Steps 1, 1b, 2a, 2b are in the
codebase and validated by `scripts/verify_r2_arch.py`. Section 0b records what the
measurements actually showed — including where this document was wrong.

---

## 0. Verdict up front

The repository is not a scaled-down YOLO. It is a **head-heavy net with almost no
feature extractor**, using a block (Ghost) whose core assumption is violated at
the widths chosen, and missing two things every YOLO since v4 has had (SPP-family
context module, and a fusion block with real depth).

Parameter audit — analytically counted from the current source:

| Component | Params | Share |
|---|---:|---:|
| `TinyBackbone` | 0.061 M | **28 %** |
| `LitePAN` | 0.081 M | 37 % |
| `TinyDetect` (nc=80) | 0.076 M | 35 % |
| **Total** | **0.218 M** | |

**The head has more parameters than the backbone.** For reference, in YOLOv8n the
backbone is ~40 % of the model and in NanoDet-Plus / PicoDet the backbone is the
majority. A detector cannot classify 80 categories from 61 k parameters of feature
extraction, and no loss-function or schedule fix changes that. This is the root
cause behind the Stage-1 class-head collapse documented in
`experiments/plan/README.md`: **the classifier is not starved of gradient so much
as starved of features.**

Honest ceiling at the current size, from published sub-1M results at ~320 px:

| Model | Params | COCO mAP50-95 |
|---|---:|---:|
| YOLO-Fastest-v2 | 0.25 M | ~11.9 |
| NanoDet-Plus-m | 0.95 M | 27.0 |
| PicoDet-S | 0.99 M | 30.6 |
| YOLOv8n | 3.2 M | 37.3 |
| **TinyYOLO (current)** | **0.22 M** | target ~10–13 |

A correct 0.22 M model lands near YOLO-Fastest-v2. Anything above ~15 mAP50-95
requires more capacity, not better tuning. Plan accordingly.

---

## 0b. Measured results (supersedes the reasoning below where they disagree)

Single-class structural gate, coco128, train==val, 320px, batch 8, 150 epochs.
`--single-cls` makes classification trivial so the box + confidence path is
measured in isolation.

| Configuration | box loss | mAP50 | mAP50-95 |
|---|---:|---:|---:|
| original (obj + `exp` + mosaic) | 0.627 **pinned** | 0.0112 | — |
| + 2a/2b (obj removed, `ltrb`) | 0.494 | 0.0098 | — |
| + mosaic off (150 ep) | 0.290, still falling | 0.3349 | 0.1623 |
| **+ step 3, 300 ep (`--assign-mode level`)** | **0.146** | **0.7793** | **0.5951** |

**The nc=1 structural gate passes** (bar was mAP50 > 0.40). Box + confidence path
is correct. Predictions at `conf 0.001` fell from 32261 to 6671 and precision rose
0.018 -> 0.110, i.e. the model became genuinely selective rather than spraying.

Three defects, each confirmed by the metric moving when it was fixed:

1. **D3 (objectness) — confirmed.** The decisive evidence was the nc=1 run: with
   classification trivially solved (cls loss 0.001) mAP50 was still 0.011. The
   class head was never the binding constraint. Score was `obj x cls`; with cls
   supervised at positives only it saturated near 1 everywhere, so the score
   collapsed to `obj` alone, which then had to rank ~41 negatives per positive.
2. **New defect, not in the original audit — `exp` codec unreachable targets.**
   Centre reach under `2*sigmoid(t)-0.5` is (-0.5,+1.5) cells, but the assigner's
   only spatial prior is "cell centre inside GT box". Edge cells of large objects
   were assigned targets they physically could not emit. A gradient-descent probe
   (`verify_r2_arch.py` section 6) fits one edge cell to a 5x5-cell GT: `exp`
   caps at **IoU 0.614**, `ltrb` reaches **0.995**. That cap is the observed
   box-loss floor.
3. **Mosaic on a 128-image set.** Training on 4-image composites while evaluating
   on clean images, with 128 images and a 0.25M model. Turning it off was worth
   more than 2a and 2b combined (0.0098 -> 0.3349). This was flagged in the very
   first review of the repo and then not acted on for several iterations.

### Where this document was wrong

- The original audit claimed the class head was "starved of features" and made D2
  (backbone is 28% of params) the headline. The nc=1 gate disproved that as the
  *binding* constraint. D2 remains true and still limits the ceiling, but it was
  not what produced mAP 0.001.
- D12 (add a P2/stride-4 level) was walked back after the competitive analysis —
  both sub-1M leaders added stride **64**, not stride 4.
- ~~mAP50-95/mAP50 of 0.48 means boxes are loose; pull DFL forward.~~ **Retracted.**
  That ratio was measured on a 150-epoch run that had not converged. At 300 epochs
  it is **0.76** (0.5951/0.7793) — better than YOLOv8n's 0.71 and NanoDet-Plus's
  0.65. Box tightness is not a defect; DFL returns to the "open ablation" pile.
- ~~Step 3: hard level routing (D10) is a band-aid and should be removed in favour
  of YOLOv8-style global assignment.~~ **Retracted — measured backwards.**

  | epoch | `--assign-mode level` | `--assign-mode global` |
  |---|---:|---:|
  | 90 | **0.2513** | 0.1614 |
  | 120 | **0.5575** | 0.3485 |
  | 150 | **0.6901** | 0.4998 |

  `level` also had lower box and cls loss throughout. TAL ranks candidate cells by
  the IoU of the model's *current* predictions, which are random at init; global
  assignment must discover the size->level mapping from that noisy signal, while
  hard routing supplies it as a prior. YOLOv8 and NanoDet absorb that discovery
  cost across 118k images / ~370k steps; at 4800 steps on 128 images it never pays
  for itself. **`level` is now the default.** Worth re-testing at VOC/COCO scale —
  this may invert with more data.

### Loss-weight rescaling (easy to miss)

Switching cls from positives-only to dense soft targets changed its magnitude
~100x (positives-only ~4, dense ~100, because the normaliser is the summed soft
target rather than the positive count). The inherited `box=2.0 / cls=1.0` weights
left cls outweighing box **63:1**. R2 uses YOLOv8's `box=7.5 / cls=0.5`, which
were tuned for exactly this formulation. `verify_r2_arch.py` now asserts the two
weighted contributions stay within 20:1.

---

## 1. Current architecture, as built

```
Input (320)
 └─ stem  ConvBNAct 3→16, k3 s2                                    /2
 └─ stage1  GhostBottleneck 16→24  s2                       d=1    /4
 └─ stage2  GhostBottleneck 24→40  s2                       d=2    /8   → P3 (40ch)
 └─ stage3  GhostBottleneck 40→80  s2  → attn3              d=3    /16  → P4 (80ch)
 └─ stage4  GhostBottleneck 80→160 s2  → attn4              d=2    /32  → P5 (160ch)
      ▼
   LitePAN   lat1×1 → 64 | FPN top-down | PAN bottom-up
             fusion node = concat(128) → 1×1 → 64 → DWConv3×3        (one 3×3 of mixing)
      ▼
   TinyDetect (per scale, decoupled)
      cls: DWConv3×3 ×2 → 1×1 → nc
      reg: DWConv3×3 ×2 → 1×1 → 4        (tx,ty,tw,th; w,h via exp)
      obj: 1×1 → 1                        ← straight off the neck, no conv stem
```

---

## 2. Lineage check — what TinyYOLO inherited, and what it skipped

| Generation | Key contribution | In TinyYOLO? |
|---|---|---|
| v1–v2 | grid regression, anchors, BN | n/a (anchor-free) |
| v3 | FPN 3-scale, multi-label cls, residual backbone | partial — FPN yes, residuals weak |
| v4 | **SPP**, PANet, mosaic, CIoU | PAN ✓, mosaic ✓, CIoU ✓, **SPP ✗** |
| v5 | **SPPF**, CSP (C3) blocks, strong aug pipeline | **✗ / ✗** |
| YOLOX | decoupled head, anchor-free, SimOTA, objectness | decoupled ✓, obj ✓ (kept — see D3) |
| v6 | **RepVGG re-param**, Rep-PAN, TAL | **✗** |
| v7 | E-ELAN, planned re-param, aux deep supervision | **✗** |
| v8 | **C2f fusion**, **objectness removed**, TAL + soft cls targets, DFL | TAL ✓ (assigner only), rest **✗** |
| v9 | PGI, GELAN | ✗ |
| v10 | **NMS-free dual assignment (o2m + o2o)**, rank-guided blocks, PSA | ✗ (head docstring claims "capable") |
| v11 | C3k2, C2PSA | ✗ |
| v12 | area attention, R-ELAN | ✗ |
| **YOLO26** (Jan 2026) | **DFL removed**, native NMS-free, **ProgLoss**, **STAL**, MuSGD | DFL-free ✓ (by accident), rest ✗ |

Two observations:

1. TinyYOLO's DFL-free direct regression **accidentally agrees with YOLO26**, which
   removed DFL precisely to enable clean one-to-one assignment and drop NMS. But
   YOLO26 removed DFL *alongside* ProgLoss, STAL, and an end-to-end head. Removing
   DFL without those is not modern — it is v3-era.
2. The one modern idea actually wired in (TAL) is undercut by keeping an objectness
   head, which forces the positives-only classification loss that starves the
   class head.

---

## 3. Defect list

Severity: **S1** blocks accuracy fundamentally · **S2** significant · **S3** polish.

| # | Defect | Sev | Where |
|---|---|---|---|
| D1 ✅ FIXED | **No SPP/SPPF.** P5 is 10×10 at 320 px and built from depthwise stacks, whose *effective* receptive field is far below the theoretical one. No global context anywhere in the network. | **S1** | `backbone.py` |
| D2 | **Backbone is 28 % of params.** Feature extraction is 61 k params for an 80-way problem. | **S1** | `models.py` |
| D3 ✅ FIXED | **Objectness head retained.** v8/v10/v11/YOLO26 all removed it; confidence = cls score with TAL soft targets. Keeping obj forces positives-only cls BCE → 79 down-gradients per positive → class logits collapse to the prior. | **S1** | `heads.py`, `train.py:~1030` |
| D4 | **Ghost blocks below their operating regime.** GhostNet exploits redundancy at 160–960 ch. At 24–40 ch there is none to exploit. Worse, `GhostBottleneck` *narrows* (`mid = c2//2`) and `GhostConv` halves again → the information path through stage2 is **10 channels wide**. Inverted residuals (expand→dw→project) are the correct primitive at this width. | **S1** | `common.py` |
| D5 | **Objectness head has no conv stem** — bare 1×1 off the neck while cls/reg each get two DWConv blocks. The hardest prediction gets the least capacity. | S2 | `heads.py:~72` |
| D6 | **Neck fusion is one depthwise 3×3 per node.** v8 uses C2f (split + n bottlenecks + dense concat). Multi-scale reasoning happens here and there is almost none. | S2 | `neck.py` |
| D7 | **No re-parameterization.** RepVGG-style train-time branches fuse to a single conv at export — free accuracy for an edge model, which is this repo's entire premise. | S2 | all |
| D8 | **P3 is only 40 ch.** Small objects are where tiny detectors lose the most mAP, and they live on P3. The neck then *expands* it to 64 — a lateral conv doing upsizing is wasted compute. | S2 | `backbone.py` |
| D9 ✅ FIXED | **`exp()` w/h regression.** High-variance, needs `clamp(max=4)` to not blow up. `ltrb` distance regression (FCOS/YOLOX/v8) is bounded, positive, stable, and pairs directly with CIoU. | S2 | `boxcodec.py` |
| D10 ❌ NOT A DEFECT | **Hard GT→level routing** (`_select_level_gts`, size thresholds 0.08/0.24) sends each GT to exactly one FPN level. Standard TAL assigns across all levels and lets the alignment metric decide. This band-aid exists to contain the objectness blow-up and becomes unnecessary once D3 is fixed. | S2 | `train.py:~935` |
| D11 ✅ FIXED | **Assigner does a Python loop with `.item()` per assignment** — ~1000 GPU→CPU syncs per step. Correctness-neutral, throughput-fatal. | S2 | `train.py:~245` |
| D12 | No P2 (/4) level and no option for one. | S3 | `backbone.py` |
| D13 | `SEBlock` uses `nn.Linear` on pooled features and `ECABlock` uses `Conv1d`; both break on several edge NPU compilers. 1×1 `Conv2d` is the portable form. | S3 | `common.py` |
| D14 | Attention only on P4/P5, none in the neck, and `attn3` is applied to `p4` (confusing off-by-one naming). | S3 | `backbone.py` |
| D15 | Depthwise convs quantize badly under per-tensor schemes (wide per-channel weight ranges). The `quantized` variant needs **per-channel** weight observers documented as a hard requirement. | S3 | `quantize.py` |

---

## 4. Proposed architecture — TinyYOLO v2

### 4.1 Block: `IRRep` (replaces `GhostBottleneck`)

```
IRRep(c1, c2, s=1, e=2):
    pw_expand : Conv1×1  c1 → c1*e     + BN + act
    dw        : RepDW3×3 (train: 3×3dw ‖ 1×1dw ‖ BN-identity  →  fuse to one 3×3dw)
    attn      : ECA (1×1 Conv2d form)                          [optional]
    pw_project: Conv1×1  c1*e → c2     + BN, no act
    + identity residual when s==1 and c1==c2
```

Rationale: expansion instead of narrowing (MobileNetV2's central result — at these
widths the bottleneck is the whole problem); re-parameterized depthwise gives
train-time capacity at zero inference cost; drops the current block's expensive
dual-conv shortcut path (`dw + pw`, both `act='none'`) in favour of identity.

### 4.2 Backbone

```
stem      ConvBNAct 3→16 k3 s2                              /2
stage1    IRRep 16→24   s2   d=1                            /4
stage2    IRRep 24→48   s2   d=2                            /8   → P3 (48 ch)
stage3    IRRep 48→96   s2   d=3   + ECA                    /16  → P4 (96 ch)
stage4    IRRep 96→192  s2   d=2   + ECA                    /32
SPPF      192→192, k=5 ×3, reduce to 48 internally          /32  → P5 (192 ch)
```

P3 widened 40→48. SPPF added (~25 k params, negligible FLOPs — this is the single
highest value-per-parameter change in the whole plan).

### 4.3 Neck: `BiLitePAN`

Keep the PAN topology, change the fusion node:

- **Weighted add instead of concat** (BiFPN fast normalized fusion,
  `Σ wᵢ·xᵢ / (Σ wᵢ + ε)`, `wᵢ ≥ 0` via ReLU). Removes the `concat → 128 → 1×1 → 64`
  step entirely; the merge conv shrinks 4×. Cost: 2–3 learnable scalars per node.
- **`C2fLite` block after each merge** — split channels, run *n* DW bottlenecks on
  one half, concat all intermediates, 1×1 fuse. Real depth where fusion happens.
- **Extra BiFPN skip**: backbone P4 → bottom-up P4 node directly.

### 4.4 Head: `TinyDetectV2`

```
per scale (weights SHARED across scales — 3× fewer head params):
    cls: DWConv3×3 → 1×1 → nc
    reg: DWConv3×3 → 1×1 → 4    (l, t, r, b via softplus)
    NO objectness branch
confidence = sigmoid(cls)
```

- **Objectness removed** (D3). Classification is supervised over *all* anchors with
  soft targets = normalized TAL alignment metric at positives, 0 elsewhere. This is
  the v8/v10/v11/YOLO26 formulation and it is what makes the class head learnable.
- **Weight sharing across scales** with per-scale BN (as in FCOS/NanoDet) cuts the
  head from 76 k → 23 k while *improving* generalization on small datasets.
- **`ltrb` regression** (D9) replaces `exp()` w/h.
- **Optional dual head for NMS-free** (v10/YOLO26): a one-to-many branch for rich
  training gradient + a one-to-one branch used at inference. The o2m branch is
  dropped at export → zero inference cost, no NMS. This finally makes the existing
  "NMS-free capable" docstring true.

### 4.5 Training changes (paired with the architecture)

| Change | Source | Fixes |
|---|---|---|
| Cls BCE over all anchors, soft TAL targets | YOLOv8 | D3 — class-head collapse |
| Remove hard GT→level routing | YOLOv8 TAL | D10 |
| Vectorize assigner (no `.item()` loop) | — | D11 |
| **STAL**: floor of ≥4 positive anchors for objects < 8 px | YOLO26 | small-object recall |
| **ProgLoss**: schedule box/cls weights over training | YOLO26 | late-stage large-object domination |
| Per-channel weight quantization observers | — | D15 |

### 4.6 Budget

Analytically counted (params exact; FLOPs given as a ratio to current because the
absolute estimator is uncalibrated against the repo's `estimate_flops`).

| Config | Channels | Depths | e | neck | Params (nc=80) | FLOPs vs current |
|---|---|---|---:|---:|---:|---:|
| current | 16,24,40,80,160 | 1,1,2,3,2 | — | 64 | 0.218 M | 1.0× |
| **v2-n** | 16,24,40,80,160 | 1,1,2,3,2 | 2 | 48 | **0.317 M** | ~2.6× |
| **v2-s** ← default | 16,24,48,96,192 | 1,1,2,3,2 | 2 | 48 | **0.429 M** | ~3.5× |
| v2-m | 16,32,64,128,256 | 1,1,2,3,2 | 2 | 64 | 0.749 M | ~5.9× |

`v2-s` at 0.43 M sits between YOLO-Fastest-v2 (0.25 M) and NanoDet-Plus (0.95 M),
is still **7× smaller than YOLOv8n**, and — unlike the current model — has enough
backbone to support an 80-way classifier. Backbone share rises from 28 % → ~66 %.

---

## 5. Staged migration — each step independently gated

Do these **in order**. Each is a separate commit with a measurable gate, so a
regression is attributable to one change. Do not batch them.

| Step | Change | Gate (VOC, nc=20, 320 px) | Expected |
|---|---|---|---|
| **0** | Fix Stage-1 measurement: run `--nc1` structural gate first | nc=1 mAP50 > 0.40 | confirms box+obj path |
| **1** | Add SPPF to backbone | mAP50-95 ≥ baseline + 1.5 | cheapest win available |
| **2** | Remove objectness; cls-as-confidence + soft TAL targets | ≥ step1 + 3.0 | the big one |
| **3** | Vectorize assigner; drop hard level routing; add STAL floor | ≥ step2 + 1.0, **≥1.5× faster/epoch** | |
| **4** | `GhostBottleneck` → `IRRep` (e=2, no rep yet) | ≥ step3 + 2.0 | |
| **5** | Add RepDW branches (train-time only) | ≥ step4 + 0.8, identical export params | verify fused == unfused output |
| **6** | Neck → `BiLitePAN` (weighted fusion + C2fLite) | ≥ step5 + 1.5 | |
| **7** | Head weight-sharing + `ltrb` regression | ≥ step6, params −50 k | |
| **8** | ProgLoss schedule | ≥ step7 + 0.5 | |
| **9** | *(optional)* dual o2m/o2o head, NMS-free export | ≥ step8 − 0.5, NMS removed | latency win |
| **10** | INT8 QAT with per-channel observers | ≤ 2.0 mAP drop vs FP32 | |

Steps 1–3 are loss/context fixes that require **no architecture rewrite** and
should be done before anything else — they are likely worth more than steps 4–7
combined at the current model size.

## 6. Ablation matrix for the paper

Every step above is a row. Run each on VOC at fixed seed ×3, report mean±std.
Additional single-factor ablations worth having:

- attention: none / ECA / SE / CBAM-lite, at P4+P5 vs all levels vs neck
- expansion factor e ∈ {1, 2, 3, 4} at fixed params (compensate with width)
- SPPF kernel k ∈ {3, 5, 9} and pool count ∈ {2, 3, 4}
- P2 level on/off (small-object AP_s specifically)
- assigner topk ∈ {5, 10, 13} and STAL floor on/off, reported on AP_s
- ltrb vs exp-wh box parametrization
- rep-branches: 3×3 only / +1×1 / +identity

---

## 7. What this does not fix

Even fully executed, a 0.43 M model will not approach YOLOv8n. The realistic target
is **NanoDet-Plus territory at half the parameters** — roughly 20–24 mAP50-95 on
COCO at 320 px. If the project needs more, the honest lever is capacity (`v2-m`
at 0.75 M), not further architectural cleverness.

---

## References

- [Ultralytics YOLO26 — Docs](https://docs.ultralytics.com/models/yolo26)
- [Why YOLO26 removes NMS and how that changes deployment](https://www.ultralytics.com/blog/why-ultralytics-yolo26-removes-nms-and-how-that-changes-deployment)
- [How YOLO26 trains smarter with ProgLoss, STAL, and MuSGD](https://www.ultralytics.com/blog/how-ultralytics-yolo26-trains-smarter-with-progloss-stal-and-musgd)
- [YOLO26: Key Architectural Enhancements (arXiv 2509.25164)](https://arxiv.org/pdf/2509.25164)
- [Ultralytics YOLO Evolution: YOLO26, YOLO11, YOLOv8, YOLOv5 (arXiv 2510.09653)](https://arxiv.org/pdf/2510.09653)
- [Native NMS-free inference with YOLO26 — LearnOpenCV](https://learnopencv.com/yolo26-nms-free-inference/)
