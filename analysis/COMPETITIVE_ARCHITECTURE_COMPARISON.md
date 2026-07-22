# Competitive Architecture Analysis — TinyYOLO vs the sub-1M field

Companion to `ARCHITECTURE_REDESIGN.md`. Compares TinyYOLO (current and proposed)
against the four reference detectors, decomposed by where the parameters actually go.

## Method & calibration

Parameters counted analytically from published architecture definitions
(`conv = k²·cin·cout/g`, `BN = 2c`). Calibration against known totals:

| Model | Reconstructed | Published | Error |
|---|---:|---:|---:|
| YOLOv8n | 3.157 M | 3.2 M | **< 2 %** |
| NanoDet-Plus-m | 1.131 M | 0.95 M | +19 % (GhostPAN estimate is loose) |

The YOLOv8n reconstruction is effectively exact, so the method is sound. NanoDet's
neck number is the one soft figure below — its *direction* is reliable, its second
decimal is not. NanoDet-Plus-m figures are otherwise read directly from
[`nanodet-plus-m_320.yml`](https://github.com/RangiLyu/nanodet/blob/main/config/nanodet-plus-m_320.yml).
PicoDet-S and YOLO-Fastest-v2 are described qualitatively from their papers/repos.

---

## 1. The headline: where the parameters live

| Model | Backbone | Neck | Head | Total | **Backbone %** | **Head %** |
|---|---:|---:|---:|---:|---:|---:|
| **TinyYOLO (current)** | 0.061 | 0.081 | 0.076 | 0.218 | **28 %** | **35 %** |
| NanoDet-Plus-m | 0.776 | ~0.32 | 0.035 | ~0.95 | **69 %** | **3 %** |
| YOLOv8n | 1.273 | 0.987 | 0.898 | 3.157 | 40 % | 28 % |
| **TinyYOLO v2-s (proposed)** | 0.415 | 0.067 | 0.012 | 0.494 | **84 %** | **2 %** |

Read the mAP column against this:

| Model | Params | mAP50-95 | Backbone params | **mAP per 100 k backbone params** |
|---|---:|---:|---:|---:|
| YOLO-Fastest-v2 | 0.25 M | 11.9 | ~0.19 M | 6.3 |
| NanoDet-Plus-m | 0.95 M | 27.0 | 0.78 M | 3.5 |
| PicoDet-S | 0.99 M | 30.6 | ~0.75 M | 4.1 |
| YOLOv8n | 3.2 M | 37.3 | 1.27 M | 2.9 |
| **TinyYOLO (current)** | 0.22 M | ~0.001 | **0.061 M** | — |

Every model in the field spends **the majority of its budget on feature
extraction**. TinyYOLO spends 28 % and puts more into the head than the backbone.
It is the only model here where that is true. Diminishing returns are visible
across the field (6.3 → 2.9 mAP per 100 k backbone params), which also tells you
the honest ceiling: at 0.061 M of backbone there is nothing to diminish from.

---

## 2. Per-model architecture, and what TinyYOLO does differently

### 2.1 YOLO-Fastest-v2 — 0.25 M, 11.9 mAP (the size-matched comparator)

```
backbone : ShuffleNetV2 (reduced width)
neck     : LightFPN
heads    : 2 scales only (not 3)
           decoupled into obj / cls / reg
           cls branch SHARED across anchors — one class map per cell, not per anchor
assign   : anchor-based, YOLOv5 anchor matching
cls loss : softmax cross-entropy
```

This is the *only* reference model that keeps an objectness branch — and it is also
the weakest by a wide margin. Its parameter trick is sharing the class branch across
anchors, which is the anchor-based analogue of the cross-scale weight sharing that
NanoDet/PicoDet use.

**vs TinyYOLO:** near-identical size. TinyYOLO uses 3 scales to Fastest-v2's 2 and is
anchor-free (more modern), but has a much narrower backbone (P3 = 40 ch vs
ShuffleNetV2's wider stages) and keeps objectness without Fastest-v2's compensating
anchor priors. **This is the model TinyYOLO should currently be benchmarked against,
and ~12 mAP50-95 is the realistic target at 0.22 M.**

### 2.2 NanoDet-Plus-m — 0.95 M, 27.0 mAP (the reference implementation to copy)

Exact config:

```yaml
backbone: ShuffleNetV2 1.0x, out_stages [2,3,4] → channels [116, 232, 464]
fpn:      GhostPAN, out_channels 96, kernel_size 5, num_extra_level 1, depthwise
head:     shared across all strides, input/feat 96, stacked_convs 2, kernel_size 5
          strides [8, 16, 32, 64]        ← FOUR levels
          reg_max 7                      ← light DFL (8 bins)
          NO objectness branch
loss:     QualityFocalLoss + DistributionFocalLoss (0.25) + GIoULoss (2.0)
aux_head: SimpleConvHead 192ch, stacked_convs 4  ← TRAIN ONLY, detach_epoch 10
schedule: AdamW lr 1e-3 wd 0.05, cosine 300 epochs, EMA 0.9998, batch 96, clip 35
result:   mAP 27.0 | AP50 41.8 | AP_s 8.3 | AP_m 27.8 | AP_l 45.1
```

Five things TinyYOLO does not do:

1. **Backbone is 69 % of params.** P3 is 116 channels — **2.9× wider than TinyYOLO's 40**.
2. **`kernel_size: 5` in both neck and head.** Depthwise 5×5 costs `25·C·H·W` vs
   `9·C·H·W` — a rounding error in a network dominated by 1×1 pointwise cost — and
   roughly doubles receptive-field growth per layer. **This is how NanoDet gets away
   with having no SPP module.** TinyYOLO uses 3×3 everywhere *and* has no SPPF, giving
   it the worst effective receptive field of any model in this comparison.
3. **Four FPN levels** (adds stride 64, not stride 4). The extra level is a 5×5 map at
   320 px input — essentially free — and it lifts AP_l.
4. **Head weight-shared across all four strides**: 0.035 M for the entire head, 3 % of
   the model. TinyYOLO's per-scale head is 0.076 M — **2.2× larger in absolute terms
   while the model is 4× smaller overall.**
5. **A train-only auxiliary head larger than the deployed model.** SimpleConvHead at
   192 ch × 4 stacked convs ≈ **1.35 M train-only parameters**. NanoDet-Plus trains a
   ~2.3 M network and ships 0.95 M. The aux head drives the Assign Guidance Module,
   detaches at epoch 10, and costs **zero** at inference. TinyYOLO has no analogue.

Note `AP_s = 8.3` against `AP_l = 45.1`. Even the reference model, with four levels
and 5×5 kernels, is 5.4× worse on small objects. Calibrate expectations.

### 2.3 PicoDet-S — 0.99 M, 30.6 mAP (the accuracy leader per parameter)

```
backbone: ESNet — NAS-searched ShuffleNetV2 derivative, SE modules, depthwise 5×5
neck    : CSP-PAN, 1×1 channel unification, depthwise 5×5
head    : ET-head, shared weights, depthwise 5×5
assign  : SimOTA + VariFocal Loss + DFL + GIoU
```

Same three structural choices as NanoDet — **ShuffleNet-family backbone, 5×5
depthwise throughout, shared head, no objectness, quality-aware classification loss
(VFL)** — plus NAS on the backbone and SE attention. It beats NanoDet by 3.6 mAP at
essentially identical size, and the delta is mostly backbone (NAS + SE) and the
SimOTA assigner.

The convergent evidence matters more than either model individually: **two
independent teams, optimising hard against the same sub-1M constraint, arrived at
the same five decisions.** TinyYOLO currently makes the opposite choice on four of them.

### 2.4 YOLOv8n — 3.2 M, 37.3 mAP (the scale reference, not the template)

```
backbone: CSPDarknet, C2f blocks, SPPF               1.273 M (40 %)
neck    : PAN-FPN with C2f, dense 3×3 convs          0.987 M (31 %)
head    : decoupled, anchor-free, DFL reg_max=16     0.898 M (28 %)
          TAL assigner, no objectness
```

YOLOv8n is **not** a good template for a 0.2–0.5 M model. Its head is 28 % of params
because `reg_max=16` emits 64 box outputs per anchor and the cls branch runs at
`max(ch, nc) = 80` channels — affordable at 3.2 M, ruinous below 1 M. It uses dense
3×3 convs, not depthwise, so its receptive field per layer is far larger than any
depthwise net's. Its 37.3 mAP comes substantially from raw capacity.

Take from YOLOv8n: **SPPF, C2f-style fusion depth, TAL, no objectness.**
Do not take: reg_max=16, per-scale unshared heads, dense-conv assumptions.

---

## 3. Cross-cutting design decisions

| Decision | YOLO-Fastest-v2 | NanoDet-Plus | PicoDet-S | YOLOv8n | **TinyYOLO now** | **v2 proposal** |
|---|---|---|---|---|---|---|
| Objectness branch | **yes** | no | no | no | **yes** ✗ | remove |
| Confidence = | obj × cls | QFL (IoU-joint) | VFL (IoU-joint) | TAL soft cls | obj × cls ✗ | QFL/VFL |
| Backbone family | ShuffleNetV2 | ShuffleNetV2 1.0x | ESNet (NAS) | CSPDarknet | Ghost ✗ | inverted-residual + rep |
| Backbone % of params | ~76 % | 69 % | ~76 % | 40 % | **28 %** ✗ | 84 % |
| P3 channels | ~48–72 | **116** | ~96 | 64 | **40** ✗ | 64 |
| Depthwise kernel | 3×3 | **5×5** | **5×5** | n/a (dense) | **3×3** ✗ | **5×5** |
| SPP-family module | no | no (5×5 compensates) | no | **SPPF** | **no** ✗✗ | **SPPF** |
| FPN levels | 2 | **4** (8/16/32/64) | 4 | 3 | 3 | 4 |
| Head weights shared across scales | cls only | **yes** | **yes** | no | **no** ✗ | yes |
| Head % of params | — | **3 %** | ~4 % | 28 % | **35 %** ✗ | 2 % |
| Box repr. | anchor + wh | DFL reg_max 7 | DFL | DFL reg_max 16 | `exp(wh)` ✗ | `ltrb` (± DFL 7) |
| Assigner | YOLOv5 anchor match | DSLA + AGM | SimOTA | TAL | TAL (crippled) | TAL + STAL |
| Train-only aux head | no | **yes, 1.35 M** | no | no | **no** ✗ | **add** |
| Epochs | — | **300** | 300 | 500 | 100 ✗ | 300 |

Six red marks in the TinyYOLO column. Four of them (objectness, 3×3 kernels, no SPPF,
head-heavy allocation) are each independently sufficient to explain a large accuracy
deficit.

---

## 4. What this comparison changes in the redesign plan

Three revisions to `ARCHITECTURE_REDESIGN.md`:

**Added — depthwise `k=5` throughout neck and head (new, high priority).**
Both sub-1M leaders use it; neither uses SPP. The cost is negligible in a
pointwise-dominated network and it is the cheapest receptive-field purchase
available. Should be promoted to **Step 1b**, alongside SPPF. Doing both is not
redundant: SPPF gives global context at P5, 5×5 gives dense local context everywhere.

**Added — train-only auxiliary head (new, high priority).**
NanoDet-Plus trains with 2.3 M parameters and deploys 0.95 M. A 4-conv aux head with
`detach_epoch` is free at inference and is one of the largest single contributors to
NanoDet-Plus's +7 mAP over NanoDet. Nothing in TinyYOLO resembles this. New **Step 8b**.

**Walked back — D12 (add a P2 / stride-4 level).**
Neither NanoDet nor PicoDet added stride 4; both added stride **64** instead. At 320 px
a P2 map is 80×80 — expensive — and NanoDet's `AP_s = 8.3` shows the leaders did not
solve small objects this way regardless. Revised recommendation: **add stride 64, not
stride 4.** Cheap, lifts AP_l, matches both references.

**Reopened — DFL.** Contested, not settled. Both sub-1M leaders use light DFL
(`reg_max=7`, 32 extra outputs/anchor). YOLO26 removed DFL entirely — but paired that
removal with STAL, ProgLoss, and an end-to-end one-to-one head. Treat `reg_max ∈ {0, 7}`
as a first-class ablation rather than assuming DFL-free is modern.

### Revised sizing

| Variant | Channels | neck | k | reg_max | Params | Backbone % |
|---|---|---:|---:|---:|---:|---:|
| current | 16,24,40,80,160 | 64 | 3 | 0 | 0.218 M | 28 % |
| **v2-s** | 16,24,48,96,192 | 48 | 5 | 0 | **0.494 M** | 84 % |
| **v2-m** | 16,32,64,128,256 | 64 | 5 | 0 | **0.861 M** | 85 % |
| v2-m-dfl | 16,32,64,128,256 | 64 | 5 | 7 | 0.862 M | 84 % |

`v2-m` at 0.86 M is a direct like-for-like against NanoDet-Plus-m (0.95 M / 27.0) and
PicoDet-S (0.99 M / 30.6) — that is the honest benchmark table for a paper. `v2-s` at
0.49 M occupies an under-served gap between YOLO-Fastest-v2 and NanoDet-Plus and is
the more defensible novelty claim.

---

## 5. Realistic targets

| Config | Params | Expected mAP50-95 @320 | Basis |
|---|---:|---:|---|
| current, all bugs fixed | 0.22 M | 10–13 | YOLO-Fastest-v2 parity |
| v2-s | 0.49 M | 18–23 | interpolation, Fastest-v2 → NanoDet |
| v2-m | 0.86 M | 25–29 | NanoDet-Plus parity ±2 |

Beating PicoDet-S at equal size would require NAS on the backbone, which is out of
scope. Matching NanoDet-Plus at 10 % fewer parameters is a credible and publishable
result. Beating YOLOv8n is not achievable at this scale and should not be claimed.

---

## References

- [NanoDet-Plus — RangiLyu/nanodet](https://github.com/RangiLyu/nanodet) · [`nanodet-plus-m_320.yml`](https://github.com/RangiLyu/nanodet/blob/main/config/nanodet-plus-m_320.yml)
- [Yolo-FastestV2 — dog-qiuqiu](https://github.com/dog-qiuqiu/Yolo-FastestV2)
- [PP-PicoDet: Mobile Real-Time Detector (arXiv 2111.00902)](https://www.emergentmind.com/papers/2111.00902) · [PaddleDetection PicoDet configs](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8.1/configs/picodet/README_en.md)
- [Tech Report: One-stage Lightweight Object Detectors (arXiv 2210.17151)](https://arxiv.org/pdf/2210.17151)
- [Ultralytics YOLO26 Docs](https://docs.ultralytics.com/models/yolo26)
