# Peer Review Report

**Manuscript:** "TinyYOLO: Ultra-Lightweight Object Detection for Edge Deployment"
**Review Date:** 2026-05-15
**Reviewer Expertise:** YOLO architectures, lightweight object detection, edge AI, model compression, embedded ML systems

---

## A. Paper Summary

This manuscript presents TinyYOLO, an ultra-lightweight object detection framework targeting edge deployment with a sub-0.3M parameter budget (~0.23M for detection). The architecture follows a three-stage pipeline—Backbone (Ghost-based, ~0.08M), Neck (LitePAN with depthwise separable FPN+PAN, ~0.06M), and Head (decoupled TinyDetect, ~0.09M)—and supports five vision tasks: detection, segmentation, pose estimation, classification, and oriented bounding box detection.

The key contributions claimed are:

1. **Sub-1M multi-task framework** — the first YOLO-family model below 1M parameters supporting five vision tasks
2. **Dual-variant architecture** — a standard variant (SiLU + LightSpatialAttn + SE) and an INT8-native quantized variant (ReLU6 + ECA) designed from the ground up for edge accelerators
3. **CIoU-based training pipeline** — with YOLO-standard BatchNorm, AdamW with separated weight decay, cosine annealing, EMA, and comprehensive per-epoch evaluation
4. **Resolution ablation** — identifying 416×416 as the optimal resolution for the parameter budget
5. **Cross-platform validation** — training on both Google Colab and Kaggle with Tesla T4 GPUs

The claimed results include a peak mAP@50 of 0.353 (quantized variant) and 0.328 (standard variant at 416×416) on COCO128 (128 images, 80 classes), with model sizes under 1 MB and compute requirements of 0.15–0.25 GFLOPs.

---

## B. Major Strengths

### S1. Clearly Defined and Practical Problem Space
The manuscript correctly identifies a genuine gap in the YOLO ecosystem: no existing YOLO variant operates below 1M parameters while supporting multiple vision tasks. The constraint table in Section 1.1 effectively quantifies the deployment gap between YOLOv8n and typical edge hardware limits. The motivation for building up from efficient primitives rather than pruning a large model is well-reasoned and technically sound — aggressive pruning to sub-0.3M from a 3.2M model does indeed destroy learned feature hierarchies.

### S2. INT8-Native Dual-Variant Design
The decision to provide a dedicated quantized architecture variant rather than relying on post-training quantization is architecturally sound and practically valuable. The identification that SiLU quantizes poorly to INT8 (due to the smooth non-monotonic region around zero) while ReLU6's bounded output [0, 6] maps cleanly to INT8 range is correct. The substitution of SE with ECA for the quantized variant (avoiding the FC layers that create quantization bottlenecks) demonstrates genuine understanding of deployment constraints.

### S3. Modular and Well-Engineered Codebase
The source code demonstrates strong software engineering practices. The separation of concerns (backbone/neck/heads as independent modules), the factory pattern in `models.py`, the unified training script with auto-environment detection, and the comprehensive YAML configuration system are all well-executed. The codebase is production-oriented — featuring gradient clipping, AMP support, EMA, separated weight decay groups, and a robust dataset loader supporting multiple formats. This modularity genuinely enables research experimentation.

### S4. Honest Acknowledgment of Limitations
The manuscript explicitly states (Section 1.4, Section 6.7) that TinyYOLO is not competing with full-size YOLOs on accuracy and that COCO128 results are not directly comparable to full COCO benchmarks. This intellectual honesty is commendable and avoids the overclaiming that plagues many lightweight detection papers. The "cherry-pick" framing is refreshingly transparent.

### S5. Comprehensive Training Pipeline with CIoU
The transition from MSE to CIoU loss (Section 6.1) is well-motivated and the 7.1× improvement in best mAP@50 validates the choice. The implementation of CIoU in `train.py` (lines 478–538) is mathematically correct, including the center distance term, enclosing box diagonal, and aspect ratio consistency term. The training recipe (AdamW, cosine annealing, YOLO-standard BatchNorm) follows established best practices.

### S6. Resolution Ablation Study
The systematic resolution sweep across 160–640 (Section 6.5) provides useful guidance on the capacity–resolution trade-off for sub-1M models. The finding that 160/224 are too small for 0.23M parameters, that 416×416 is optimal, and that 640 yields diminishing returns is a practically useful contribution for the edge deployment community.

---

## C. Major Weaknesses

### W1. Fundamentally Insufficient Experimental Evaluation (Critical)
The entire experimental validation is conducted on **COCO128** — a 128-image subset that is universally recognized as a debugging/smoke-test dataset, not a benchmark. No Q1 journal will accept results evaluated solely on 128 images across 80 classes (1.6 images per class). The claimed mAP@50 values (0.328–0.353) are **statistically meaningless** on this dataset: with so few images per class, a single correct or missed detection can swing AP by 20–50 percentage points. The reproducibility section (Section 6.2) actually demonstrates this instability: Run 1 achieves final mAP@50 of 0.0314 while Run 2 achieves 0.1464 — a **4.7× variance** that would be unacceptable in any rigorous evaluation.

The manuscript acknowledges that VOC and full COCO evaluations are "pending" (Section 7.2), but these are **mandatory** for publication, not optional future work. At minimum, Pascal VOC (16.5K images, 20 classes) and COCO val2017 (5K images) evaluations are required.

### W2. No Comparison with Relevant Lightweight SOTA (Critical)
Table 2.2 lists YOLOv3-tiny through YOLO26n, but **all comparisons are on different datasets** (full COCO vs. COCO128), rendering the comparison table meaningless. More critically, the paper entirely omits comparison with the most relevant lightweight competitors:

- **NanoDet** (0.95M params, designed for edge) — the closest competitor in parameter count
- **NanoDet-Plus** — with ShuffleNetV2 backbone
- **PicoDet** (PaddlePaddle) — specifically designed for sub-1M edge deployment
- **YOLO-Fastest** (~0.25M params) — directly comparable in scale
- **MCUNet** — designed for microcontroller deployment
- **MobileDets** — NAS-optimized for mobile
- **PP-YOLOE-S** — lightweight PaddlePaddle detector

Without same-dataset comparisons against these models, the paper cannot establish its claimed contribution of being the first sub-1M multi-task YOLO framework, because YOLO-Fastest already operates at ~0.25M parameters.

### W3. No Real Edge Hardware Evaluation (Critical)
Despite the title claiming "Edge Deployment," **all experiments run on Tesla T4 GPUs** — a data center GPU, not an edge device. The paper provides zero evidence of deployment on:

- Jetson Nano / Orin Nano (NVIDIA edge)
- Raspberry Pi 4/5 (ARM CPU)
- Google Coral Edge TPU
- STM32 or ESP32 microcontrollers
- Any mobile device (Android/iOS)

The latency benchmarks (Section 6.6) showing 30–55 ms on T4 are irrelevant to edge deployment claims. A T4 has 16 GB VRAM and 65W TDP — the operational regime is fundamentally different from a 4W Jetson Nano or a 15W Raspberry Pi. The paper must demonstrate actual INT8 inference latency on at least 2–3 edge platforms to substantiate its title and primary contribution.

### W4. Hardcoded SiLU in Detection Heads Breaks Quantized Variant Design
A critical implementation inconsistency: the detection heads in [heads.py](file:///Applications/XAMPP/xamppfiles/htdocs/tinyYOLO/tinyYOLO/modules/heads.py) hardcode `act='silu'` in all DWConv layers (lines 54–55, 62–63, 165, 225–226), regardless of whether the model is the quantized variant. This means the quantized variant's backbone and neck use ReLU6 (correct), but the **head uses SiLU** (incorrect). Since the head accounts for ~39% of parameters (Section 4.1), the claim that the quantized variant is "fully INT8-safe" is **factually incorrect** in the current implementation. This is a design bug that undermines the core INT8-native contribution.

### W5. Naive Target Assignment Strategy
The loss function in [train.py](file:///Applications/XAMPP/xamppfiles/htdocs/tinyYOLO/scripts/train.py) (lines 540–610) uses a simplistic single-cell target assignment: each ground truth is assigned to exactly one grid cell at each scale based on its center position. This approach has been superseded by:

- **SimOTA** (YOLOX) — optimal transport-based assignment
- **TAL (Task-Aligned Learning)** (YOLOv8) — task-aligned assigner
- **STAL** (YOLO26) — simplified task-aligned assigner

The single-cell assignment is particularly harmful for small models because it provides very sparse supervision signals. With 2,100 total grid cells and typically 5–15 objects per image in COCO128, over 99% of cells receive no positive signal. Modern dynamic label assignment strategies assign multiple positives per ground truth, providing denser gradients that are especially critical for parameter-limited models. This likely explains the persistently low precision (0.03–0.10) and recall (0.04–0.05) observed across all experiments.

### W6. Training and Validation on Identical Data
The code in [train.py](file:///Applications/XAMPP/xamppfiles/htdocs/tinyYOLO/scripts/train.py) (line 731) creates the validation dataset from the **same directory** as training (`val_dataset = SimpleDetectionDataset(train_dir, ...)`). For COCO128, this means the model is evaluated on its own training data — a fundamental methodological error that invalidates all reported metrics. Even the modest mAP@50 values of 0.15–0.35 may be optimistic given this data leakage.

### W7. Incomplete Multi-Task Validation
The manuscript claims support for 5 vision tasks (detection, segmentation, pose, classification, OBB), but **only detection is experimentally validated**. No training results, loss curves, or accuracy metrics are provided for segmentation, pose estimation, classification, or OBB. The forward-pass shape verification performed in the notebooks is necessary but insufficient — it proves the architecture computes outputs of the correct shape, not that it can learn meaningful representations for these tasks. For a multi-task claim to be credible, at least detection and one additional task (e.g., segmentation on COCO-Seg or pose on COCO-Pose) must be experimentally validated with quantitative results.

### W8. Missing Ablation Studies on Architectural Components
The paper provides a resolution ablation (Section 6.5) and a loss function comparison (Section 6.1), but lacks ablations on the core architectural choices:

- **Ghost vs. standard convolutions** — what is the accuracy cost of using Ghost modules?
- **Attention mechanism impact** — SE vs. ECA vs. no attention: what does each contribute?
- **Spatial attention value** — does LightSpatialAttn actually improve mAP, or is it overhead?
- **Neck depth** — is LitePAN necessary, or would a simpler FPN suffice?
- **Channel width sensitivity** — how does the width multiplier affect the accuracy/size trade-off?
- **Decoupled vs. coupled head** — what is the benefit of separate cls/reg branches at this scale?

Without these ablations, the reader cannot assess whether the architectural choices are justified or arbitrary.

---

## D. Detailed Technical Comments

### D1. Architecture Design

The backbone design is sound in principle — Ghost convolutions with staged downsampling and attention at P4/P5 is a well-established pattern. However, the channel progression [16, 24, 40, 80, 160] appears to be chosen heuristically without justification. The 2× expansion from 80→160 at Stage4 is unusually aggressive for a tiny model — the P5 features at 10×10 spatial resolution with 160 channels represent a disproportionate parameter allocation for large object detection, which is arguably less important in edge deployments where the camera is typically close to objects.

The LitePAN neck implementation is clean but employs an unusual merge strategy: `ConvBNAct(out_channel * 2, out_channel, 1, 1)` followed by `DWConv(out_channel, out_channel, 3, 1)`. This uses a pointwise 1×1 for channel reduction after concatenation, then a depthwise separable 3×3 for spatial processing. While functional, this differs from standard PAN implementations that use CSP blocks or C2f modules for feature fusion. The impact of this simplification on multi-scale feature quality is not analyzed.

The `width_mult` parameter in `build_backbone()` enables scaling, but the head does not respect it — `HEAD_KWARGS` for classification hardcodes `in_channel=160`, and neck channel is computed separately. This creates inconsistency when `width_mult != 1.0`.

### D2. Loss Function and Training Methodology

The CIoU implementation is mathematically correct, but the loss computation has a subtle issue: `n_targets_total` is reset inside the per-prediction loop (line 594) but accumulated across scales. If multiple scales contain the same target (which they do — every target is assigned to every scale), the normalization denominator is inflated, diluting the gradient signal. This explains why the box loss plateaus at ~0.86 — the effective learning rate for box regression is too low due to over-normalization.

The objectness proxy (`obj_pred = pred_cls.max(dim=1, keepdim=True)[0]`) is unconventional. Standard YOLO objectness uses a dedicated objectness head, not the maximum class logit. Using max class confidence as objectness creates a coupling between classification and localization that can cause the model to suppress low-confidence but correctly localized predictions.

The loss weights (2.0 × CIoU + 1.0 × BCE_cls + 1.0 × BCE_obj) are stated to follow "pfeatherstone/tinyyolo conventions," but the comment on line 465 says `7.5 × IoU` while the actual implementation uses `2.0`. This inconsistency in the documentation creates confusion about the actual recipe.

### D3. Experimental Protocol

The experimental design has several methodological issues beyond those noted in Weaknesses:

1. **No random seed control**: The training script does not set `torch.manual_seed()`, `np.random.seed()`, or `torch.backends.cudnn.deterministic`. The 4.7× variance between Run 1 and Run 2 mAP@50 (Section 6.2) is likely partly due to non-deterministic operations.

2. **Inconsistent evaluation epochs**: Eval frequency is `max(1, epochs // 10)`, meaning evaluation happens every 10 epochs for 100-epoch runs. The "best mAP@50" metric is therefore sampled at only 10 points, missing potential peaks between evaluation epochs.

3. **No warmup period**: Modern YOLO training uses 3–5 epochs of linear warmup. Training a tiny model with full learning rate from epoch 1 likely causes initial gradient instability, as evidenced by the high initial losses (~2.63).

4. **No mosaic augmentation**: All YOLO variants since v4 use mosaic augmentation, which is particularly beneficial for small datasets. Its absence is surprising and likely contributes to the low mAP.

### D4. Deployment Claims

The manuscript's deployment narrative is aspirational rather than demonstrated. Specific concerns:

- **ONNX export**: The export script exists but no ONNX model size, inference latency, or compatibility testing is reported. The `_clean_state_dict()` function (line 848) suggests there were issues with profiler metadata contaminating exports — this deserves explanation.
- **TensorRT/TFLite**: Listed as "Use Ultralytics export" with no verification that TinyYOLO's custom modules are compatible.
- **INT8 quantization**: No actual INT8 quantization results are presented — no comparison of FP32 vs. INT8 accuracy, no quantization-aware training (despite the YAML configs mentioning QAT settings), no calibration procedure.
- **Memory footprint**: No peak memory usage measurements during inference.
- **Energy efficiency**: No power consumption measurements on any platform.

### D5. Benchmarking Fairness

The comparison table (Section 6.7) places TinyYOLO results on COCO128 alongside YOLOv5n/v8n/YOLO11n results on full COCO. While the paper acknowledges this disparity, presenting them in the same table — even with a disclaimer — is misleading. A reader scanning the table could incorrectly conclude that TinyYOLO at 0.23M parameters achieves mAP comparable to YOLOv5n at 1.9M parameters. The table should either be removed or restructured to clearly separate dataset-specific results.

### D6. Statistical Rigor

With only 2–5 runs per configuration and no reported confidence intervals, standard deviations, or statistical significance tests, the results lack statistical rigor. The claim that "quantized consistently achieves higher mAP@50" (Section 6.3) is based on 2 quantized runs vs. 3 standard runs — insufficient evidence for a statistical claim. The 0.353 peak mAP@50 for the quantized variant could easily be a favorable random initialization on 128 training images.

---

## E. Questions for the Authors

1. **Target assignment strategy**: Why was single-cell assignment chosen over SimOTA or TAL? Have you experimented with multi-positive assignment strategies? Given the extremely sparse positive-to-negative ratio (~0.5% positive cells), dynamic assignment could substantially improve convergence.

2. **Head activation inconsistency**: The detection heads hardcode `act='silu'` regardless of variant. Is this intentional? If so, what is the rationale for mixing SiLU (head) with ReLU6 (backbone/neck) in the quantized variant? How does this affect INT8 inference on actual edge accelerators?

3. **Validation data leakage**: The validation dataset is constructed from the same directory as training data (`train_dir`). For COCO128, this means train=val. Can you confirm that all reported mAP values are effectively training set metrics? What are the results on a held-out validation set?

4. **Scalability to standard benchmarks**: The paper states that VOC and full COCO evaluations are "pending." What preliminary results, if any, exist on these datasets? What is the expected mAP@50 on VOC given the model's 0.23M parameter budget?

5. **Comparison with YOLO-Fastest**: YOLO-Fastest [Y. Wu, 2021] operates at ~0.25M parameters — nearly identical to TinyYOLO. How does TinyYOLO compare on the same dataset and hardware? This is the most directly relevant baseline and its omission is a significant gap.

6. **Multi-task training results**: Have any of the four non-detection tasks (segmentation, pose, classification, OBB) been trained, even on small datasets? What mAP/accuracy values were achieved? Are the task-specific heads (proto-masks, keypoint regression) architecturally validated beyond shape checking?

7. **Quantized variant superiority**: The paper claims the quantized variant achieves 88% higher mAP@50 than the standard variant (Section 6.3). Is this due to ReLU6's bounded activation preventing gradient explosion in tiny models, or is it a statistical artifact of the small sample size (2 vs. 3 runs on 128 images)?

8. **Ghost convolution ratio**: The GhostConv uses a fixed ratio=2 (50% primary, 50% cheap). Has the impact of different ratios (e.g., ratio=3 or ratio=4) been explored? For a 0.23M model, even small efficiency gains from higher ratios could be meaningful.

9. **No warmup or mosaic**: Why were linear LR warmup and mosaic augmentation omitted from the training recipe? Both are standard in all YOLO variants since v4/v5 and are known to improve convergence, especially on small datasets.

10. **Actual INT8 inference**: Can you provide INT8 inference latency on at least one edge device (e.g., Raspberry Pi 4 with ONNX Runtime or Jetson Nano with TensorRT)? Without this, the "edge deployment" claim remains unsubstantiated.

11. **EMA vs. best checkpoint**: The training saves both `best.pt` (by mAP) and `ema.pt`, but all reported results use the best checkpoint. What is the accuracy difference between the EMA model and the best-epoch model?

12. **Per-class analysis depth**: Section 6.4 shows only 2–3 classes with non-zero AP. For a model trained on 80 COCO classes, how many classes achieve AP@50 > 0? Is the model effectively a 3–5 class detector despite being configured for 80 classes?

13. **Augmentation pipeline**: RandomPerspective with `distortion_scale=0.2` can be aggressive for small models. Has the impact of augmentation strength been ablated? Over-augmentation can harm tiny models that lack capacity to learn invariances.

14. **FLOPs measurement methodology**: The paper reports 0.15 GFLOPs at 320×320. Is this measured with `thop`, manual calculation, or another tool? Does it account for all operations including attention mechanisms, or only convolutions?

15. **Width multiplier utility**: The `build_backbone()` function supports `width_mult` for scaling, but no experiments explore widths other than 1.0. Have smaller (0.5×, 0.75×) or larger (1.25×, 1.5×) widths been tested? This could provide a richer accuracy-efficiency trade-off analysis.

---

## F. Required Revisions

### Mandatory Revisions

> [!CAUTION]
> These revisions are **required** before the paper can be considered for publication. Failure to address any of these will result in rejection.

1. **Evaluate on standard benchmarks**: Train and evaluate on Pascal VOC 2007+2012 (at minimum) and COCO val2017. Report mAP@50 and mAP@50-95 with confidence intervals from at least 3 independent runs with fixed random seeds.

2. **Fix train/val data leakage**: Use a proper held-out validation set for all evaluation. For COCO128, use COCO val2017 as the evaluation set. Re-run all experiments with this fix and update all tables.

3. **Add direct SOTA comparisons**: Compare with NanoDet, NanoDet-Plus, PicoDet, YOLO-Fastest, and MCUNet on the **same dataset and hardware**. Use their official pretrained models or retrain at comparable settings.

4. **Fix head activation bug**: Pass the variant-appropriate activation to all head DWConv layers. Verify that the quantized variant uses ReLU6 end-to-end. Report any accuracy impact of this fix.

5. **Demonstrate real edge deployment**: Measure inference latency and accuracy on at least two edge platforms (e.g., Raspberry Pi 4 + Jetson Nano). Report FPS, latency, peak memory, and INT8 vs. FP32 accuracy comparison.

6. **Add architectural ablation studies**: Provide ablations for: (a) Ghost vs. standard conv, (b) attention mechanism impact (SE/ECA/Spatial/None), (c) neck design (LitePAN vs. simple FPN), (d) channel width sensitivity.

7. **Set random seeds and report variance**: Set deterministic training (seed, cudnn deterministic) and report mean ± std across at least 3 runs for all key metrics.

8. **Validate at least one additional task**: Train and evaluate segmentation or pose estimation with quantitative results to substantiate the multi-task claim.

### Minor Revisions

> [!NOTE]
> These revisions improve quality but are not individually blocking.

1. **Implement modern target assignment**: Replace single-cell assignment with TAL or SimOTA. Report the accuracy impact as an ablation.

2. **Add LR warmup**: Implement 3–5 epoch linear warmup, standard in all modern YOLO training recipes.

3. **Add mosaic augmentation**: Implement mosaic/mixup augmentation, particularly critical for small dataset training.

4. **Clarify loss weight documentation**: The comment on line 465 (`7.5 × IoU`) contradicts the actual implementation (`2.0 × CIoU`). Clarify and justify the chosen weights.

5. **Improve Table 6.7**: Either remove the mixed-dataset comparison table or restructure it with clear visual separation and prominent disclaimers. Consider using separate sub-tables.

6. **Add FLOPs breakdown**: Report per-component FLOPs (backbone, neck, head) alongside the parameter breakdown in Section 4.1.

7. **Report ONNX model sizes**: Export to ONNX and TFLite, report file sizes, and verify inference compatibility.

8. **Add training curves for best runs**: Include actual training loss and mAP curves as figures (not just tabular data) to show convergence behavior.

9. **Expand related work**: Add dedicated subsections for Ghost-based detectors, NAS-based lightweight models, and knowledge distillation approaches for small detectors.

10. **Proofread equation formatting**: Section 5.1 presents CIoU as inline code rather than proper LaTeX equations. Use proper mathematical notation with numbered equations.

11. **Add a dedicated objectness head**: Replace the max-class-logit proxy with a proper objectness prediction branch, as used in YOLOX and standard YOLO implementations.

---

## G. Recommendation

### **Major Revision**

### Justification

The manuscript addresses a genuine and practically important problem — ultra-lightweight object detection for resource-constrained edge devices. The dual-variant INT8-native design philosophy is novel and well-motivated, the codebase demonstrates strong engineering quality, and the modular architecture enables meaningful research experimentation. These elements represent a foundation with real potential.

However, the current submission has **critical experimental deficiencies** that preclude publication in a Q1 venue:

1. **The evaluation dataset (COCO128, 128 images) is fundamentally insufficient** for any scientific claim. The high variance between runs (4.7× in mAP@50) confirms that results on this dataset are statistically unreliable. Standard benchmarks (VOC, COCO val2017) are mandatory.

2. **The absence of direct comparisons with relevant SOTA** (NanoDet, PicoDet, YOLO-Fastest) means the paper's positioning claim — that it fills a gap no existing model addresses — is unsubstantiated. YOLO-Fastest at ~0.25M parameters is a direct competitor that must be addressed.

3. **The title and framing promise "Edge Deployment" but deliver only GPU benchmarks.** A paper claiming edge deployment without any edge hardware results would be rejected by any serious venue in the embedded systems or edge AI domain.

4. **The train=val data leakage** invalidates all currently reported metrics. This must be fixed before any results can be trusted.

5. **A critical implementation bug** (SiLU hardcoded in heads for the quantized variant) contradicts the core INT8-safety claim.

These issues are individually serious and collectively constitute grounds for rejection. However, the underlying work has sufficient merit that a thorough revision addressing all mandatory items could produce a publishable contribution. The framework's modular design, the INT8-native architecture concept, and the systematic approach to multi-task support at extreme parameter budgets are all valuable — they simply require proper experimental validation to support the claims.

I recommend **Major Revision** with the expectation that the authors will:
- Conduct full VOC and COCO evaluations
- Benchmark on real edge hardware
- Fix the implementation bugs
- Add proper SOTA comparisons
- Provide the missing ablation studies

If these are addressed comprehensively, the paper could be suitable for a Q1 venue in a subsequent round.

---

*End of Review*
