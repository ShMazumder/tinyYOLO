# TinyYOLO: Ultra-Lightweight Object Detection for Edge Deployment via Ghost-Based Architecture and INT8-Native Design

## Revised Manuscript — Part 1: Abstract, Introduction, Related Work

---

## Abstract

Deploying object detection on resource-constrained edge devices — microcontrollers, mobile SoCs, and single-board computers operating under sub-1 MB memory and sub-0.5 GFLOP compute budgets — remains an open challenge. Existing YOLO-family detectors, even at their smallest official configurations (e.g., YOLOv8n at 3.2M parameters, 8.7 GFLOPs), exceed these constraints by an order of magnitude, while purpose-built lightweight detectors such as NanoDet and PicoDet typically operate in the 0.9–1.5M parameter range without providing native INT8 quantization compatibility or multi-task extensibility.

This paper presents TinyYOLO, a 0.23M-parameter object detection framework constructed from Ghost convolutions, depthwise separable feature fusion (LitePAN), and decoupled anchor-free detection heads. TinyYOLO introduces a dual-variant architecture: a *standard* variant employing SiLU activations with SE and spatial attention for FP32/FP16 deployment, and a *quantized* variant replacing all activations with ReLU6 and all attention modules with ECA blocks to guarantee end-to-end INT8 compatibility on edge accelerators. The framework provides architectural support for five vision tasks — detection, instance segmentation, pose estimation, image classification, and oriented bounding box detection — through task-specific heads sharing a common 0.08M-parameter backbone, with quantitative validation provided for detection, segmentation, and pose estimation.

We evaluate TinyYOLO on Pascal VOC 2007+2012 (16.5K images, 20 classes) and COCO val2017 (5K images, 80 classes) under controlled experimental conditions with fixed random seeds and deterministic training. On VOC, TinyYOLO achieves mAP@50 of 38.7 ± 0.9% (standard) and 41.2 ± 0.7% (quantized) at 416×416 resolution, with INT8 inference latencies of 28.3 ms on Jetson Nano (TensorRT) and 67.4 ms on Raspberry Pi 4 (TFLite). Comprehensive ablation studies validate each architectural component, and direct comparisons against NanoDet (0.95M), YOLO-Fastest (0.25M), PicoDet-XS (0.93M), and MCUNetV2 (0.74M) on identical hardware establish TinyYOLO's position within the accuracy–efficiency Pareto frontier for sub-1M detectors. COCO comparisons use official baseline numbers; VOC comparisons are author-reproduced under identical conditions and clearly attributed. To our knowledge, TinyYOLO is the smallest YOLO-family framework supporting five vision tasks with a dedicated INT8-native architecture variant, though we note that YOLO-Fastest achieves comparable parameter counts for single-task detection.

**Keywords:** lightweight object detection, edge deployment, Ghost convolution, INT8 quantization, YOLO, anchor-free detection, depthwise separable convolution

---

## 1. Introduction

### 1.1 Motivation and Problem Statement

The proliferation of edge computing platforms — from NVIDIA Jetson modules and Google Coral Edge TPUs to ARM Cortex-M microcontrollers and mobile neural processing units (NPUs) — has created unprecedented demand for object detection models that operate within severe resource constraints. Industrial inspection systems, autonomous micro-drones, wearable medical devices, and agricultural monitoring sensors share a common requirement: real-time visual understanding under power budgets of 1–15W, memory limits of 0.5–4 MB, and compute ceilings of 0.1–1.0 GFLOPs [1, 2].

The YOLO family of detectors has emerged as the dominant paradigm for real-time object detection, with successive generations (v3 through v12, YOLO11, YOLO26) achieving progressively better accuracy–speed trade-offs [3–13]. However, even the smallest official configurations of modern YOLO variants remain impractical for genuine edge deployment:

| Constraint | Typical Edge Limit | YOLOv8n | YOLO11n | YOLO26n | Gap Factor |
|---|---|---|---|---|---|
| Parameters | < 500K | 3.2M | 2.6M | 1.7M | 3.4–6.4× |
| Model Size | < 1 MB (INT8) | 6.3 MB | 5.4 MB | 3.5 MB | 3.5–6.3× |
| Compute | < 0.5 GFLOPs | 8.7 | 6.5 | 5.2 | 10–17× |
| INT8 Native | Required | No | No | No | Unsupported |

This gap is not merely quantitative. Post-training quantization (PTQ) of models designed for FP32 inference introduces systematic accuracy degradation, particularly in architectures employing SiLU activations whose smooth non-monotonic region near zero maps poorly to the discrete INT8 representation [14, 15]. Furthermore, aggressive pruning or knowledge distillation applied to 1.7–3.2M parameter models to reach sub-0.5M targets typically destroys learned feature hierarchies, as the remaining capacity is insufficient to preserve the representational structure of the original network [16, 17].

### 1.2 Approach: Building Up from Efficient Primitives

Rather than compressing a large model downward, TinyYOLO constructs a detection framework *upward* from primitives specifically chosen for parameter efficiency and quantization compatibility:

1. **Ghost Convolutions** [18] exploit the observation that approximately 50% of feature maps in trained CNNs are near-linear transforms of others. By generating half the output features through standard convolution and the remainder via cheap depthwise operations, Ghost modules halve computational cost with minimal representational loss.

2. **Depthwise Separable Convolutions** [19] factorize standard convolutions into spatial and channel-wise components, reducing the parameter count of the neck (LitePAN) by approximately 8× compared to standard PAN implementations.

3. **Dual-Variant Activation Design** addresses the quantization problem at the architectural level rather than as a post-hoc optimization. The quantized variant exclusively employs ReLU6 (bounded output in [0, 6] mapping cleanly to INT8 range [0, 255]) and ECA attention (1D convolution replacing the FC layers in SE that create quantization bottlenecks).

4. **Task-Aligned Label Assignment (TAL)** [20] replaces naive single-cell target assignment with dynamic multi-positive supervision, providing denser gradient signals critical for convergence in parameter-limited models.

### 1.3 Contributions

This work makes the following contributions, each validated experimentally:

1. **A sub-0.25M parameter multi-task-capable detection framework** providing architectural support for detection (0.23M), segmentation (0.29M), pose estimation (0.27M), classification (0.10M), and OBB detection (0.24M) through modular task-specific heads sharing a Ghost-based backbone. We note that YOLO-Fastest [21] achieves comparable parameter counts (~0.25M) for single-task detection; TinyYOLO's distinguishing contribution is the framework's multi-task extensibility under this extreme budget, with representative tasks (detection, segmentation, pose) validated in this work.

2. **An INT8-native dual-variant architecture** providing a dedicated quantized model (ReLU6 + ECA end-to-end, including detection heads) validated through actual INT8 inference on Jetson Nano (TensorRT) and Raspberry Pi 4 (TFLite), with accuracy degradation below 1.5% relative to FP32 baselines.

3. **Systematic experimental validation** on Pascal VOC 2007+2012 and COCO val2017, with direct same-dataset, same-hardware comparisons against NanoDet, NanoDet-Plus, PicoDet-XS, YOLO-Fastest, and MCUNet, establishing accuracy–efficiency positioning within the sub-1M detector landscape.

4. **Comprehensive ablation studies** isolating the contribution of each architectural decision: Ghost vs. standard convolutions, SE vs. ECA vs. no attention, LitePAN vs. simple FPN, ReLU6 vs. SiLU, width multiplier scaling, TAL vs. single-cell assignment, and QAT vs. PTQ quantization strategies.

### 1.4 Scope and Limitations

TinyYOLO is not designed to compete with full-size YOLO variants on absolute accuracy. A 0.23M parameter model cannot match the representational capacity of a 3.2M parameter network. Instead, this work targets a specific and practically important deployment regime where model size, compute, and power constraints preclude the use of standard detectors, and where INT8 inference compatibility is a hard requirement rather than an optional optimization.

We acknowledge several limitations upfront: (i) performance on COCO's 80-class benchmark is substantially lower than on VOC's 20 classes, reflecting the capacity constraints of the architecture; (ii) multi-task validation beyond detection is demonstrated for segmentation and pose estimation but not exhaustively for all five tasks; (iii) edge deployment is validated on two platforms (Jetson Nano, Raspberry Pi 4) and does not cover microcontroller-class devices (e.g., STM32, ESP32) where further optimization would be required.

### 1.5 Paper Organization

The remainder of this paper is organized as follows. Section 2 surveys related work in lightweight detection, Ghost-based architectures, and quantization-aware design. Section 3 details the TinyYOLO architecture. Section 4 describes the training methodology including TAL assignment and CIoU loss formulation. Section 5 presents the experimental setup. Section 6 reports results including ablation studies and edge deployment benchmarks. Section 7 discusses limitations and future directions. Section 8 concludes.

---

## 2. Related Work

### 2.1 Evolution of YOLO Architectures

The YOLO (You Only Look Once) paradigm, introduced by Redmon et al. [3], reframed object detection as a single-pass regression problem, enabling real-time inference. Successive iterations have introduced anchor-based multi-scale detection (YOLOv2/v3 [4, 5]), cross-stage partial connections and mosaic augmentation (YOLOv4 [6]), anchor-free decoupled heads (YOLOX [7]), efficient reparameterization (YOLOv6 [8]), E-ELAN feature aggregation (YOLOv7 [9]), C2f modules with TAL assignment (YOLOv8 [10]), NMS-free dual-head design (YOLOv10 [11]), and area-attention mechanisms (YOLO12 [12]). Most recently, YOLO26 [13] introduced hardware-friendly design principles eliminating DFL (Distribution Focal Loss) in favor of direct regression and employing Simplified TAL (STAL) for efficient label assignment.

Despite this progression toward efficiency, even the smallest official configurations of these architectures (the "nano" variants) maintain parameter counts of 1.7–3.2M, compute requirements of 5.2–8.7 GFLOPs, and reliance on SiLU activations incompatible with INT8 quantization. No official YOLO variant provides a sub-1M parameter configuration or an architecture designed natively for INT8 deployment.

### 2.2 Lightweight Object Detectors

Several purpose-built lightweight detectors target the sub-2M parameter regime:

**NanoDet / NanoDet-Plus** [22, 23]: Built on ShuffleNetV2 backbones with FCOS-style anchor-free heads, NanoDet achieves 20.6% mAP@50-95 on COCO at 0.95M parameters. NanoDet-Plus adds auxiliary modules and achieves 27.0% mAP@50-95 at 1.17M parameters. Both employ GeneralizedFocalLoss for dense supervision. NanoDet represents the closest competitor to TinyYOLO in the sub-1M regime, though it lacks multi-task extensibility and INT8-native design.

**PicoDet** [24]: Developed within PaddlePaddle, PicoDet-XS operates at 0.93M parameters with ESNet backbone and CSP-PAN neck, achieving 26.2% mAP@50-95 on COCO. PicoDet employs neural architecture search (NAS) to optimize block configurations, representing a complementary approach to TinyYOLO's manual architecture design.

**YOLO-Fastest** [21]: At approximately 0.25M parameters with 0.23 GFLOPs, YOLO-Fastest is the most directly comparable model in terms of scale. It employs light squeeze-and-excitation modules with an anchor-based detection paradigm. TinyYOLO's parameter count (0.23M) is comparable, but differs in providing anchor-free detection, multi-task support, and INT8-native architecture.

**MCUNet** [25, 26]: MCUNet v1 [25] uses neural architecture search to optimize classification models for microcontroller deployment under 256 KB memory constraints, achieving 61.8% top-1 accuracy on ImageNet at 0.74M parameters. MCUNetV2 [26] extends this to object detection, achieving 64.6% mAP on VOC2007 under the same SRAM constraints. MCUNet v1 is classification-only and should not be used as a detection baseline; MCUNetV2 is the appropriate comparison for detection tasks.

**MobileDets** [27]: NAS-optimized detectors for mobile deployment, achieving state-of-the-art accuracy–latency trade-offs on mobile CPUs, DSPs, and Edge TPUs. MobileDets demonstrate that NAS can produce architectures outperforming manually designed mobile backbones, though at higher parameter counts (3–5M) than TinyYOLO's target range.

**PP-YOLOE** [28]: PaddlePaddle's anchor-free detector with RepVGG backbone. The smallest variant (PP-YOLOE-S) operates at 7.9M parameters, well above TinyYOLO's target but included for completeness as a lightweight YOLO-inspired alternative.

### 2.3 Ghost-Based Architectures

GhostNet [18] demonstrated that redundancy in CNN feature maps could be exploited for efficiency. The Ghost module generates a subset of feature maps through standard convolution, then produces complementary features via cheap depthwise linear operations. GhostNetV2 [29] extended this with a decoupled fully connected attention mechanism that captures long-range dependencies without the computational overhead of self-attention.

Ghost modules have been integrated into detection architectures primarily through backbone substitution — replacing standard convolution blocks in existing detectors with Ghost equivalents [30, 31]. TinyYOLO extends this approach by employing Ghost convolutions throughout the backbone while using depthwise separable convolutions in the neck and head, creating a heterogeneous efficiency pipeline optimized for each component's computational profile.

### 2.4 Quantization-Aware Architecture Design

Post-training quantization (PTQ) applies quantization after training using a small calibration dataset, but can introduce significant accuracy degradation in models not designed for quantized inference [14, 32]. Quantization-aware training (QAT) inserts simulated quantization operations (fake quantization nodes) during training, allowing the model to adapt its weight distributions to quantized representations [33, 34].

The choice of activation function critically impacts quantization robustness. SiLU (Swish), defined as $f(x) = x \cdot \sigma(x)$, produces unbounded positive outputs and a smooth non-monotonic region near zero where small input perturbations cause disproportionate output changes — problematic for INT8's 256-level discretization [15]. ReLU6, defined as $f(x) = \min(\max(0, x), 6)$, bounds outputs to [0, 6], enabling a clean linear mapping to INT8 range with uniform quantization step size $\Delta = 6/255 \approx 0.0235$ [35].

Similarly, attention mechanisms differ in quantization friendliness. SE blocks [36] employ FC layers that create internal bottleneck dimensions where quantization error accumulates through two successive linear projections. ECA blocks [37] replace FC layers with a single 1D convolution, eliminating the bottleneck and reducing quantization-sensitive operations.

TinyYOLO's dual-variant approach is, to our knowledge, the first YOLO-family architecture providing a dedicated quantized variant designed from the ground up for INT8 inference, rather than relying on post-hoc quantization of an FP32-optimized design.

### 2.5 Knowledge Distillation for Small Detectors

Knowledge distillation (KD) [38] transfers learned representations from a large teacher model to a smaller student. For object detection, feature-level distillation [39, 40] and logit-level distillation [41] have shown effectiveness in improving small model accuracy. While TinyYOLO does not employ KD in its current training pipeline, the modular architecture is compatible with feature-based distillation from any YOLO-family teacher, and we identify this as a promising direction for future accuracy improvement (Section 7).

### 2.6 Label Assignment Strategies

Modern detectors have moved from static, hand-crafted assignment rules to dynamic, training-aware strategies. OTA [42] formulates label assignment as optimal transport. SimOTA [7] provides a simplified approximation used in YOLOX. TAL (Task-Aligned Learning) [20], adopted in YOLOv8, scores candidate assignments based on a joint metric of classification confidence and localization quality, providing task-aligned dense supervision. STAL [13] simplifies TAL for hardware-efficient deployment.

For parameter-limited models, assignment strategy has outsized impact: with fewer parameters to distribute across the prediction grid, dense positive supervision from multi-positive assignment is critical for gradient signal quality and convergence stability [7, 20].

---

*End of Part 1*
