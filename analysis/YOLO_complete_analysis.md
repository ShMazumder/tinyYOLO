# YOLO Complete Analysis: v1 → v26

## 1. The YOLO Timeline

```mermaid
gantt
    title YOLO Version Timeline
    dateFormat YYYY-MM
    axisFormat %Y
    section Foundational
        YOLOv1 (Redmon)        :2015-06, 2016-06
        YOLOv2 / YOLO9000      :2016-12, 2017-12
        YOLOv3                  :2018-04, 2019-06
    section Modular Era
        YOLOv4 (Bochkovskiy)   :2020-04, 2020-12
        YOLOv5 (Ultralytics)   :2020-06, 2022-01
        YOLOX (Megvii)          :2021-07, 2022-01
        YOLOR                   :2021-05, 2022-01
        YOLOv6 (Meituan)       :2022-06, 2022-12
        YOLOv7 (WongKinYiu)    :2022-07, 2023-01
    section Modern Era
        YOLOv8 (Ultralytics)   :2023-01, 2023-12
        YOLOv9 (PGI+GELAN)     :2024-02, 2024-06
        YOLOv10 (NMS-Free)     :2024-05, 2024-09
        YOLO11 (Ultralytics)   :2024-09, 2025-01
        YOLOv12 (Attention)    :2025-02, 2025-08
        YOLO26 (Edge E2E)      :2026-01, 2026-05
```

---

## 2. Version-by-Version Deep Dive

### Phase 1: Foundational (2015–2018)

#### YOLOv1 (2015) — *"You Only Look Once"*
| Attribute | Value |
|-----------|-------|
| **Author** | Joseph Redmon et al. |
| **Backbone** | Custom 24-layer CNN (Darknet) |
| **Detection** | Grid-based regression (7×7 grid, 2 boxes/cell) |
| **mAP (VOC)** | ~63.4% |
| **Params** | ~45M |
| **Speed** | 45 FPS (Titan X) |

**Key Innovations:**
- First single-shot detector — unified detection as regression
- Processes entire image in one forward pass (vs. R-CNN's region proposals)
- Global context reasoning

**Limitations:**
- Poor small object detection (coarse 7×7 grid)
- Max 2 objects per grid cell
- High localization error

---

#### YOLOv2 / YOLO9000 (2016) — *"Better, Faster, Stronger"*
| Attribute | Value |
|-----------|-------|
| **Backbone** | Darknet-19 (19 conv + 5 maxpool) |
| **Detection** | Anchor boxes (5 priors via k-means) |
| **mAP (VOC)** | ~78.6% |
| **mAP (COCO)** | ~21.6% |
| **Params** | ~50M |
| **Speed** | 67 FPS |

**Key Innovations:**
- Batch Normalization on all conv layers (+2% mAP)
- Anchor boxes via k-means clustering
- Multi-scale training (random resize 320–608)
- Passthrough layer for fine-grained features
- WordTree for hierarchical 9000-class detection

---

#### YOLOv3 (2018) — *"An Incremental Improvement"*
| Attribute | Value |
|-----------|-------|
| **Backbone** | Darknet-53 (53 conv layers + residual connections) |
| **Neck** | FPN (Feature Pyramid Network) |
| **Detection** | 3-scale prediction (13×13, 26×26, 52×52) |
| **mAP (COCO)** | ~33.0% (mAP@0.5: 57.9%) |
| **Params** | ~62M |
| **GFLOPs** | ~140 |

**Key Innovations:**
- Residual connections in backbone (from ResNet)
- Multi-scale detection via FPN — major improvement for small objects
- Independent logistic classifiers (multi-label per box)
- 9 anchor boxes (3 per scale)

---

### Phase 2: Modular & Community Era (2020–2022)

#### YOLOv4 (2020) — *"Optimal Speed and Accuracy"*
| Attribute | Value |
|-----------|-------|
| **Author** | Alexey Bochkovskiy et al. |
| **Backbone** | CSPDarknet53 |
| **Neck** | SPP + PANet |
| **mAP (COCO)** | ~43.5% |
| **Params** | ~64M |
| **GFLOPs** | ~120 |

**Key Innovations:**
- **Bag of Freebies** (training tricks): Mosaic augmentation, CutMix, DropBlock, label smoothing
- **Bag of Specials** (architecture): SPP, SAM, PAN, Mish activation
- CSP (Cross Stage Partial) connections for gradient flow
- Self-adversarial training

---

#### YOLOv5 (2020) — *"Production-Ready YOLO"*
| Variant | mAP@50-95 | Params (M) | GFLOPs |
|---------|-----------|------------|--------|
| v5n | 28.0% | 1.9 | 4.5 |
| v5s | 37.4% | 7.2 | 16.5 |
| v5m | 45.4% | 21.2 | 48.3 |
| v5l | 49.0% | 46.5 | 109.1 |
| v5x | 50.7% | 86.7 | 205.7 |

**Key Innovations:**
- First PyTorch-native YOLO (Ultralytics)
- Auto-anchor learning
- Standardized model family (n/s/m/l/x)
- Hyperparameter evolution
- Built-in export (ONNX, TensorRT, CoreML, TFLite)
- Massive community adoption

---

#### YOLOX (2021) — *Megvii*
**Key Innovations:**
- Anchor-free detection head
- Decoupled head (separate cls/reg branches)
- SimOTA dynamic label assignment
- Strong data augmentation (MixUp + Mosaic)

#### YOLOR (2021) — *Unified Representation*
**Key Innovation:** Implicit + explicit knowledge integration in a single network.

#### YOLOv6 (2022) — *Meituan*
| Variant | mAP@50-95 | Params (M) | GFLOPs |
|---------|-----------|------------|--------|
| v6n | 37.5% | 4.7 | 11.4 |
| v6s | 45.0% | 12.5 | 28.6 |

**Key Innovations:**
- EfficientRep backbone
- Rep-PAN neck
- Task Alignment Learning (TAL)
- Self-distillation strategy
- Industry/hardware-aware design

---

#### YOLOv7 (2022) — *WongKinYiu*
| Attribute | Value |
|-----------|-------|
| **mAP (COCO)** | ~51.4% (at 640px) |
| **Key Module** | E-ELAN (Extended ELAN) |

**Key Innovations:**
- E-ELAN: expand/shuffle/merge for gradient diversity
- Compound model scaling (depth + width + resolution)
- Planned re-parameterized convolution
- Coarse-to-fine lead head guided label assigner

---

### Phase 3: Modern Architectural Innovations (2023–2026)

#### YOLOv8 (2023) — *Ultralytics Unified Framework*
| Variant | mAP@50-95 | Params (M) | GFLOPs |
|---------|-----------|------------|--------|
| v8n | 37.3% | 3.2 | 8.7 |
| v8s | 44.9% | 11.2 | 28.7 |
| v8m | 50.2% | 25.9 | 78.9 |
| v8l | 52.9% | 43.7 | 165.2 |
| v8x | 53.9% | 68.2 | 257.8 |

**Key Innovations:**
- Anchor-free detection head
- C2f module (Cross-Stage Partial with 2 convolutions + flow)
- Unified multi-task: Detection, Segmentation, Pose, Classification, OBB
- Distribution Focal Loss (DFL) for box regression
- TaskAlignedAssigner

---

#### YOLOv9 (Feb 2024) — *"Information Retention"*
| Variant | mAP@50-95 | Params (M) | GFLOPs |
|---------|-----------|------------|--------|
| v9t | ~38.3% | 2.0 | 7.7 |
| v9s | ~46.8% | 7.3 | 26.8 |

**Key Innovations:**
- **PGI** (Programmable Gradient Information): Combats information loss during downsampling
- **GELAN** (Generalized ELAN): Combines CSPNet + ELAN strengths
- Auxiliary reversible branch for training
- Superior parameter utilization

---

#### YOLOv10 (May 2024) — *"End-to-End NMS-Free"*
| Variant | mAP@50-95 | Params (M) | GFLOPs |
|---------|-----------|------------|--------|
| v10n | 38.5% | 2.3 | 5.9 |
| v10s | 44.3% | 7.2 | 17.2 |

**Key Innovations:**
- **NMS-Free** inference via consistent dual assignments
- Rank-guided block design (removes redundancy)
- Spatial-channel decoupled downsampling
- Lightweight classification head
- Large-kernel convolution in deeper stages

---

#### YOLO11 (Sept 2024) — *"Refined Production"*
| Variant | mAP@50-95 | Params (M) | GFLOPs |
|---------|-----------|------------|--------|
| 11n | 39.5% | 2.6 | 6.6 |
| 11s | 47.0% | 9.4 | 21.7 |
| 11m | 51.5% | 20.1 | 68.0 |
| 11l | 53.4% | 25.3 | 86.9 |

**Key Innovations:**
- **C3k2 Block**: Optimized CSP with smaller kernels → fewer FLOPs
- **C2PSA** (Cross-Stage Partial Spatial Attention)
- Better small/occluded object detection
- Refined backbone/neck architecture
- Full multi-task support (Det/Seg/Pose/OBB/Cls)

---

#### YOLOv12 (Feb 2025) — *"Attention-Centric YOLO"*
| Variant | mAP@50-95 | Params (M) | GFLOPs |
|---------|-----------|------------|--------|
| v12n | 40.6% | 2.6 | 6.5 |
| v12s | 48.0% | 9.3 | 21.5 |

**Key Innovations:**
- **Area Attention**: Local-to-global attention with reduced cost vs. full self-attention
- **R-ELAN** (Residual ELAN): Residual connections + scaling factors for stability
- **FlashAttention** integration for memory efficiency
- 7×7 separable conv replaces positional encoding
- Adjusted MLP ratio for attention/FFN balance
- CNN-speed with Transformer-quality representations

> [!IMPORTANT]
> YOLOv12 is the first YOLO to fundamentally shift from pure CNN to attention-centric design while maintaining real-time speed.

---

#### YOLO26 (Jan 2026) — *"Deployment-First Edge YOLO"*
| Variant | mAP@50-95 | Params (M) | GFLOPs |
|---------|-----------|------------|--------|
| 26n | 39.8% (40.3 e2e) | 1.7 | 2.4 |
| 26s | 47.2% (47.6 e2e) | 2.7 | 9.5 |
| 26m | 51.5% (51.7 e2e) | 4.9 | 20.4 |
| 26l | 53.0% | 6.5 | 24.8 |

**Key Innovations:**
- **Native NMS-Free** end-to-end inference
- **Removed DFL** → lighter parameterization, easier export
- **MuSGD Optimizer**: SGD + Muon hybrid for faster convergence
- **ProgLoss**: Progressive loss balancing during training
- **STAL**: Small-Target-Aware Label Assignment
- Up to **43% faster CPU inference** vs YOLO11n
- Full multi-task: Det/Seg/Pose/OBB/Cls

---

## 3. Grand Comparison Table (Nano/Small Variants)

| Model | Year | Params (M) | GFLOPs | mAP@50-95 | NMS-Free | Attention | Multi-Task |
|-------|------|-----------|--------|-----------|----------|-----------|------------|
| YOLOv5n | 2020 | 1.9 | 4.5 | 28.0% | ❌ | ❌ | ❌ |
| YOLOv5s | 2020 | 7.2 | 16.5 | 37.4% | ❌ | ❌ | ❌ |
| YOLOv6n | 2022 | 4.7 | 11.4 | 37.5% | ❌ | ❌ | ❌ |
| YOLOv8n | 2023 | 3.2 | 8.7 | 37.3% | ❌ | ❌ | ✅ |
| YOLOv8s | 2023 | 11.2 | 28.7 | 44.9% | ❌ | ❌ | ✅ |
| YOLOv9t | 2024 | 2.0 | 7.7 | 38.3% | ❌ | ❌ | ✅ |
| YOLOv10n | 2024 | 2.3 | 5.9 | 38.5% | ✅ | ❌ | ❌ |
| YOLO11n | 2024 | 2.6 | 6.6 | 39.5% | ❌ | Partial | ✅ |
| YOLO11s | 2024 | 9.4 | 21.7 | 47.0% | ❌ | Partial | ✅ |
| YOLOv12n | 2025 | 2.6 | 6.5 | 40.6% | ❌ | ✅ | ✅ |
| YOLOv12s | 2025 | 9.3 | 21.5 | 48.0% | ❌ | ✅ | ✅ |
| **YOLO26n** | **2026** | **1.7** | **2.4** | **39.8%** | **✅** | ❌ | **✅** |
| **YOLO26s** | **2026** | **2.7** | **9.5** | **47.2%** | **✅** | ❌ | **✅** |

---

## 4. Efficiency Evolution (mAP per Parameter)

```mermaid
xychart-beta
    title "Efficiency: mAP@50-95 vs Parameters (Nano variants)"
    x-axis ["v5n","v6n","v8n","v9t","v10n","11n","v12n","26n"]
    y-axis "mAP / M params" 0 --> 25
    bar [14.7, 8.0, 11.7, 19.2, 16.7, 15.2, 15.6, 23.4]
```

> [!TIP]
> YOLO26n achieves the highest mAP-per-parameter ratio at **23.4 mAP/M**, followed by YOLOv9t at **19.2**. This indicates YOLO26 has the most efficient parameter utilization.

---

## 5. Key Innovations Timeline

| Innovation | First Introduced | Impact |
|-----------|-----------------|--------|
| Single-shot detection | v1 (2015) | Paradigm shift from R-CNN |
| Anchor boxes | v2 (2016) | Better localization |
| Batch Normalization | v2 (2016) | Training stability |
| Multi-scale FPN | v3 (2018) | Small object detection |
| Residual connections | v3 (2018) | Deeper networks |
| CSP connections | v4 (2020) | Gradient flow optimization |
| Mosaic augmentation | v4 (2020) | Better generalization |
| Model family (n/s/m/l/x) | v5 (2020) | Scalability |
| Anchor-free detection | YOLOX (2021) | Simpler pipeline |
| Decoupled head | YOLOX (2021) | Better cls/reg separation |
| E-ELAN | v7 (2022) | Gradient diversity |
| Distribution Focal Loss | v8 (2023) | Better box regression |
| Unified multi-task | v8 (2023) | One model, many tasks |
| PGI + GELAN | v9 (2024) | Information retention |
| NMS-Free inference | v10 (2024) | Latency reduction |
| C2PSA attention | YOLO11 (2024) | Spatial awareness |
| Area Attention | v12 (2025) | CNN+Transformer fusion |
| R-ELAN | v12 (2025) | Stable attention training |
| MuSGD optimizer | YOLO26 (2026) | Faster convergence |
| STAL label assignment | YOLO26 (2026) | Small target accuracy |
| DFL removal | YOLO26 (2026) | Edge deployment simplicity |

---

## 6. Best & Worst Use Cases by Version

### Best Model for Each Scenario

| Scenario | Best Choice | Why |
|----------|-------------|-----|
| **Edge/Mobile (CPU)** | YOLO26n | Lowest FLOPs (2.4B), NMS-free, 43% faster CPU |
| **Real-time GPU** | YOLO11s/YOLOv12s | Best mAP at moderate compute |
| **Maximum accuracy** | YOLOv12x / YOLO11x | Attention mechanisms boost ceiling |
| **Small objects** | YOLO26 (STAL) / YOLOv12 | Specialized small-target mechanisms |
| **Crowded scenes** | YOLOv12 (Area Attention) | Global context via attention |
| **Multi-task (Seg+Pose+OBB)** | YOLO11 / YOLO26 | Native unified support |
| **Easy deployment/export** | YOLO26 | No NMS, no DFL, clean graph |
| **Research/experimentation** | YOLOv9 (PGI) | Novel gradient information concepts |
| **Legacy/compatibility** | YOLOv5/v8 | Massive ecosystem & community |
| **Autonomous vehicles** | YOLO11l / YOLO26m | Balance of speed + accuracy |

### Known Weaknesses

| Version | Primary Weakness |
|---------|-----------------|
| v1–v3 | Small objects, crowded scenes, high params |
| v4 | Complex training setup, Darknet framework |
| v5 | No anchor-free option, aging backbone |
| v6 | Limited multi-task, PaddlePaddle-centric (v6 variants) |
| v7 | Complex scaling, limited ecosystem |
| v8 | DFL adds export complexity, NMS required |
| v9 | Auxiliary branch adds training overhead |
| v10 | Limited multi-task support initially |
| YOLO11 | Still requires NMS in standard mode |
| v12 | Higher memory (attention), complex architecture |
| YOLO26 | Newest — less battle-tested in production |

---

## 7. Architectural Evolution Diagram

```mermaid
flowchart TD
    A["YOLOv1<br/>Darknet-24<br/>Grid Regression"] --> B["YOLOv2<br/>Darknet-19<br/>Anchor Boxes + BN"]
    B --> C["YOLOv3<br/>Darknet-53<br/>FPN + Residual"]
    C --> D["YOLOv4<br/>CSPDarknet53<br/>SPP + PANet + Mosaic"]
    C --> E["YOLOX<br/>Anchor-Free<br/>Decoupled Head"]
    D --> F["YOLOv5<br/>PyTorch Native<br/>Model Family n/s/m/l/x"]
    D --> G["YOLOv7<br/>E-ELAN<br/>Compound Scaling"]
    F --> H["YOLOv8<br/>C2f + DFL<br/>Anchor-Free + Multi-Task"]
    G --> I["YOLOv9<br/>PGI + GELAN<br/>Information Retention"]
    H --> J["YOLOv10<br/>NMS-Free<br/>Rank-Guided Blocks"]
    H --> K["YOLO11<br/>C3k2 + C2PSA<br/>Spatial Attention"]
    K --> L["YOLOv12<br/>Area Attention + R-ELAN<br/>FlashAttention"]
    J --> M["YOLO26<br/>NMS-Free + No DFL<br/>MuSGD + STAL"]
    K --> M
    L --> M

    style A fill:#ff6b6b,color:#fff
    style C fill:#ffa726,color:#fff
    style H fill:#42a5f5,color:#fff
    style L fill:#ab47bc,color:#fff
    style M fill:#66bb6a,color:#fff
```

---

## 8. Design Direction for TinyYOLO

Based on the analysis above, here's the strategic direction for our custom **tinyYOLO**:

### Design Philosophy
> Build a model that sits in the **"sweet spot"** — smaller than YOLO26n in parameters, but smarter in feature utilization, targeting **≤1.5M params**, **≤2.0 GFLOPs**, while maintaining **≥35% mAP@50-95** on COCO.

### Techniques to Cherry-Pick from Each Version

| From | Technique | Why |
|------|-----------|-----|
| YOLOv4 | Mosaic augmentation | Free accuracy boost (training only) |
| YOLOX | Decoupled head | Better cls/reg separation |
| YOLOv8 | Anchor-free detection | Simpler, fewer hyperparameters |
| YOLOv9 | GELAN-inspired blocks | Efficient feature aggregation |
| YOLOv10 | NMS-free inference | Lower latency, cleaner deployment |
| YOLO11 | Lightweight spatial attention | Focus on important regions |
| YOLO26 | No DFL, STAL | Simpler export + small object handling |

### Architecture Blueprint (Preliminary)

```
Input (320×320 or 416×416)
    │
    ├── Backbone: GhostNet-inspired + Depthwise Separable Conv
    │   ├── Stage 1: 3×3 DWConv, stride 2 → 16ch
    │   ├── Stage 2: Ghost Bottleneck → 32ch
    │   ├── Stage 3: Ghost Bottleneck → 64ch  ← P3 (small objects)
    │   ├── Stage 4: Ghost Bottleneck → 128ch ← P4 (medium)
    │   └── Stage 5: Ghost Bottleneck → 256ch ← P5 (large)
    │
    ├── Neck: Lite-PAN (depthwise separable FPN+PAN)
    │   ├── Lightweight channel attention (SE or ECA)
    │   └── Feature fusion at P3, P4, P5
    │
    └── Head: Decoupled, Anchor-Free, NMS-Free
        ├── Classification branch (lightweight)
        ├── Regression branch (no DFL)
        └── Consistent dual assignment (training)
```

### Target Specifications

| Metric | Target | Comparison (YOLO26n) |
|--------|--------|---------------------|
| Parameters | ≤1.5M | 1.7M |
| GFLOPs | ≤2.0 | 2.4 |
| mAP@50-95 | ≥35% | 39.8% |
| Input Size | 320×320 | 640×640 |
| NMS-Free | ✅ | ✅ |
| Export | ONNX/TFLite/CoreML | ✅ |
| Edge FPS (CPU) | ≥60 | ~45 |

### Training Strategy
1. **Knowledge Distillation** from YOLO26s (teacher) → tinyYOLO (student)
2. **Progressive resizing**: Train 160→224→320
3. **Mosaic + MixUp** augmentation
4. **MuSGD** or Lion optimizer
5. **STAL** label assignment for small objects
6. **Cosine annealing** with warmup

> [!NOTE]
> The goal is NOT to beat YOLO26n on accuracy — it's to achieve **≥88% of its accuracy at ≤60% of its compute**, making it viable for microcontrollers, drones, and IoT devices.

---

## 9. Open Questions for You

1. **Primary deployment target?** (MCU/Raspberry Pi/Jetson Nano/Mobile/Browser?)
2. **Target task?** (Detection only, or also segmentation/pose?)
3. **Dataset?** (COCO, custom domain like medical/aerial/industrial?)
4. **Input resolution preference?** (224/320/416?)
5. **Framework preference?** (PyTorch from scratch, or extend Ultralytics?)
6. **Quantization?** (INT8/FP16 support needed?)

---

*Analysis completed: May 2026. Sources: Original papers, Ultralytics docs, arXiv, community benchmarks.*
