"""
TinyYOLO Metrics
==================
Comprehensive evaluation metrics for object detection:
- IoU, Precision, Recall, F1
- AP, mAP@50, mAP@50-95
- Confusion Matrix
- Per-class breakdown
- Visualization (curves, matrices)
"""

import numpy as np
import torch
from pathlib import Path


# ---------------------------------------------------------------------------
# IoU Computation
# ---------------------------------------------------------------------------

def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes (xyxy format).

    Args:
        boxes1: [N, 4] tensor.
        boxes2: [M, 4] tensor.

    Returns:
        [N, M] IoU matrix.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + 1e-7)


# ---------------------------------------------------------------------------
# Matching & Core Metrics
# ---------------------------------------------------------------------------

def match_predictions(pred_boxes, pred_scores, pred_classes,
                      gt_boxes, gt_classes, iou_thresh=0.5):
    """
    Match predictions to ground truth using IoU threshold.

    Returns:
        tp: [N] boolean array — True Positives
        fp: [N] boolean array — False Positives
        fn_count: int — number of unmatched ground truths
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    tp = np.zeros(n_pred, dtype=bool)
    fp = np.zeros(n_pred, dtype=bool)

    if n_gt == 0:
        fp[:] = True
        return tp, fp, 0

    if n_pred == 0:
        return tp, fp, n_gt

    # Compute IoU matrix
    if isinstance(pred_boxes, torch.Tensor):
        iou_matrix = box_iou(pred_boxes, gt_boxes).cpu().numpy()
    else:
        iou_matrix = box_iou(
            torch.tensor(pred_boxes, dtype=torch.float32),
            torch.tensor(gt_boxes, dtype=torch.float32)
        ).numpy()

    # Sort predictions by confidence (descending)
    sort_idx = np.argsort(-pred_scores if isinstance(pred_scores, np.ndarray)
                          else -pred_scores.cpu().numpy())

    gt_matched = np.zeros(n_gt, dtype=bool)

    for idx in sort_idx:
        # Find best matching GT for this prediction
        ious = iou_matrix[idx]

        # Filter by class
        pred_cls = pred_classes[idx] if isinstance(pred_classes, np.ndarray) \
            else pred_classes[idx].item()

        best_iou = 0
        best_gt = -1
        for j in range(n_gt):
            gt_cls = gt_classes[j] if isinstance(gt_classes, np.ndarray) \
                else gt_classes[j].item()
            if gt_matched[j] or gt_cls != pred_cls:
                continue
            if ious[j] > best_iou:
                best_iou = ious[j]
                best_gt = j

        if best_iou >= iou_thresh and best_gt >= 0:
            tp[idx] = True
            gt_matched[best_gt] = True
        else:
            fp[idx] = True

    fn_count = (~gt_matched).sum()
    return tp, fp, int(fn_count)


def compute_precision_recall(tp, fp, n_gt):
    """Compute precision and recall from TP/FP arrays."""
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision = tp_cum / (tp_cum + fp_cum + 1e-7)
    recall = tp_cum / (n_gt + 1e-7)
    return precision, recall


def compute_ap(precision, recall):
    """
    Compute Average Precision using 101-point step-function interpolation (COCO standard).
    Eliminates linear interpolation overestimation and sentinel boundary bugs.
    """
    if len(recall) == 0 or len(precision) == 0:
        return 0.0

    # Ensure arrays are sorted by recall ascending
    sort_idx = np.argsort(recall)
    recall = recall[sort_idx]
    precision = precision[sort_idx]

    # Compute suffix maximum (precision envelope)
    suffix_max = np.maximum.accumulate(precision[::-1])[::-1]

    # Sample at COCO's 101 recall levels (0.00, 0.01, ..., 1.00)
    recall_levels = np.linspace(0, 1, 101)
    
    # Find index of first recall point >= each level
    indices = np.searchsorted(recall, recall_levels, side='left')

    # Map indices to suffix_max, defaulting to 0.0 if out of bounds (recall_level > max_recall)
    precision_interp = np.zeros(101)
    valid_mask = indices < len(recall)
    precision_interp[valid_mask] = suffix_max[indices[valid_mask]]

    return np.mean(precision_interp)



# ---------------------------------------------------------------------------
# Full Evaluation
# ---------------------------------------------------------------------------

class DetectionMetrics:
    """
    Accumulates predictions and ground truths across batches,
    then computes comprehensive detection metrics per-image to prevent memory overflow.
    """

    def __init__(self, nc=80, conf_thresh=0.25, iou_thresholds=None):
        self.nc = nc
        self.conf_thresh = conf_thresh
        self.iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)

        # Accumulated data
        self.all_pred_boxes = []
        self.all_pred_scores = []
        self.all_pred_classes = []
        self.all_gt_boxes = []
        self.all_gt_classes = []

    def _to_numpy(self, t):
        """Convert tensor or array to numpy, handling NumPy version incompatibilities."""
        if isinstance(t, np.ndarray):
            return t
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu()
            try:
                return t.numpy()
            except (RuntimeError, TypeError):
                # Fallback for NumPy 2.x incompatibility with older torch
                return np.array(t.tolist())
        return np.array(t) if len(t) > 0 else np.zeros(0)

    def update(self, pred_list, gt_list):
        """
        Add a batch of predictions and ground truths.

        Args:
            pred_list: List of [N, 6] tensors (x1, y1, x2, y2, conf, cls).
            gt_list: List of [M, 5] tensors (x1, y1, x2, y2, cls).
        """
        for pred, gt in zip(pred_list, gt_list):
            pred_np = self._to_numpy(pred) if len(pred) > 0 else np.zeros((0, 6))
            gt_np = self._to_numpy(gt) if len(gt) > 0 else np.zeros((0, 5))

            if len(pred_np) > 0 and pred_np.ndim == 2:
                self.all_pred_boxes.append(pred_np[:, :4])
                self.all_pred_scores.append(pred_np[:, 4])
                self.all_pred_classes.append(pred_np[:, 5].astype(int))
            else:
                self.all_pred_boxes.append(np.zeros((0, 4)))
                self.all_pred_scores.append(np.zeros(0))
                self.all_pred_classes.append(np.zeros(0, dtype=int))

            if len(gt_np) > 0 and gt_np.ndim == 2:
                self.all_gt_boxes.append(gt_np[:, :4])
                self.all_gt_classes.append(gt_np[:, 4].astype(int))
            else:
                self.all_gt_boxes.append(np.zeros((0, 4)))
                self.all_gt_classes.append(np.zeros(0, dtype=int))

    def _match_per_image(self, iou_thresh):
        """
        Perform class-aware matching of predictions to ground truth per image.
        This prevents giant global IoU matrices and ensures 100% mathematical correctness.
        """
        n_images = len(self.all_pred_boxes)
        
        class_tps = {c: [] for c in range(self.nc)}
        class_fps = {c: [] for c in range(self.nc)}
        class_scores = {c: [] for c in range(self.nc)}
        class_gt_counts = {c: 0 for c in range(self.nc)}
        
        global_tps = []
        global_fps = []
        global_fn_count = 0
        
        for img_idx in range(n_images):
            pred_boxes = self.all_pred_boxes[img_idx]
            pred_scores = self.all_pred_scores[img_idx]
            pred_classes = self.all_pred_classes[img_idx]
            
            gt_boxes = self.all_gt_boxes[img_idx]
            gt_classes = self.all_gt_classes[img_idx]
            
            # Run matching for this image
            tp_img, fp_img, fn_img_count = match_predictions(
                pred_boxes, pred_scores, pred_classes,
                gt_boxes, gt_classes, iou_thresh
            )
            
            global_tps.append(tp_img)
            global_fps.append(fp_img)
            global_fn_count += fn_img_count
            
            # Group by class
            for k in range(len(pred_boxes)):
                c = int(pred_classes[k])
                if 0 <= c < self.nc:
                    class_tps[c].append(tp_img[k])
                    class_fps[c].append(fp_img[k])
                    class_scores[c].append(pred_scores[k])
                    
            for j in range(len(gt_boxes)):
                c = int(gt_classes[j])
                if 0 <= c < self.nc:
                    class_gt_counts[c] += 1
                    
        # Concatenate global arrays
        global_tp = np.concatenate(global_tps) if global_tps else np.zeros(0, dtype=bool)
        global_fp = np.concatenate(global_fps) if global_fps else np.zeros(0, dtype=bool)
        
        return class_tps, class_fps, class_scores, class_gt_counts, global_tp, global_fp, global_fn_count

    def compute(self):
        """
        Compute all metrics.

        Returns:
            dict with: precision, recall, f1, mAP50, mAP50_95,
            per_class_ap, confusion_matrix, etc.
        """
        n_pred = sum(len(x) for x in self.all_pred_boxes)
        n_gt = sum(len(x) for x in self.all_gt_boxes)

        results = {
            'n_predictions': n_pred,
            'n_ground_truths': n_gt,
        }

        if n_pred == 0 or n_gt == 0:
            results.update({
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'mAP50': 0.0, 'mAP50_95': 0.0,
                'per_class_ap50': {}, 'per_class_metrics': {},
            })
            return results

        # --- Match per image for each IoU threshold ---
        ap_per_thresh = []
        
        tp50 = None
        fp50 = None
        fn50_count = 0
        per_class_ap50 = {}

        for iou_thresh in self.iou_thresholds:
            class_tps, class_fps, class_scores, class_gt_counts, global_tp, global_fp, global_fn = self._match_per_image(iou_thresh)
            
            # Save IoU=0.5 results for overall metrics
            if abs(iou_thresh - 0.5) < 1e-4:
                tp50 = global_tp
                fp50 = global_fp
                fn50_count = global_fn
                
            per_class_ap = {}
            for cls_id in range(self.nc):
                n_cls_gt = class_gt_counts[cls_id]
                cls_scores = np.array(class_scores[cls_id])
                
                # Exclude classes with zero ground truths on the test split
                # to match official COCO evaluation protocol
                if n_cls_gt == 0:
                    continue
                if len(cls_scores) == 0:
                    per_class_ap[cls_id] = 0.0
                    continue
                    
                cls_tp = np.array(class_tps[cls_id])
                cls_fp = np.array(class_fps[cls_id])
                
                # Sort by score for AP
                sort_idx = np.argsort(-cls_scores)
                tp_sorted = cls_tp[sort_idx]
                fp_sorted = cls_fp[sort_idx]
                
                precision, recall = compute_precision_recall(
                    tp_sorted, fp_sorted, n_cls_gt
                )
                per_class_ap[cls_id] = compute_ap(precision, recall)
                
            if abs(iou_thresh - 0.5) < 1e-4:
                per_class_ap50 = per_class_ap
                
            # Mean AP at this threshold (averaged ONLY over classes that have ground truths)
            mean_ap = np.mean(list(per_class_ap.values())) if per_class_ap else 0.0
            ap_per_thresh.append(mean_ap)

        results['per_class_ap50'] = per_class_ap50

        # --- Overall metrics at IoU=0.5 ---
        if tp50 is None:
            _, _, _, _, tp50, fp50, fn50_count = self._match_per_image(0.5)

        tp_count = tp50.sum()
        fp_count = fp50.sum()

        precision = tp_count / (tp_count + fp_count + 1e-7)
        recall = tp_count / (n_gt + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        results.update({
            'precision': round(float(precision), 4),
            'recall': round(float(recall), 4),
            'f1': round(float(f1), 4),
            'tp': int(tp_count),
            'fp': int(fp_count),
            'fn': int(fn50_count),
            'mAP50': round(float(ap_per_thresh[0]), 4) if ap_per_thresh else 0.0,
            'mAP50_95': round(float(np.mean(ap_per_thresh)), 4) if ap_per_thresh else 0.0,
        })

        # --- Per-class metrics at IoU=0.5 ---
        all_pred_classes = np.concatenate(self.all_pred_classes) if self.all_pred_classes else np.zeros(0, dtype=int)
        all_gt_classes = np.concatenate(self.all_gt_classes) if self.all_gt_classes else np.zeros(0, dtype=int)

        per_class_metrics = {}
        for cls_id in range(self.nc):
            pred_mask = all_pred_classes == cls_id
            gt_mask = all_gt_classes == cls_id
            n_cls_pred = pred_mask.sum()
            n_cls_gt = gt_mask.sum()

            if n_cls_gt > 0 or n_cls_pred > 0:
                cls_tp = tp50[pred_mask].sum() if n_cls_pred > 0 else 0
                cls_fp = fp50[pred_mask].sum() if n_cls_pred > 0 else 0
                cls_p = cls_tp / (cls_tp + cls_fp + 1e-7)
                cls_r = cls_tp / (n_cls_gt + 1e-7)
                cls_f1 = 2 * cls_p * cls_r / (cls_p + cls_r + 1e-7)

                per_class_metrics[cls_id] = {
                    'precision': round(float(cls_p), 4),
                    'recall': round(float(cls_r), 4),
                    'f1': round(float(cls_f1), 4),
                    'n_gt': int(n_cls_gt),
                    'n_pred': int(n_cls_pred),
                    'ap50': round(float(results['per_class_ap50'].get(cls_id, 0)), 4),
                }

        results['per_class_metrics'] = per_class_metrics

        # --- Confusion Matrix ---
        results['confusion_matrix'] = self._build_confusion_matrix(
            all_pred_classes, all_gt_classes, tp50, fp50, n_gt
        )

        return results

    def _build_confusion_matrix(self, pred_cls, gt_cls, tp, fp, n_gt):
        """Build a simplified confusion matrix."""
        # Active classes only
        active_classes = sorted(set(
            list(np.unique(pred_cls)) + list(np.unique(gt_cls))
        ))
        n_active = len(active_classes)
        cls_to_idx = {c: i for i, c in enumerate(active_classes)}

        # Matrix: rows = predicted, cols = ground truth
        matrix = np.zeros((n_active + 1, n_active + 1), dtype=int)  # +1 for background

        # TP: pred[i] matched gt[j] — both same class
        for i, (is_tp, p_cls) in enumerate(zip(tp, pred_cls)):
            if is_tp and p_cls in cls_to_idx:
                idx = cls_to_idx[int(p_cls)]
                matrix[idx, idx] += 1

        # FP: pred[i] not matched — goes to pred_cls row, background col
        for i, (is_fp, p_cls) in enumerate(zip(fp, pred_cls)):
            if is_fp and p_cls in cls_to_idx:
                idx = cls_to_idx[int(p_cls)]
                matrix[idx, -1] += 1  # Predicted but no GT match

        return {
            'matrix': matrix.tolist(),
            'classes': [int(c) for c in active_classes],
        }

    def reset(self):
        """Clear accumulated data."""
        self.all_pred_boxes.clear()
        self.all_pred_scores.clear()
        self.all_pred_classes.clear()
        self.all_gt_boxes.clear()
        self.all_gt_classes.clear()


# ---------------------------------------------------------------------------
# Reporting & Visualization
# ---------------------------------------------------------------------------

def print_metrics_report(metrics, class_names=None):
    """Print a formatted metrics report."""
    print(f"\n{'='*60}")
    print(f"  Detection Metrics Report")
    print(f"{'='*60}")
    print(f"  Predictions:   {metrics['n_predictions']}")
    print(f"  Ground Truths: {metrics['n_ground_truths']}")
    print(f"  TP: {metrics.get('tp', 0)}  |  FP: {metrics.get('fp', 0)}  |  FN: {metrics.get('fn', 0)}")
    print(f"{'─'*60}")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print(f"  F1 Score:      {metrics['f1']:.4f}")
    print(f"  mAP@50:        {metrics['mAP50']:.4f}")
    print(f"  mAP@50-95:     {metrics['mAP50_95']:.4f}")
    print(f"{'='*60}")

    # Per-class breakdown (top 20)
    per_class = metrics.get('per_class_metrics', {})
    if per_class:
        print(f"\n  Per-Class Metrics (IoU=0.5):")
        print(f"  {'Class':>6} {'Name':<20} {'P':>8} {'R':>8} {'F1':>8} {'AP50':>8} {'GT':>6} {'Pred':>6}")
        print(f"  {'-'*68}")
        sorted_cls = sorted(per_class.items(), key=lambda x: x[1]['n_gt'], reverse=True)
        for cls_id, m in sorted_cls[:20]:
            name = class_names[cls_id] if class_names and cls_id < len(class_names) else f"cls_{cls_id}"
            print(f"  {cls_id:>6} {name:<20} {m['precision']:>8.4f} {m['recall']:>8.4f} "
                  f"{m['f1']:>8.4f} {m['ap50']:>8.4f} {m['n_gt']:>6} {m['n_pred']:>6}")


def plot_training_curves(history, save_path=None):
    """
    Plot comprehensive training curves.

    Args:
        history: List of dicts with per-epoch metrics.
        save_path: Path to save the plot.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = [h['epoch'] for h in history]

    # Determine which metrics are available
    has_metrics = 'precision' in history[-1] if history else False

    if has_metrics:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes = [axes]  # Wrap for consistent indexing

    # Row 1: Losses
    loss_axes = axes[0] if has_metrics else axes[0]

    for ax, key, color, title in [
        (loss_axes[0], 'box', '#2196F3', 'Box Loss'),
        (loss_axes[1], 'cls', '#FF9800', 'Classification Loss'),
        (loss_axes[2], 'total', '#4CAF50', 'Total Loss'),
    ]:
        vals = [h.get(key, 0) for h in history]
        ax.plot(epochs, vals, color=color, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # Row 2: Metrics (if available)
    if has_metrics:
        # Precision & Recall
        axes[1][0].plot(epochs, [h.get('precision', 0) for h in history],
                        color='#2196F3', linewidth=2, label='Precision')
        axes[1][0].plot(epochs, [h.get('recall', 0) for h in history],
                        color='#FF5722', linewidth=2, label='Recall')
        axes[1][0].set_xlabel('Epoch')
        axes[1][0].set_ylabel('Score')
        axes[1][0].set_title('Precision & Recall')
        axes[1][0].legend()
        axes[1][0].set_ylim(0, 1)
        axes[1][0].grid(True, alpha=0.3)

        # F1
        axes[1][1].plot(epochs, [h.get('f1', 0) for h in history],
                        color='#9C27B0', linewidth=2)
        axes[1][1].set_xlabel('Epoch')
        axes[1][1].set_ylabel('F1 Score')
        axes[1][1].set_title('F1 Score')
        axes[1][1].set_ylim(0, 1)
        axes[1][1].grid(True, alpha=0.3)

        # mAP
        axes[1][2].plot(epochs, [h.get('mAP50', 0) for h in history],
                        color='#4CAF50', linewidth=2, label='mAP@50')
        axes[1][2].plot(epochs, [h.get('mAP50_95', 0) for h in history],
                        color='#2196F3', linewidth=2, label='mAP@50-95', linestyle='--')
        axes[1][2].set_xlabel('Epoch')
        axes[1][2].set_ylabel('mAP')
        axes[1][2].set_title('mAP')
        axes[1][2].legend()
        axes[1][2].set_ylim(0, 1)
        axes[1][2].grid(True, alpha=0.3)

    plt.suptitle('TinyYOLO Training Report', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Training curves saved: {save_path}")
    plt.close()


def plot_confusion_matrix(cm_data, class_names=None, save_path=None):
    """Plot confusion matrix heatmap."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    matrix = np.array(cm_data['matrix'])
    classes = cm_data['classes']

    fig, ax = plt.subplots(figsize=(max(8, len(classes)), max(6, len(classes) * 0.8)))

    im = ax.imshow(matrix, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix')
    plt.colorbar(im, ax=ax)

    labels = [class_names[c] if class_names and c < len(class_names) else f'{c}'
              for c in classes] + ['BG']

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(matrix)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels[:len(matrix)], fontsize=8)
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Predicted')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_full_report(metrics, history, save_dir, class_names=None):
    """
    Generate a complete experiment report.

    Saves:
        - metrics.json
        - training_curves.png (losses + metrics)
        - confusion_matrix.png
        - per_class_report.txt
        - hyperparams_report.txt
    """
    import json

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save metrics JSON
    with open(save_dir / 'metrics.json', 'w') as f:
        # Convert numpy types for JSON serialization
        clean_metrics = json.loads(json.dumps(metrics, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x))
        json.dump(clean_metrics, f, indent=2)

    # 2. Training curves
    plot_training_curves(history, save_dir / 'training_curves.png')

    # 3. Confusion matrix
    if 'confusion_matrix' in metrics:
        plot_confusion_matrix(
            metrics['confusion_matrix'], class_names,
            save_dir / 'confusion_matrix.png'
        )

    # 4. Per-class report
    with open(save_dir / 'per_class_report.txt', 'w') as f:
        f.write("TinyYOLO Per-Class Detection Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Class':>6} {'Name':<20} {'P':>8} {'R':>8} {'F1':>8} {'AP50':>8} {'GT':>6}\n")
        f.write("-" * 70 + "\n")
        for cls_id, m in sorted(metrics.get('per_class_metrics', {}).items()):
            name = class_names[cls_id] if class_names and cls_id < len(class_names) else f"cls_{cls_id}"
            f.write(f"{cls_id:>6} {name:<20} {m['precision']:>8.4f} {m['recall']:>8.4f} "
                    f"{m['f1']:>8.4f} {m['ap50']:>8.4f} {m['n_gt']:>6}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'MEAN':<27} {metrics['precision']:>8.4f} {metrics['recall']:>8.4f} "
                f"{metrics['f1']:>8.4f} {metrics['mAP50']:>8.4f} {metrics['n_ground_truths']:>6}\n")

    print(f"  Full report saved to: {save_dir}")
