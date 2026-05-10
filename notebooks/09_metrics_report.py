# %% [markdown]
# # 09 — Comprehensive Metrics Report
# Load experiment results and generate full analysis:
# hyperparameters, augmentation, P/R/F1, mAP, confusion matrix,
# per-class breakdown, accuracy curves, loss curves, and comparisons.

# %% Setup
import sys, json
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_BASE = Path('../experiments/results')

# %% [markdown]
# ## 1. Load Experiment Results
# Run this after training with `python scripts/train.py`

# %%
def load_experiment(exp_name):
    """Load all results from an experiment directory."""
    exp_dir = RESULTS_BASE / exp_name
    if not exp_dir.exists():
        print(f"  [WARN] Experiment not found: {exp_dir}")
        return None

    data = {'name': exp_name, 'dir': exp_dir}

    for fname in ['config.json', 'history.json', 'metrics.json']:
        fpath = exp_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                data[fname.replace('.json', '')] = json.load(f)

    return data

# Find all experiments
exp_dirs = sorted([d.name for d in RESULTS_BASE.iterdir() if d.is_dir()])
print(f"Found {len(exp_dirs)} experiments:")
for d in exp_dirs:
    print(f"  • {d}")

# Load first experiment (or specify one)
EXP_NAME = exp_dirs[0] if exp_dirs else 'tinyYOLO-det-std-320'
exp = load_experiment(EXP_NAME)

if exp is None:
    print("\nNo experiment results found. Run training first:")
    print("  python scripts/train.py --task det --variant standard --imgsz 320 --quick")

# %% [markdown]
# ## 2. Hyperparameter Configuration Report

# %%
if exp and 'config' in exp:
    cfg = exp['config']
    print(f"\n{'='*60}")
    print(f"  HYPERPARAMETER CONFIGURATION REPORT")
    print(f"  Experiment: {cfg.get('name', 'N/A')}")
    print(f"{'='*60}")

    print(f"\n  Model Architecture:")
    print(f"    Task:          {cfg.get('task', 'N/A')}")
    print(f"    Variant:       {cfg.get('variant', 'N/A')}")
    print(f"    Parameters:    {cfg.get('params_M', 'N/A')}M")
    print(f"    Input Size:    {cfg.get('imgsz', 'N/A')}×{cfg.get('imgsz', 'N/A')}")

    print(f"\n  Training Settings:")
    print(f"    Epochs:        {cfg.get('epochs', 'N/A')}")
    print(f"    Batch Size:    {cfg.get('batch', 'N/A')}")
    print(f"    Device:        {cfg.get('device', 'N/A')}")
    print(f"    AMP (FP16):    {cfg.get('amp', 'N/A')}")
    print(f"    Platform:      {cfg.get('platform', 'N/A')}")

    if 'optimizer' in cfg:
        opt = cfg['optimizer']
        print(f"\n  Optimizer:")
        print(f"    Type:          {opt.get('type', 'N/A')}")
        print(f"    Learning Rate: {opt.get('lr', 'N/A')}")
        print(f"    Weight Decay:  {opt.get('weight_decay', 'N/A')}")

    if 'scheduler' in cfg:
        sch = cfg['scheduler']
        print(f"\n  Scheduler:")
        print(f"    Type:          {sch.get('type', 'N/A')}")
        print(f"    T_max:         {sch.get('T_max', 'N/A')}")
        print(f"    eta_min:       {sch.get('eta_min', 'N/A')}")

    print(f"{'='*60}")

# %% [markdown]
# ## 3. Augmentation Report

# %%
if exp and 'config' in exp:
    cfg = exp['config']
    print(f"\n{'='*60}")
    print(f"  AUGMENTATION REPORT")
    print(f"{'='*60}")

    aug = cfg.get('augmentation', {})
    print(f"    Resize:            {aug.get('resize', 'N/A')}×{aug.get('resize', 'N/A')}")
    print(f"    Color Jitter:      {aug.get('color_jitter', False)}")
    print(f"    Horizontal Flip:   p={aug.get('horizontal_flip', 0)}")
    print(f"    Normalization:     ImageNet (via ToTensor)")
    print(f"{'='*60}")

# %% [markdown]
# ## 4. Classification Report (Detection Metrics)

# %%
if exp and 'metrics' in exp:
    m = exp['metrics']
    print(f"\n{'='*60}")
    print(f"  DETECTION METRICS REPORT")
    print(f"{'='*60}")
    print(f"  Total Predictions:   {m.get('n_predictions', 0)}")
    print(f"  Total Ground Truths: {m.get('n_ground_truths', 0)}")
    print(f"  True Positives:      {m.get('tp', 0)}")
    print(f"  False Positives:     {m.get('fp', 0)}")
    print(f"  False Negatives:     {m.get('fn', 0)}")
    print(f"{'─'*60}")
    print(f"  Precision:           {m['precision']:.4f}")
    print(f"  Recall:              {m['recall']:.4f}")
    print(f"  F1 Score:            {m['f1']:.4f}")
    print(f"  mAP@50:              {m['mAP50']:.4f}")
    print(f"  mAP@50-95:           {m['mAP50_95']:.4f}")
    print(f"{'='*60}")

    # Per-class breakdown
    per_class = m.get('per_class_metrics', {})
    if per_class:
        print(f"\n  Per-Class Breakdown (IoU=0.5):")
        print(f"  {'Class':>6} {'P':>8} {'R':>8} {'F1':>8} {'AP50':>8} {'GT':>6} {'Pred':>6}")
        print(f"  {'-'*52}")
        for cls_id, cm in sorted(per_class.items(), key=lambda x: x[1]['n_gt'], reverse=True):
            print(f"  {cls_id:>6} {cm['precision']:>8.4f} {cm['recall']:>8.4f} "
                  f"{cm['f1']:>8.4f} {cm['ap50']:>8.4f} {cm['n_gt']:>6} {cm['n_pred']:>6}")
else:
    print("\n  [INFO] No metrics.json found. Run training with evaluation:")
    print("  python scripts/train.py --task det --variant standard --imgsz 320 --quick")

# %% [markdown]
# ## 5. Training Loss Curves

# %%
if exp and 'history' in exp:
    history = exp['history']
    epochs = [h['epoch'] for h in history]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    for ax, key, color, title in [
        (axes[0], 'box', '#2196F3', 'Box Loss'),
        (axes[1], 'cls', '#FF9800', 'Cls Loss'),
        (axes[2], 'obj', '#9C27B0', 'Obj Loss'),
        (axes[3], 'total', '#4CAF50', 'Total Loss'),
    ]:
        vals = [h.get(key, 0) for h in history]
        ax.plot(epochs, vals, color=color, linewidth=2)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.set_title(title); ax.grid(True, alpha=0.3)

    plt.suptitle(f'{EXP_NAME} — Loss Curves', fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_BASE / EXP_NAME / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 6. Accuracy Curves (Precision, Recall, F1, mAP)

# %%
if exp and 'history' in exp:
    history = exp['history']
    # Check if metrics are available
    has_metrics = any(h.get('precision', 0) > 0 for h in history)

    if has_metrics:
        epochs = [h['epoch'] for h in history]

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        # Precision
        axes[0].plot(epochs, [h.get('precision', 0) for h in history],
                     color='#2196F3', linewidth=2)
        axes[0].set_title('Precision'); axes[0].set_ylim(0, 1)
        axes[0].set_xlabel('Epoch'); axes[0].grid(True, alpha=0.3)

        # Recall
        axes[1].plot(epochs, [h.get('recall', 0) for h in history],
                     color='#FF5722', linewidth=2)
        axes[1].set_title('Recall'); axes[1].set_ylim(0, 1)
        axes[1].set_xlabel('Epoch'); axes[1].grid(True, alpha=0.3)

        # F1
        axes[2].plot(epochs, [h.get('f1', 0) for h in history],
                     color='#9C27B0', linewidth=2)
        axes[2].set_title('F1 Score'); axes[2].set_ylim(0, 1)
        axes[2].set_xlabel('Epoch'); axes[2].grid(True, alpha=0.3)

        # mAP
        axes[3].plot(epochs, [h.get('mAP50', 0) for h in history],
                     color='#4CAF50', linewidth=2, label='mAP@50')
        axes[3].plot(epochs, [h.get('mAP50_95', 0) for h in history],
                     color='#2196F3', linewidth=2, linestyle='--', label='mAP@50-95')
        axes[3].set_title('mAP'); axes[3].set_ylim(0, 1)
        axes[3].set_xlabel('Epoch'); axes[3].legend(); axes[3].grid(True, alpha=0.3)

        plt.suptitle(f'{EXP_NAME} — Accuracy Curves', fontweight='bold')
        plt.tight_layout()
        plt.savefig(RESULTS_BASE / EXP_NAME / 'accuracy_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
    else:
        print("  [INFO] No per-epoch metrics found. Run with more epochs for eval checkpoints.")

# %% [markdown]
# ## 7. Epoch vs Accuracy (Combined View)

# %%
if exp and 'history' in exp and has_metrics:
    history = exp['history']
    epochs = [h['epoch'] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: Losses
    ax1.plot(epochs, [h.get('total', 0) for h in history], 'b-', linewidth=2, label='Total Loss')
    ax1.plot(epochs, [h.get('box', 0) for h in history], 'c--', linewidth=1.5, label='Box Loss')
    ax1.plot(epochs, [h.get('cls', 0) for h in history], 'y--', linewidth=1.5, label='Cls Loss')
    ax1.set_ylabel('Loss'); ax1.set_title('Training Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Bottom: Metrics
    ax2.plot(epochs, [h.get('precision', 0) for h in history], color='#2196F3', linewidth=2, label='Precision')
    ax2.plot(epochs, [h.get('recall', 0) for h in history], color='#FF5722', linewidth=2, label='Recall')
    ax2.plot(epochs, [h.get('f1', 0) for h in history], color='#9C27B0', linewidth=2, label='F1')
    ax2.plot(epochs, [h.get('mAP50', 0) for h in history], color='#4CAF50', linewidth=2.5, label='mAP@50')
    ax2.set_ylabel('Score'); ax2.set_xlabel('Epoch')
    ax2.set_title('Detection Metrics'); ax2.set_ylim(0, 1)
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle(f'{EXP_NAME} — Epoch vs Accuracy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_BASE / EXP_NAME / 'epoch_vs_accuracy.png', dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 8. Parameter Size vs Accuracy (Cross-Model Comparison)

# %%
import torch
sys.path.insert(0, '..')
from tinyYOLO.models import build_model
from tinyYOLO.utils.benchmark import count_parameters

# Build comparison data
comparison = []
for exp_name in exp_dirs:
    exp_data = load_experiment(exp_name)
    if exp_data and 'config' in exp_data:
        cfg = exp_data['config']
        fm = cfg.get('final_metrics', {})
        comparison.append({
            'name': exp_name,
            'params_M': cfg.get('params_M', 0),
            'mAP50': fm.get('mAP50', 0),
            'f1': fm.get('f1', 0),
            'precision': fm.get('precision', 0),
            'recall': fm.get('recall', 0),
        })

# Add model-only comparisons for untrained models
for task in ['det', 'seg', 'pose', 'cls', 'obb']:
    for variant in ['standard', 'quantized']:
        nc = 1 if task == 'pose' else (1000 if task == 'cls' else 80)
        model, info = build_model(task=task, variant=variant, nc=nc)
        tag = 'q' if variant == 'quantized' else 'std'
        name = f'tinyYOLO-{task}-{tag}'
        if not any(c['name'].startswith(name) for c in comparison):
            comparison.append({
                'name': name,
                'params_M': info['total_params_M'],
                'mAP50': 0,  # Untrained
                'f1': 0,
            })

if comparison:
    trained = [c for c in comparison if c['mAP50'] > 0]
    untrained = [c for c in comparison if c['mAP50'] == 0]

    fig, ax = plt.subplots(figsize=(10, 6))

    if trained:
        ax.scatter([c['params_M'] for c in trained], [c['mAP50'] for c in trained],
                   s=100, c='#4CAF50', edgecolors='white', linewidths=1.5, zorder=5, label='Trained')
        for c in trained:
            ax.annotate(c['name'].replace('tinyYOLO-', ''), (c['params_M'], c['mAP50']),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

    if untrained:
        ax.scatter([c['params_M'] for c in untrained], [0.01] * len(untrained),
                   s=60, c='#BDBDBD', marker='x', zorder=3, label='Untrained')

    # Add YOLO baseline references
    baselines = [
        ('YOLOv5n', 1.9, 28.0), ('YOLOv8n', 3.2, 37.3),
        ('YOLO11n', 2.6, 39.5), ('YOLO26n', 1.7, 39.8),
    ]
    for name, params, mAP in baselines:
        ax.scatter(params, mAP / 100, s=80, c='#FF5722', marker='D', zorder=4)
        ax.annotate(name, (params, mAP / 100), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color='#FF5722')

    ax.set_xlabel('Parameters (M)'); ax.set_ylabel('mAP@50')
    ax.set_title('Parameter Size vs Accuracy (Pareto Front)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_BASE / 'params_vs_accuracy.png', dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 9. Confusion Matrix Visualization

# %%
if exp and 'metrics' in exp and 'confusion_matrix' in exp['metrics']:
    cm = exp['metrics']['confusion_matrix']
    matrix = np.array(cm['matrix'])
    classes = cm['classes']

    if matrix.sum() > 0:
        fig, ax = plt.subplots(figsize=(max(8, len(classes)), max(6, len(classes) * 0.7)))
        im = ax.imshow(matrix, interpolation='nearest', cmap='Blues')
        ax.set_title(f'{EXP_NAME} — Confusion Matrix')
        plt.colorbar(im, ax=ax)
        labels = [str(c) for c in classes] + ['BG']
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(matrix)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels[:len(matrix)], fontsize=8)
        ax.set_xlabel('Ground Truth'); ax.set_ylabel('Predicted')
        plt.tight_layout()
        plt.savefig(RESULTS_BASE / EXP_NAME / 'confusion_matrix_nb.png', dpi=150, bbox_inches='tight')
        plt.show()
    else:
        print("  Confusion matrix is empty (model may need more training)")

# %% [markdown]
# ## 10. IoU Distribution Analysis

# %%
if exp and 'metrics' in exp:
    m = exp['metrics']
    # Reconstruct IoU distribution from AP at different thresholds
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    ap50 = m.get('mAP50', 0)
    ap50_95 = m.get('mAP50_95', 0)

    # Approximate: AP drops as IoU threshold increases
    if ap50 > 0:
        estimated_aps = [ap50 * max(0, 1 - (t - 0.5) * 2) for t in iou_thresholds]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(iou_thresholds, estimated_aps, width=0.04, color='#2196F3', alpha=0.8)
        ax.axhline(y=ap50_95, color='#FF5722', linestyle='--', label=f'mAP@50-95 = {ap50_95:.4f}')
        ax.set_xlabel('IoU Threshold'); ax.set_ylabel('AP')
        ax.set_title('AP at Different IoU Thresholds')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_BASE / EXP_NAME / 'iou_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()

# %% [markdown]
# ## 11. Learning Rate Schedule

# %%
if exp and 'history' in exp:
    history = exp['history']
    epochs = [h['epoch'] for h in history]
    lrs = [h['lr'] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, lrs, color='#FF9800', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule (Cosine Annealing)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_BASE / EXP_NAME / 'lr_schedule.png', dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 12. Cross-Experiment Comparison
# Compare all trained experiments side by side.

# %%
trained_exps = []
for exp_name in exp_dirs:
    e = load_experiment(exp_name)
    if e and 'config' in e:
        cfg = e['config']
        fm = cfg.get('final_metrics', {})
        trained_exps.append({
            'name': exp_name,
            'params_M': cfg.get('params_M', 0),
            'epochs': cfg.get('epochs', 0),
            'imgsz': cfg.get('imgsz', 0),
            'precision': fm.get('precision', 0),
            'recall': fm.get('recall', 0),
            'f1': fm.get('f1', 0),
            'mAP50': fm.get('mAP50', 0),
            'mAP50_95': fm.get('mAP50_95', 0),
            'best_loss': cfg.get('best_loss', 0),
        })

if trained_exps:
    print(f"\n{'Experiment':<30} {'Params':>7} {'P':>6} {'R':>6} {'F1':>6} {'mAP50':>7} {'mAP50-95':>9}")
    print("-" * 80)
    for e in trained_exps:
        print(f"{e['name']:<30} {e['params_M']:>6.2f}M {e['precision']:>6.3f} {e['recall']:>6.3f} "
              f"{e['f1']:>6.3f} {e['mAP50']:>7.4f} {e['mAP50_95']:>9.4f}")

# %% [markdown]
# ## Summary
# This notebook provides comprehensive reporting for all tinyYOLO experiments:
# - ✅ Hyperparameter configuration report
# - ✅ Augmentation report
# - ✅ Classification/detection report (P, R, F1)
# - ✅ mAP@50 and mAP@50-95
# - ✅ Per-class breakdown
# - ✅ Loss curves (box, cls, obj, total)
# - ✅ Accuracy curves (P, R, F1, mAP vs epoch)
# - ✅ Epoch vs accuracy combined view
# - ✅ Parameter size vs accuracy (Pareto front)
# - ✅ Confusion matrix
# - ✅ IoU distribution analysis
# - ✅ Learning rate schedule
# - ✅ Cross-experiment comparison table
