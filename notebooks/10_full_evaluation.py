# %% [markdown]
# # TinyYOLO — Complete Evaluation Pipeline (R1 Updated)
# Run all steps sequentially on Google Colab.
# Each cell is independent — run them in order.
# R1 features: seed control, warmup, mosaic, TAL, quantization.

# %% Step 0: Setup
import sys, os
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent if '__file__' in dir() else Path('.')
_PROJECT_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == 'notebooks' else _SCRIPT_DIR
os.chdir(str(_PROJECT_ROOT))
sys.path.insert(0, '.')

# Verify environment
from tinyYOLO.utils.env import print_env_report
print_env_report()

# %% Step 1: Train Standard Variant (if not already done)
# R1: Added --seed 42 --warmup 3 for deterministic training with linear LR warmup
os.system("python scripts/train.py --task det --variant standard --imgsz 320 --epochs 100 --seed 42 --warmup 3")

# %% Step 2: Train Quantized Variant
os.system("python scripts/train.py --task det --variant quantized --imgsz 320 --epochs 100 --seed 42 --warmup 3")

# %% Step 3: Resolution Ablation (standard variant at 4 resolutions)
for res in [160, 224, 416, 640]:
    print(f"\n{'='*60}")
    print(f"  Resolution Ablation: {res}×{res}")
    print(f"{'='*60}")
    os.system(f"python scripts/train.py --task det --variant standard --imgsz {res} --epochs 50 --seed 42 --warmup 3")

# %% Step 4: Run Metrics Report Notebook
os.system("python notebooks/09_metrics_report.py")

# %% Step 5: ONNX Export
os.system("python scripts/export.py"
          " --weights experiments/results/tinyYOLO-det-std-320/best.pt"
          " --task det --variant standard --imgsz 320 --formats onnx")

# %% Step 5b: INT8 Quantization (R1 NEW)
# Post-Training Quantization of the quantized variant
os.system("python scripts/quantize.py --mode ptq"
          " --weights experiments/results/tinyYOLO-det-q-320/best.pt"
          " --task det --variant quantized --data coco128.yaml"
          " --imgsz 320 --n-calib 100 --backend qnnpack")

# %% Step 6: Cross-Experiment Comparison
import json

results_dir = Path('experiments/results')
if results_dir.exists():
    experiments = sorted([d.name for d in results_dir.iterdir() if d.is_dir()])

    print(f"\n{'Experiment':<35} {'Params':>7} {'P':>6} {'R':>6} {'F1':>6} {'mAP50':>7} {'Loss':>7}")
    print("-" * 85)

    for exp_name in experiments:
        cfg_path = results_dir / exp_name / 'config.json'
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            fm = cfg.get('final_metrics', {})
            print(f"{exp_name:<35} {cfg.get('params_M', 0):>6.2f}M "
                  f"{fm.get('precision', 0):>6.3f} {fm.get('recall', 0):>6.3f} "
                  f"{fm.get('f1', 0):>6.3f} {fm.get('mAP50', 0):>7.4f} "
                  f"{cfg.get('best_loss', 0):>7.4f}")
else:
    print("No experiments found. Run training steps first.")

# %% Step 7: Benchmark All Models
os.system("python scripts/benchmark_models.py --tasks det --variants standard,quantized --imgsz 160,224,320,416,640")

# %% Step 8: Generate Final Plots
import matplotlib.pyplot as plt
import numpy as np

if results_dir.exists():
    experiments = sorted([d.name for d in results_dir.iterdir() if d.is_dir()])

    # Load all experiment histories
    all_data = {}
    for exp_name in experiments:
        hist_path = results_dir / exp_name / 'history.json'
        if hist_path.exists():
            with open(hist_path) as f:
                all_data[exp_name] = json.load(f)

    if all_data:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Loss comparison
        ax = axes[0, 0]
        for name, hist in all_data.items():
            epochs = [h['epoch'] for h in hist]
            losses = [h.get('total', 0) for h in hist]
            label = name.replace('tinyYOLO-det-', '')
            ax.plot(epochs, losses, linewidth=2, label=label)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Total Loss')
        ax.set_title('Loss Convergence Comparison'); ax.legend(); ax.grid(True, alpha=0.3)

        # Plot 2: mAP comparison
        ax = axes[0, 1]
        for name, hist in all_data.items():
            epochs = [h['epoch'] for h in hist]
            maps = [h.get('mAP50', 0) for h in hist]
            label = name.replace('tinyYOLO-det-', '')
            ax.plot(epochs, maps, linewidth=2, label=label)
        ax.set_xlabel('Epoch'); ax.set_ylabel('mAP@50')
        ax.set_title('mAP@50 Progression'); ax.legend(); ax.grid(True, alpha=0.3)

        # Plot 3: Params vs mAP (Pareto)
        ax = axes[1, 0]
        for exp_name in experiments:
            cfg_path = results_dir / exp_name / 'config.json'
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = json.load(f)
                fm = cfg.get('final_metrics', {})
                params = cfg.get('params_M', 0)
                mAP = fm.get('mAP50', 0)
                label = exp_name.replace('tinyYOLO-det-', '')
                ax.scatter(params, mAP, s=120, zorder=5)
                ax.annotate(label, (params, mAP), textcoords="offset points",
                           xytext=(5, 5), fontsize=9)

        # Add YOLO baselines
        for name, p, m in [('YOLOv5n', 1.9, 0.28), ('YOLOv8n', 3.2, 0.373), ('YOLO11n', 2.6, 0.395)]:
            ax.scatter(p, m, s=80, marker='D', c='#FF5722', zorder=4)
            ax.annotate(name, (p, m), textcoords="offset points", xytext=(5, 5),
                       fontsize=8, color='#FF5722')
        ax.set_xlabel('Parameters (M)'); ax.set_ylabel('mAP@50')
        ax.set_title('Params vs Accuracy (Pareto Front)'); ax.grid(True, alpha=0.3)

        # Plot 4: Box loss detail
        ax = axes[1, 1]
        for name, hist in all_data.items():
            epochs = [h['epoch'] for h in hist]
            box = [h.get('box', 0) for h in hist]
            label = name.replace('tinyYOLO-det-', '')
            ax.plot(epochs, box, linewidth=2, label=label)
        ax.set_xlabel('Epoch'); ax.set_ylabel('CIoU Box Loss')
        ax.set_title('Box Loss (CIoU) Convergence'); ax.legend(); ax.grid(True, alpha=0.3)

        plt.suptitle('TinyYOLO Cross-Experiment Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('experiments/results/cross_experiment_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved: experiments/results/cross_experiment_analysis.png")

# %% [markdown]
# ## Summary of Commands
# ```bash
# # Quick reference — copy-paste individual commands as needed:
#
# # Train standard (R1: with seed + warmup)
# python scripts/train.py --task det --variant standard --imgsz 320 --epochs 100 --seed 42 --warmup 3
#
# # Train quantized (R1: with seed + warmup)
# python scripts/train.py --task det --variant quantized --imgsz 320 --epochs 100 --seed 42 --warmup 3
#
# # Resolution sweep
# python scripts/train.py --task det --variant standard --imgsz 160 --epochs 50 --seed 42 --warmup 3
# python scripts/train.py --task det --variant standard --imgsz 224 --epochs 50 --seed 42 --warmup 3
# python scripts/train.py --task det --variant standard --imgsz 416 --epochs 50 --seed 42 --warmup 3
# python scripts/train.py --task det --variant standard --imgsz 640 --epochs 50 --seed 42 --warmup 3
#
# # Metrics report
# python notebooks/09_metrics_report.py
#
# # ONNX export
# python scripts/export.py --weights experiments/results/tinyYOLO-det-std-320/best.pt --task det --variant standard --imgsz 320 --formats onnx
#
# # INT8 Quantization (R1 NEW)
# python scripts/quantize.py --mode ptq --weights experiments/results/tinyYOLO-det-q-320/best.pt --data coco128.yaml --n-calib 100
# python scripts/quantize.py --mode qat --weights experiments/results/tinyYOLO-det-q-320/best.pt --data coco128.yaml --epochs 10
#
# # Benchmark
# python scripts/benchmark_models.py
# ```

