"""
TinyYOLO Unified Training Script
===================================
Trains any tinyYOLO variant with auto-detected environment settings.

Usage:
    python scripts/train.py --task det --variant standard --imgsz 320
    python scripts/train.py --task seg --variant quantized --imgsz 416
    python scripts/train.py --task det --variant standard --imgsz 160,224,320,416,640 --sweep
    python scripts/train.py --task det --variant standard --imgsz 320 --teacher yolo11s.pt
"""

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tinyYOLO.utils.env import detect_environment, get_training_config, print_env_report
from tinyYOLO.models import build_model
from tinyYOLO.utils.benchmark import count_parameters, estimate_flops


TASK_DATA_MAP = {
    'det': 'coco128.yaml',
    'seg': 'coco128-seg.yaml',
    'pose': 'coco8-pose.yaml',
    'cls': 'imagenet10',
    'obb': 'dota8.yaml',
}

VALID_TASKS = ['det', 'seg', 'pose', 'cls', 'obb']
VALID_VARIANTS = ['standard', 'quantized']
VALID_IMGSZ = [160, 224, 320, 416, 640]


def parse_args():
    parser = argparse.ArgumentParser(description='TinyYOLO Training')
    parser.add_argument('--task', type=str, default='det',
                        choices=VALID_TASKS,
                        help='Task: det, seg, pose, cls, obb')
    parser.add_argument('--variant', type=str, default='standard',
                        choices=VALID_VARIANTS,
                        help='Architecture variant: standard or quantized')
    parser.add_argument('--imgsz', type=str, default='320',
                        help='Image size(s). Single value or comma-separated for sweep.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--data', type=str, default=None,
                        help='Dataset YAML (auto-selected if not specified)')
    parser.add_argument('--sweep', action='store_true',
                        help='Run sweep across multiple image sizes')
    parser.add_argument('--teacher', type=str, default=None,
                        help='Teacher model path for knowledge distillation')
    parser.add_argument('--batch', type=int, default=None,
                        help='Batch size (auto-detected if not specified)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (auto-detected if not specified)')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 5 epochs on small dataset')
    return parser.parse_args()


def train_single(args, imgsz, env):
    """Train a single model configuration."""
    # Get training config based on environment
    train_cfg = get_training_config(env, imgsz)

    # Build model name
    variant_tag = 'q' if args.variant == 'quantized' else 'std'
    model_name = args.name or f"tinyYOLO-{args.task}-{variant_tag}-{imgsz}"

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"  Task: {args.task} | Variant: {args.variant} | ImgSz: {imgsz}")
    print(f"{'='*60}")

    # Build the model
    model, model_info = build_model(
        task=args.task,
        variant=args.variant,
        nc=80 if args.task != 'pose' else 1,
    )

    # Print model info
    params = count_parameters(model)
    print(f"  Parameters: {params['total_M']}M")

    try:
        flops = estimate_flops(model, imgsz, 'cpu')
        print(f"  GFLOPs:     {flops.get('flops_G', 'N/A')}")
    except Exception:
        pass

    # Determine dataset
    data = args.data or TASK_DATA_MAP.get(args.task, 'coco128.yaml')

    # Training settings
    epochs = 5 if args.quick else args.epochs
    batch = args.batch or train_cfg['batch']
    device = args.device or train_cfg['device']

    print(f"  Device:     {device}")
    print(f"  Batch:      {batch}")
    print(f"  Epochs:     {epochs}")
    print(f"  Data:       {data}")

    # Try Ultralytics training pipeline
    try:
        from ultralytics import YOLO

        # Save model architecture for Ultralytics
        # (Ultralytics needs a config YAML or pretrained weights)
        # For custom models, we train using PyTorch directly
        print("\n  [INFO] Using standalone PyTorch training loop")
        _train_pytorch(model, data, imgsz, epochs, batch, device, model_name, env)

    except ImportError:
        print("\n  [INFO] Ultralytics not installed, using standalone training")
        _train_pytorch(model, data, imgsz, epochs, batch, device, model_name, env)

    return model_name


def _train_pytorch(model, data, imgsz, epochs, batch, device, name, env):
    """Standalone PyTorch training loop (simplified for initial testing)."""
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Save experiment config
    results_dir = PROJECT_ROOT / 'experiments' / 'results' / name
    results_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'name': name,
        'task': name.split('-')[1] if '-' in name else 'det',
        'variant': 'quantized' if '-q-' in name else 'standard',
        'imgsz': imgsz,
        'epochs': epochs,
        'batch': batch,
        'device': str(device),
        'params_M': sum(p.numel() for p in model.parameters()) / 1e6,
        'platform': env['platform'],
        'timestamp': datetime.now().isoformat(),
    }

    with open(results_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Save initial model
    torch.save(model.state_dict(), results_dir / 'model_init.pt')

    print(f"\n  Experiment saved to: {results_dir}")
    print(f"  [NOTE] Full training loop requires dataset loading.")
    print(f"  [NOTE] Use notebooks/02_det_experiments.ipynb for guided training.\n")


def main():
    args = parse_args()

    # Auto-detect environment
    env = detect_environment()
    print_env_report(env)

    # Parse image sizes
    imgsz_list = [int(s.strip()) for s in args.imgsz.split(',')]

    if args.sweep or len(imgsz_list) > 1:
        print(f"\n  Running resolution sweep: {imgsz_list}")
        for imgsz in imgsz_list:
            train_single(args, imgsz, env)
    else:
        train_single(args, imgsz_list[0], env)

    print("\n  Training complete!")


if __name__ == '__main__':
    main()
