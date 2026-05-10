"""
TinyYOLO Unified Training Script
===================================
Trains any tinyYOLO variant with auto-detected environment settings.
Uses Ultralytics data pipeline for loading + augmentation, and our custom
model for forward/backward.

Usage:
    python scripts/train.py --task det --variant standard --imgsz 320
    python scripts/train.py --task det --variant standard --imgsz 320 --quick
    python scripts/train.py --task det --variant standard --imgsz 160,224,320,416,640 --sweep
"""

import argparse
import os
import sys
import json
import time
import math
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np

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


def parse_args():
    parser = argparse.ArgumentParser(description='TinyYOLO Training')
    parser.add_argument('--task', type=str, default='det',
                        choices=VALID_TASKS, help='Task type')
    parser.add_argument('--variant', type=str, default='standard',
                        choices=VALID_VARIANTS, help='Architecture variant')
    parser.add_argument('--imgsz', type=str, default='320',
                        help='Image size(s). Comma-separated for sweep.')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--data', type=str, default=None, help='Dataset YAML path')
    parser.add_argument('--sweep', action='store_true', help='Resolution sweep')
    parser.add_argument('--teacher', type=str, default=None, help='Teacher model for distillation')
    parser.add_argument('--batch', type=int, default=None, help='Batch size (auto if omitted)')
    parser.add_argument('--device', type=str, default=None, help='Device (auto if omitted)')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--quick', action='store_true', help='Quick test: 5 epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# COCO128 Dataset Loader (works with Ultralytics data download)
# ---------------------------------------------------------------------------

def download_coco128():
    """Download COCO128 dataset using Ultralytics."""
    try:
        from ultralytics.data.utils import check_det_dataset
        data_dict = check_det_dataset('coco128.yaml')
        return data_dict
    except Exception as e:
        print(f"  [WARN] Ultralytics download failed: {e}")
        print(f"  [INFO] Attempting manual download...")
        # Manual fallback
        import urllib.request, zipfile
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
        dest = Path("datasets")
        dest.mkdir(exist_ok=True)
        zip_path = dest / "coco128.zip"
        if not (dest / "coco128").exists():
            print(f"  Downloading COCO128...")
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(dest)
            zip_path.unlink()
        return {'train': str(dest / 'coco128' / 'images' / 'train2017'),
                'val': str(dest / 'coco128' / 'images' / 'train2017'),
                'nc': 80, 'names': {i: f'class_{i}' for i in range(80)}}


class SimpleDetectionDataset(Dataset):
    """
    Loads images + YOLO-format labels from a directory.
    Works with COCO128 standard layout.
    """

    def __init__(self, img_dir, imgsz=320, augment=True):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(str(img_dir).replace('images', 'labels'))
        self.imgsz = imgsz
        self.augment = augment

        # Find all images
        self.img_files = sorted([
            f for f in self.img_dir.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
        ])

        if not self.img_files:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

        # Transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((imgsz, imgsz)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01) if augment else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip(p=0.5) if augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        import cv2

        # Load image
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            # Return random image as fallback
            img = np.random.randint(0, 255, (self.imgsz, self.imgsz, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transforms
        img_tensor = self.transform(img)  # [3, imgsz, imgsz]

        # Load labels (YOLO format: class cx cy w h)
        label_path = self.label_dir / (img_path.stem + '.txt')
        labels = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]]
                        labels.append([cls] + bbox)

        # Pad labels to fixed size (max 100 objects per image)
        max_objects = 100
        if labels:
            labels = torch.tensor(labels, dtype=torch.float32)
            if len(labels) > max_objects:
                labels = labels[:max_objects]
            pad = torch.zeros(max_objects - len(labels), 5)
            labels = torch.cat([labels, pad], dim=0)
        else:
            labels = torch.zeros(max_objects, 5)

        return img_tensor, labels


# ---------------------------------------------------------------------------
# Detection Loss
# ---------------------------------------------------------------------------

class DetectionLoss(nn.Module):
    """
    Simplified detection loss for tinyYOLO training.
    Components: objectness + classification + box regression.
    """

    def __init__(self, nc=80):
        super().__init__()
        self.nc = nc
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, predictions, targets):
        """
        Args:
            predictions: list of [B, 4+nc, H, W] from model
            targets: [B, max_objects, 5] (cls, cx, cy, w, h)
        """
        total_loss = torch.tensor(0.0, device=predictions[0].device)
        box_loss = torch.tensor(0.0, device=predictions[0].device)
        cls_loss = torch.tensor(0.0, device=predictions[0].device)
        obj_loss = torch.tensor(0.0, device=predictions[0].device)

        for pred in predictions:
            B, C, H, W = pred.shape
            # pred: [B, 4+nc, H, W]
            pred_box = pred[:, :4, :, :]   # [B, 4, H, W]
            pred_cls = pred[:, 4:, :, :]   # [B, nc, H, W]

            # Create target maps
            # For simplicity, compute loss against the raw prediction statistics
            # This ensures gradients flow and the model learns meaningful features

            # Valid targets mask
            valid = targets[:, :, 2] > 0  # [B, max_objects] — has width > 0

            # --- Objectness Loss ---
            # Target: cells containing object centers should have high response
            obj_target = torch.zeros(B, 1, H, W, device=pred.device)
            for b in range(B):
                for t in range(targets.shape[1]):
                    if targets[b, t, 2] > 0:  # valid target
                        cx, cy = targets[b, t, 1], targets[b, t, 2]
                        gi, gj = int(cx * W), int(cy * H)
                        gi = min(gi, W - 1)
                        gj = min(gj, H - 1)
                        obj_target[b, 0, gj, gi] = 1.0

            # Use max of class predictions as objectness proxy
            obj_pred = pred_cls.max(dim=1, keepdim=True)[0]
            obj_loss += self.bce(obj_pred, obj_target)

            # --- Classification Loss ---
            # For cells that have targets, supervise class predictions
            for b in range(B):
                for t in range(targets.shape[1]):
                    if targets[b, t, 2] > 0:
                        cls_id = int(targets[b, t, 0])
                        cx, cy = targets[b, t, 1], targets[b, t, 2]
                        gi, gj = int(cx * W), int(cy * H)
                        gi = min(gi, W - 1)
                        gj = min(gj, H - 1)

                        # Classification target (one-hot)
                        cls_target = torch.zeros(self.nc, device=pred.device)
                        if 0 <= cls_id < self.nc:
                            cls_target[cls_id] = 1.0
                        cls_loss += self.bce(pred_cls[b, :, gj, gi], cls_target)

                        # Box regression target
                        box_target = targets[b, t, 1:5].to(pred.device)
                        box_loss += self.mse(torch.sigmoid(pred_box[b, :, gj, gi]), box_target)

            n_targets = valid.sum().clamp(min=1)
            cls_loss = cls_loss / n_targets
            box_loss = box_loss / n_targets

        total_loss = obj_loss * 1.0 + cls_loss * 0.5 + box_loss * 5.0
        return total_loss, {
            'box': box_loss.item(),
            'cls': cls_loss.item(),
            'obj': obj_loss.item(),
            'total': total_loss.item(),
        }


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_single(args, imgsz, env):
    """Train a single model configuration."""
    train_cfg = get_training_config(env, imgsz)

    variant_tag = 'q' if args.variant == 'quantized' else 'std'
    model_name = args.name or f"tinyYOLO-{args.task}-{variant_tag}-{imgsz}"

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"  Task: {args.task} | Variant: {args.variant} | ImgSz: {imgsz}")
    print(f"{'='*60}")

    # Build model
    nc = 80 if args.task not in ('pose',) else 1
    model, model_info = build_model(task=args.task, variant=args.variant, nc=nc)

    params = count_parameters(model)
    print(f"  Parameters: {params['total_M']}M")

    try:
        flops = estimate_flops(model, imgsz, 'cpu')
        print(f"  GFLOPs:     {flops.get('flops_G', 'N/A')}")
    except Exception:
        pass

    # Settings
    epochs = 5 if args.quick else args.epochs
    batch = args.batch or train_cfg['batch']
    device = args.device or train_cfg['device']
    data_name = args.data or TASK_DATA_MAP.get(args.task, 'coco128.yaml')

    print(f"  Device:     {device}")
    print(f"  Batch:      {batch}")
    print(f"  Epochs:     {epochs}")
    print(f"  Data:       {data_name}")

    # Results directory
    results_dir = PROJECT_ROOT / 'experiments' / 'results' / model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Download and load dataset ---
    print(f"\n  Loading dataset...")
    data_dict = download_coco128()
    train_dir = data_dict.get('train', '')

    # Handle Ultralytics path format (may return Path or str)
    if isinstance(train_dir, (list, tuple)):
        train_dir = train_dir[0]
    train_dir = str(train_dir)
    print(f"  Train images: {train_dir}")

    dataset = SimpleDetectionDataset(train_dir, imgsz=imgsz, augment=True)
    dataloader = DataLoader(
        dataset, batch_size=batch, shuffle=True,
        num_workers=min(train_cfg['workers'], 4),
        pin_memory=(device != 'cpu'), drop_last=True,
    )
    print(f"  Dataset: {len(dataset)} images, {len(dataloader)} batches")

    # --- Setup training ---
    model = model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.lr * 0.01)

    # Use AMP if available
    use_amp = (device != 'cpu' and train_cfg.get('amp', False))
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    loss_fn = DetectionLoss(nc=nc)

    # EMA (Exponential Moving Average)
    ema_decay = 0.9999
    ema_model = {k: v.clone() for k, v in model.state_dict().items()}

    # --- Training loop ---
    history = []
    best_loss = float('inf')

    print(f"\n  {'Epoch':>6} {'Box':>10} {'Cls':>10} {'Obj':>10} {'Total':>10} {'LR':>10} {'Time':>8}")
    print(f"  {'-'*66}")

    for epoch in range(epochs):
        epoch_losses = {'box': 0, 'cls': 0, 'obj': 0, 'total': 0}
        n_batches = 0
        t0 = time.time()

        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # For seg/pose, use detection outputs
                    loss, loss_dict = loss_fn(outputs, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss, loss_dict = loss_fn(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

            # Accumulate losses
            for k in epoch_losses:
                epoch_losses[k] += loss_dict.get(k, 0)
            n_batches += 1

        scheduler.step()

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        # Update EMA
        with torch.no_grad():
            for k, v in model.state_dict().items():
                ema_model[k] = ema_decay * ema_model[k] + (1 - ema_decay) * v

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        print(f"  {epoch+1:>4}/{epochs} "
              f"{epoch_losses['box']:>10.4f} "
              f"{epoch_losses['cls']:>10.4f} "
              f"{epoch_losses['obj']:>10.4f} "
              f"{epoch_losses['total']:>10.4f} "
              f"{lr:>10.6f} "
              f"{elapsed:>7.1f}s")

        history.append({
            'epoch': epoch + 1,
            'lr': lr,
            'time': round(elapsed, 1),
            **epoch_losses,
        })

        # Save best
        if epoch_losses['total'] < best_loss:
            best_loss = epoch_losses['total']
            torch.save(model.state_dict(), results_dir / 'best.pt')

    # --- Save final results ---
    torch.save(model.state_dict(), results_dir / 'last.pt')
    torch.save(ema_model, results_dir / 'ema.pt')

    config = {
        'name': model_name,
        'task': args.task,
        'variant': args.variant,
        'nc': nc,
        'imgsz': imgsz,
        'epochs': epochs,
        'batch': batch,
        'device': str(device),
        'params_M': params['total_M'],
        'best_loss': round(best_loss, 4),
        'final_lr': lr,
        'platform': env['platform'],
        'timestamp': datetime.now().isoformat(),
    }

    with open(results_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    with open(results_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # --- Plot training curves ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        epochs_range = [h['epoch'] for h in history]

        for ax, key, color, title in [
            (axes[0], 'box', '#2196F3', 'Box Loss'),
            (axes[1], 'cls', '#FF9800', 'Classification Loss'),
            (axes[2], 'total', '#4CAF50', 'Total Loss'),
        ]:
            vals = [h[key] for h in history]
            ax.plot(epochs_range, vals, color=color, linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'{model_name} Training Curves', fontweight='bold')
        plt.tight_layout()
        plt.savefig(results_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Training curves saved: {results_dir / 'training_curves.png'}")
    except Exception as e:
        print(f"\n  [WARN] Could not plot training curves: {e}")

    print(f"\n  Results saved to: {results_dir}")
    print(f"  Best total loss:  {best_loss:.4f}")
    print(f"  Checkpoints:      best.pt, last.pt, ema.pt")

    return model_name


def main():
    args = parse_args()

    env = detect_environment()
    print_env_report(env)

    imgsz_list = [int(s.strip()) for s in args.imgsz.split(',')]

    if args.sweep or len(imgsz_list) > 1:
        print(f"\n  Running resolution sweep: {imgsz_list}")
        for imgsz in imgsz_list:
            train_single(args, imgsz, env)
    else:
        train_single(args, imgsz_list[0], env)

    print("\n  All training complete! ✓")


if __name__ == '__main__':
    main()
