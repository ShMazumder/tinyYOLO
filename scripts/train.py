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
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
from torchvision import transforms
import numpy as np
import random

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
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--warmup', type=int, default=3, help='Warmup epochs')
    return parser.parse_args()


def set_seed(seed=42):
    """Set deterministic training for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ---------------------------------------------------------------------------
# Task-Aligned Label Assignment (TAL)
# ---------------------------------------------------------------------------

class TALAssigner:
    """Task-Aligned Label Assignment following YOLOv8.

    Instead of assigning each GT to a single grid cell, TAL assigns
    multiple positive cells per GT based on an alignment metric that
    considers both classification score and IoU quality.

    This provides denser gradients critical for parameter-limited models
    where >99% of cells receive no positive signal under naive assignment.

    Args:
        topk: Number of top positive cells per GT (default 10).
        alpha: Classification score exponent in alignment metric.
        beta: IoU exponent in alignment metric.
    """

    def __init__(self, topk=10, alpha=0.5, beta=6.0):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def _box_iou_batch(boxes1, boxes2, eps=1e-7):
        """Compute pairwise IoU between two sets of (cx,cy,w,h) boxes.

        Args:
            boxes1: [N, 4] in (cx, cy, w, h) normalized format.
            boxes2: [M, 4] in (cx, cy, w, h) normalized format.

        Returns:
            iou_matrix: [N, M] IoU values.
        """
        # Convert to xyxy
        b1_x1 = boxes1[:, 0:1] - boxes1[:, 2:3] / 2
        b1_y1 = boxes1[:, 1:2] - boxes1[:, 3:4] / 2
        b1_x2 = boxes1[:, 0:1] + boxes1[:, 2:3] / 2
        b1_y2 = boxes1[:, 1:2] + boxes1[:, 3:4] / 2

        b2_x1 = boxes2[:, 0:1] - boxes2[:, 2:3] / 2
        b2_y1 = boxes2[:, 1:2] - boxes2[:, 3:4] / 2
        b2_x2 = boxes2[:, 0:1] + boxes2[:, 2:3] / 2
        b2_y2 = boxes2[:, 1:2] + boxes2[:, 3:4] / 2

        inter_x1 = torch.max(b1_x1, b2_x1.T)  # [N, M]
        inter_y1 = torch.max(b1_y1, b2_y1.T)
        inter_x2 = torch.min(b1_x2, b2_x2.T)
        inter_y2 = torch.min(b1_y2, b2_y2.T)

        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        area1 = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)  # [N, 1]
        area2 = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)  # [M, 1]

        union = area1 + area2.T - inter + eps
        return inter / union

    def assign(self, pred_scores, pred_bboxes, gt_labels, gt_bboxes, grid_cells, H, W):
        """Perform task-aligned assignment for a single image at one scale.

        Args:
            pred_scores: [N_cells, nc] predicted class scores (after sigmoid).
            pred_bboxes: [N_cells, 4] predicted boxes (cx, cy, w, h) normalized.
            gt_labels: [N_gt] integer class labels.
            gt_bboxes: [N_gt, 4] GT boxes (cx, cy, w, h) normalized.
            grid_cells: [N_cells, 2] grid cell center coordinates (normalized).
            H, W: grid height and width.

        Returns:
            pos_mask: [N_cells] boolean mask of positive cells.
            assigned_gt_idx: [N_pos] index into gt arrays for each positive.
            assigned_gt_labels: [N_pos] class label for each positive.
            assigned_gt_bboxes: [N_pos, 4] GT box for each positive.
            align_scores: [N_pos] alignment metric values for soft targets.
        """
        device = pred_scores.device
        n_gt = gt_bboxes.shape[0]
        n_cells = pred_scores.shape[0]
        nc = pred_scores.shape[1]

        if n_gt == 0:
            return (
                torch.zeros(n_cells, dtype=torch.bool, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, 4, device=device),
                torch.zeros(0, device=device),
            )

        # Step 1: Filter cells that are inside GT boxes (center prior)
        # A cell is a candidate for a GT if the cell center falls within the GT box
        cell_in_gt = torch.zeros(n_gt, n_cells, dtype=torch.bool, device=device)
        for g in range(n_gt):
            gx, gy, gw, gh = gt_bboxes[g]
            x1, y1 = gx - gw / 2, gy - gh / 2
            x2, y2 = gx + gw / 2, gy + gh / 2
            cx, cy = grid_cells[:, 0], grid_cells[:, 1]
            cell_in_gt[g] = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)

        # Step 2: Compute IoU between predictions and GTs
        iou_matrix = self._box_iou_batch(gt_bboxes, pred_bboxes)  # [n_gt, n_cells]

        # Step 3: Compute alignment metric: t = s^alpha * u^beta
        # s = predicted score for the GT class, u = IoU
        gt_cls_scores = torch.zeros(n_gt, n_cells, device=device)
        for g in range(n_gt):
            cls_id = gt_labels[g].long()
            if 0 <= cls_id < nc:
                gt_cls_scores[g] = pred_scores[:, cls_id]

        align_metric = gt_cls_scores.pow(self.alpha) * iou_matrix.pow(self.beta)  # [n_gt, n_cells]

        # Mask out cells not inside GT boxes
        align_metric = align_metric * cell_in_gt.float()

        # Step 4: Select top-k cells per GT
        topk = min(self.topk, n_cells)
        topk_metrics, topk_indices = align_metric.topk(topk, dim=1)  # [n_gt, topk]

        # Build positive mask
        pos_mask = torch.zeros(n_cells, dtype=torch.bool, device=device)
        assigned_gt = torch.full((n_cells,), -1, dtype=torch.long, device=device)
        assigned_metric = torch.zeros(n_cells, device=device)

        for g in range(n_gt):
            for k in range(topk):
                cell_idx = topk_indices[g, k].item()
                metric_val = topk_metrics[g, k].item()
                if metric_val <= 0:
                    continue
                # Resolve conflicts: if cell already assigned, keep higher metric
                if not pos_mask[cell_idx] or metric_val > assigned_metric[cell_idx]:
                    pos_mask[cell_idx] = True
                    assigned_gt[cell_idx] = g
                    assigned_metric[cell_idx] = metric_val

        # Gather outputs for positive cells
        pos_indices = pos_mask.nonzero(as_tuple=True)[0]
        gt_indices = assigned_gt[pos_indices]

        return (
            pos_mask,
            gt_indices,
            gt_labels[gt_indices] if len(gt_indices) > 0 else torch.zeros(0, dtype=torch.long, device=device),
            gt_bboxes[gt_indices] if len(gt_indices) > 0 else torch.zeros(0, 4, device=device),
            assigned_metric[pos_indices],
        )


# ---------------------------------------------------------------------------
# Dataset Loader — supports COCO128, VOC, COCO, and custom datasets
# ---------------------------------------------------------------------------

def _download_and_extract(url, dest, name="dataset"):
    """Download and extract a zip file."""
    import urllib.request, zipfile
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / f"{name}.zip"
    if not zip_path.exists():
        print(f"  Downloading {name}...")
        urllib.request.urlretrieve(url, zip_path)
    print(f"  Extracting {name}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest)
    zip_path.unlink()


def _download_voc(dest):
    """Download Pascal VOC dataset in YOLO format using Ultralytics."""
    voc_dir = dest / 'VOC'
    if (voc_dir / 'images' / 'train').exists():
        return  # Already downloaded

    print("  [INFO] Downloading Pascal VOC (YOLO format) via Ultralytics...")
    try:
        from ultralytics.data.utils import check_det_dataset
        # Use Ultralytics' built-in VOC.yaml which has correct download logic
        data_dict = check_det_dataset('VOC.yaml')
        # Ultralytics may put data in a different location — create symlink if needed
        ul_train = data_dict.get('train', '')
        if isinstance(ul_train, list):
            ul_train = ul_train[0]
        ul_train = Path(str(ul_train))

        if ul_train.exists() and not (voc_dir / 'images' / 'train').exists():
            # Create expected directory structure via symlinks
            voc_dir.mkdir(parents=True, exist_ok=True)
            (voc_dir / 'images').mkdir(exist_ok=True)
            (voc_dir / 'labels').mkdir(exist_ok=True)

            train_img = ul_train
            val_img = Path(str(data_dict.get('val', '')))
            train_lbl = Path(str(train_img).replace('images', 'labels'))
            val_lbl = Path(str(val_img).replace('images', 'labels'))

            for src, dst_name in [(train_img, 'train'), (val_img, 'val')]:
                dst = voc_dir / 'images' / dst_name
                if src.exists() and not dst.exists():
                    dst.symlink_to(src)
            for src, dst_name in [(train_lbl, 'train'), (val_lbl, 'val')]:
                dst = voc_dir / 'labels' / dst_name
                if src.exists() and not dst.exists():
                    dst.symlink_to(src)

        print(f"  [OK] VOC dataset ready at {voc_dir}")
        return
    except Exception as e:
        print(f"  [WARN] Ultralytics VOC download failed: {e}")
        print(f"  [INFO] Please download manually:")
        print(f"         pip install ultralytics")
        print(f"         python -c \"from ultralytics.data.utils import check_det_dataset; check_det_dataset('VOC.yaml')\"")
        raise


def load_dataset_config(data_name):
    """
    Load dataset from a YAML config or Ultralytics built-in name.

    Supports:
      - Local YAML:       datasets/voc.yaml, datasets/custom.yaml
      - Built-in names:   coco128.yaml, coco.yaml, VOC.yaml
      - Absolute paths:   /path/to/my_data.yaml

    Returns dict with keys: train, val, nc, names
    """
    import yaml

    # Try to set Ultralytics datasets_dir to PROJECT_ROOT
    # This ensures that 'datasets/NAME' paths in YAMLs resolve correctly
    try:
        from ultralytics.utils import settings
        settings.update({'datasets_dir': str(PROJECT_ROOT)})
    except Exception:
        pass

    # 1. Check local datasets/ folder first
    local_yaml = PROJECT_ROOT / 'datasets' / data_name
    if not str(data_name).endswith('.yaml'):
        local_yaml = PROJECT_ROOT / 'datasets' / f'{data_name}.yaml'

    if local_yaml.exists():
        with open(local_yaml) as f:
            cfg = yaml.safe_load(f)

        # Resolve paths relative to datasets folder by default
        path_val = cfg.get('path', '.')
        if Path(path_val).is_absolute():
            base = Path(path_val)
        elif str(path_val).startswith('datasets'):
            # Backward compatibility: if path already has datasets/, resolve from root
            base = PROJECT_ROOT / path_val
        else:
            base = PROJECT_ROOT / 'datasets' / path_val
        raw_train = cfg.get('train', 'images/train')
        raw_val = cfg.get('val', 'images/val')

        # Handle list-type paths (e.g., VOC has multiple train dirs)
        if isinstance(raw_train, list):
            train_dirs = [str(base / d) for d in raw_train]
            train_dir = train_dirs[0]  # Use first for existence check
        else:
            train_dirs = [str(base / raw_train)]
            train_dir = train_dirs[0]

        if isinstance(raw_val, list):
            val_dirs = [str(base / d) for d in raw_val]
            val_dir = val_dirs[0]
        else:
            val_dirs = [str(base / raw_val)]
            val_dir = val_dirs[0]

        # Check if data exists — try auto-download if not
        if not Path(train_dir).exists():
            print(f"  [INFO] Dataset not found at {train_dir}")

            # Try dataset-specific downloads
            if 'voc' in data_name.lower():
                try:
                    _download_voc(PROJECT_ROOT / 'datasets')
                except Exception:
                    pass
                # If our expected paths still don't exist, use Ultralytics paths directly
                if not Path(train_dir).exists():
                    try:
                        from ultralytics.data.utils import check_det_dataset
                        data_dict = check_det_dataset('VOC.yaml')
                        # Return Ultralytics paths with our nc/names
                        ul_train = data_dict.get('train', '')
                        if isinstance(ul_train, list):
                            ul_train = ul_train[0]
                        return {
                            'train': str(ul_train),
                            'val': str(data_dict.get('val', '')),
                            'nc': cfg.get('nc', 20),
                            'names': cfg.get('names', data_dict.get('names', {})),
                        }
                    except Exception:
                        pass
            elif 'coco128' in data_name.lower():
                _download_and_extract(
                    'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip',
                    PROJECT_ROOT / 'datasets', 'coco128'
                )
            else:
                # Try Ultralytics with full path to our YAML
                try:
                    from ultralytics.data.utils import check_det_dataset
                    data_dict = check_det_dataset(str(local_yaml))
                    return data_dict
                except Exception:
                    pass

            # Re-check after download attempt
            if not Path(train_dir).exists():
                print(f"  [WARN] Auto-download failed. Please download manually.")
                print(f"  [INFO] Expected layout:")
                print(f"         {base}/")
                print(f"         ├── images/train/  (or images/train2017/)")
                print(f"         ├── images/val/")
                print(f"         ├── labels/train/")
                print(f"         └── labels/val/")
                raise FileNotFoundError(f"Dataset not found: {train_dir}")

        return {
            'train': train_dirs if len(train_dirs) > 1 else train_dir,
            'val': val_dirs if len(val_dirs) > 1 else val_dir,
            'nc': cfg.get('nc', 80),
            'names': cfg.get('names', {i: f'cls_{i}' for i in range(cfg.get('nc', 80))}),
        }

    # 2. Try absolute path
    if Path(data_name).exists():
        with open(data_name) as f:
            cfg = yaml.safe_load(f)
        base = Path(cfg.get('path', '.')).resolve()
        return {
            'train': str(base / cfg.get('train', 'images/train')),
            'val': str(base / cfg.get('val', 'images/val')),
            'nc': cfg.get('nc', 80),
            'names': cfg.get('names', {}),
        }

    # 3. Fallback: Ultralytics built-in download
    try:
        from ultralytics.data.utils import check_det_dataset
        data_dict = check_det_dataset(data_name)
        return data_dict
    except Exception as e:
        print(f"  [WARN] Ultralytics download failed: {e}")

    # 4. Manual COCO128 fallback
    if 'coco128' in str(data_name):
        dest = PROJECT_ROOT / "datasets"
        if not (dest / "coco128").exists():
            _download_and_extract(
                'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip',
                dest, 'coco128'
            )
        return {'train': str(dest / 'coco128' / 'images' / 'train2017'),
                'val': str(dest / 'coco128' / 'images' / 'train2017'),
                'nc': 80, 'names': {i: f'cls_{i}' for i in range(80)}}

    raise FileNotFoundError(f"Dataset config not found: {data_name}\n"
                           f"  Tried: {local_yaml}, Ultralytics built-in, manual fallback")


class SimpleDetectionDataset(Dataset):
    """
    Loads images + YOLO-format labels from one or more directories.
    Works with COCO128, VOC, COCO, and custom YOLO-format datasets.
    """

    def __init__(self, img_dir, imgsz=320, augment=True):
        # Support single path or list of paths (e.g., VOC train2007 + train2012)
        if isinstance(img_dir, (list, tuple)):
            img_dirs = [Path(d) for d in img_dir]
        else:
            img_dirs = [Path(img_dir)]

        self.img_dirs = img_dirs
        self.imgsz = imgsz
        self.augment = augment

        # Find all images across all directories
        self.img_files = []
        for d in img_dirs:
            target_dir = d
            if not target_dir.exists():
                # Smart search: if 'datasets/coco/images/val2017' doesn't exist,
                # search for any 'val2017' folder under 'datasets/'
                print(f"  [INFO] {target_dir} not found, searching for '{target_dir.name}' elsewhere...")
                import glob
                search_root = PROJECT_ROOT / 'datasets'
                matches = glob.glob(str(search_root / "**" / target_dir.name), recursive=True)
                if matches:
                    target_dir = Path(matches[0])
                    print(f"  [OK] Found data at {target_dir}")

            if target_dir.exists():
                # Primary search: direct children
                found = sorted([
                    f for f in target_dir.iterdir()
                    if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
                ])
                if not found:
                    # Secondary search: recursive
                    import glob
                    found = sorted([
                        Path(f) for f in glob.glob(str(target_dir / "**" / "*"), recursive=True)
                        if Path(f).suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
                    ])
                self.img_files.extend(found)
        
        if not self.img_files:
            import os
            print(f"\n  [ERROR] No images found in {img_dirs}")
            # Diagnostic: check parent directory contents
            for d in img_dirs:
                p = d
                while not p.exists() and p != p.parent:
                    p = p.parent
                if p.exists():
                    print(f"  [DEBUG] Contents of {p}:")
                    try:
                        for item in sorted(p.iterdir()):
                            suffix = "/" if item.is_dir() else ""
                            print(f"    - {item.name}{suffix}")
                    except: pass
            raise FileNotFoundError(f"No images found in {img_dirs}")

        # Universal label root discovery
        self.label_root = None
        print(f"  [INFO] Searching for labels folder under 'datasets/'...")
        import glob
        search_root = PROJECT_ROOT / 'datasets'
        # Look for any folder named 'labels'
        matches = glob.glob(str(search_root / "**" / "labels"), recursive=True)
        if matches:
            # Pick the first one that looks like it has content
            for m in matches:
                p = Path(m)
                if any(p.iterdir()):
                    self.label_root = p
                    print(f"  [OK] Found label root at {self.label_root}")
                    break
        
        if not self.label_root:
            print(f"  [WARN] No 'labels' folder found under 'datasets/'!")
            print(f"         Losses will likely be zero.")

        # Transforms — YOLO-standard augmentation pipeline
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((imgsz, imgsz)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.015),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((imgsz, imgsz)),
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
        # Derive label path from image path (robust search)
        img_path = Path(img_path)
        label_path = img_path.with_suffix('.txt') # Default fallback
        
        # Strategy 1: Use pre-discovered label_root if available
        if self.label_root:
            # Try to match the image path's structure within the label_root
            # e.g., images/val2017/001.jpg -> label_root/val2017/001.txt
            # or just label_root/001.txt
            try_paths = [
                (self.label_root / img_path.parent.name / img_path.name).with_suffix('.txt'),
                (self.label_root / img_path.name).with_suffix('.txt')
            ]
            for tp in try_paths:
                if tp.exists():
                    label_path = tp
                    break
        
        # Strategy 2: Standard YOLO fallback (replace 'images' with 'labels')
        if not label_path.exists():
            try_path = Path(str(img_path).replace('images', 'labels')).with_suffix('.txt')
            if try_path.exists():
                label_path = try_path

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


class MosaicDataset(Dataset):
    """Wraps SimpleDetectionDataset to produce 4-image mosaics.

    Combines 4 random images into a single training image with a random
    center point. Labels are adjusted to mosaic coordinates. This is a
    key augmentation in all YOLO variants since v4, providing:
      - Implicit batch normalization (4 images per sample)
      - Better small object representation
      - Natural context enrichment

    Args:
        base_dataset: Underlying SimpleDetectionDataset.
        imgsz: Output mosaic image size.
        enable: Whether mosaic is active (disabled for last N% of training).
    """

    def __init__(self, base_dataset, imgsz=416, enable=True):
        self.base = base_dataset
        self.imgsz = imgsz
        self.enable = enable
        # Keep a reference to img_files for compatibility
        self.img_files = base_dataset.img_files

    def __len__(self):
        return len(self.base)

    def set_mosaic(self, enable):
        """Enable or disable mosaic (call at epoch boundary)."""
        self.enable = enable

    def __getitem__(self, idx):
        if not self.enable:
            return self.base[idx]

        # Select 4 images: current + 3 random
        indices = [idx] + [random.randint(0, len(self.base) - 1) for _ in range(3)]

        # Random center point for the mosaic (within middle 40-60% of image)
        cx = random.randint(int(self.imgsz * 0.3), int(self.imgsz * 0.7))
        cy = random.randint(int(self.imgsz * 0.3), int(self.imgsz * 0.7))

        # Placement regions: top-left, top-right, bottom-left, bottom-right
        # Each sub-image is resized to fit its allocated quadrant
        mosaic_img = torch.zeros(3, self.imgsz, self.imgsz)
        mosaic_labels = []

        placements = [
            (0, 0, cx, cy),                             # top-left
            (cx, 0, self.imgsz, cy),                     # top-right
            (0, cy, cx, self.imgsz),                     # bottom-left
            (cx, cy, self.imgsz, self.imgsz),             # bottom-right
        ]

        for i, (x1, y1, x2, y2) in enumerate(placements):
            img_tensor, labels = self.base[indices[i]]
            # img_tensor is [3, H, W], labels is [max_objects, 5]

            qw = max(x2 - x1, 1)
            qh = max(y2 - y1, 1)

            # Resize sub-image to fill its quadrant
            resized = torch.nn.functional.interpolate(
                img_tensor.unsqueeze(0), size=(qh, qw), mode='bilinear', align_corners=False
            ).squeeze(0)
            mosaic_img[:, y1:y1+qh, x1:x1+qw] = resized

            # Adjust labels: transform from [0,1] sub-image coords to mosaic coords
            for t in range(labels.shape[0]):
                if labels[t, 2] > 0:  # valid target
                    cls_id = labels[t, 0]
                    # Original normalized coords in sub-image
                    ocx, ocy, ow, oh = labels[t, 1:5]
                    # Map to mosaic pixel coords
                    new_cx = (x1 + ocx * qw) / self.imgsz
                    new_cy = (y1 + ocy * qh) / self.imgsz
                    new_w = (ow * qw) / self.imgsz
                    new_h = (oh * qh) / self.imgsz
                    # Clip to valid range
                    new_cx = max(0.0, min(1.0, new_cx.item() if torch.is_tensor(new_cx) else new_cx))
                    new_cy = max(0.0, min(1.0, new_cy.item() if torch.is_tensor(new_cy) else new_cy))
                    new_w = max(0.001, min(1.0, new_w.item() if torch.is_tensor(new_w) else new_w))
                    new_h = max(0.001, min(1.0, new_h.item() if torch.is_tensor(new_h) else new_h))
                    mosaic_labels.append([cls_id.item() if torch.is_tensor(cls_id) else cls_id,
                                          new_cx, new_cy, new_w, new_h])

        # Pad mosaic labels to fixed size
        max_objects = 100
        if mosaic_labels:
            ml = torch.tensor(mosaic_labels[:max_objects], dtype=torch.float32)
            if len(ml) < max_objects:
                pad = torch.zeros(max_objects - len(ml), 5)
                ml = torch.cat([ml, pad], dim=0)
        else:
            ml = torch.zeros(max_objects, 5)

        return mosaic_img, ml


# ---------------------------------------------------------------------------
# Detection Loss
# ---------------------------------------------------------------------------

class DetectionLoss(nn.Module):
    """
    Detection loss for tinyYOLO with CIoU box regression.
    Components: CIoU box loss + BCE classification + BCE objectness.
    Loss weights tuned for sub-1M parameter models:
        total = 2.0 × CIoU + 1.0 × cls + 1.0 × obj
    Note: Full-size YOLO uses 7.5× but this overwhelms cls/obj
    for tiny models where CIoU magnitude is ~0.8–1.0.
    """

    def __init__(self, nc=80):
        super().__init__()
        self.nc = nc
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        # Loss weights tuned for tiny models (CIoU starts ~1.0 for small models)
        self.box_weight = 2.0
        self.cls_weight = 1.0
        self.obj_weight = 1.0

    @staticmethod
    def _ciou(pred_box, target_box, eps=1e-7):
        """
        Compute CIoU loss between predicted and target boxes.
        Both in (cx, cy, w, h) format, values in [0, 1].

        Returns:
            ciou_loss: scalar (1 - CIoU), lower is better.
        """
        # Convert to xyxy
        px1 = pred_box[0] - pred_box[2] / 2
        py1 = pred_box[1] - pred_box[3] / 2
        px2 = pred_box[0] + pred_box[2] / 2
        py2 = pred_box[1] + pred_box[3] / 2

        tx1 = target_box[0] - target_box[2] / 2
        ty1 = target_box[1] - target_box[3] / 2
        tx2 = target_box[0] + target_box[2] / 2
        ty2 = target_box[1] + target_box[3] / 2

        # Intersection
        inter_x1 = torch.max(px1, tx1)
        inter_y1 = torch.max(py1, ty1)
        inter_x2 = torch.min(px2, tx2)
        inter_y2 = torch.min(py2, ty2)
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        # Union
        pred_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
        target_area = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
        union = pred_area + target_area - inter + eps

        iou = inter / union

        # Enclosing box
        enc_x1 = torch.min(px1, tx1)
        enc_y1 = torch.min(py1, ty1)
        enc_x2 = torch.max(px2, tx2)
        enc_y2 = torch.max(py2, ty2)

        # Distance between centers
        cx_diff = pred_box[0] - target_box[0]
        cy_diff = pred_box[1] - target_box[1]
        rho2 = cx_diff ** 2 + cy_diff ** 2

        # Diagonal of enclosing box
        c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

        # Aspect ratio consistency (v term)
        w_pred = (px2 - px1).clamp(min=eps)
        h_pred = (py2 - py1).clamp(min=eps)
        w_target = (tx2 - tx1).clamp(min=eps)
        h_target = (ty2 - ty1).clamp(min=eps)

        v = (4 / (math.pi ** 2)) * (
            torch.atan(w_target / h_target) - torch.atan(w_pred / h_pred)
        ) ** 2
        alpha = v / (1 - iou + v + eps)

        ciou = iou - rho2 / c2 - alpha * v
        return 1.0 - ciou  # Loss: 0 when perfect match

    def forward(self, predictions, targets):
        """
        Args:
            predictions: list of [B, 5+nc, H, W] from model (4 bbox + 1 obj + nc cls)
            targets: [B, max_objects, 5] (cls, cx, cy, w, h)
        """
        device = predictions[0].device
        total_box = torch.tensor(0.0, device=device)
        total_cls = torch.tensor(0.0, device=device)
        total_obj = torch.tensor(0.0, device=device)

        # Count total positive targets ONCE (Fix: no longer inflated per-scale)
        N_pos = 0
        for b in range(targets.shape[0]):
            for t in range(targets.shape[1]):
                if targets[b, t, 2] > 0:  # valid target
                    N_pos += 1
        N_pos = max(N_pos, 1)

        for pred in predictions:
            B, C, H, W = pred.shape
            pred_box = pred[:, :4, :, :]     # [B, 4, H, W]
            pred_obj = pred[:, 4:5, :, :]    # [B, 1, H, W] — dedicated objectness
            pred_cls = pred[:, 5:, :, :]     # [B, nc, H, W]

            # --- Objectness target map ---
            obj_target = torch.zeros(B, 1, H, W, device=device)

            for b in range(B):
                for t in range(targets.shape[1]):
                    if targets[b, t, 2] > 0:  # valid target (has width > 0)
                        cx, cy = targets[b, t, 1], targets[b, t, 2]
                        gi = min(int(cx * W), W - 1)
                        gj = min(int(cy * H), H - 1)
                        obj_target[b, 0, gj, gi] = 1.0

            # Objectness: dedicated objectness head output
            total_obj += self.bce(pred_obj, obj_target)

            # --- Per-target CIoU + Classification ---
            for b in range(B):
                for t in range(targets.shape[1]):
                    if targets[b, t, 2] > 0:
                        cls_id = int(targets[b, t, 0])
                        cx, cy, w, h = targets[b, t, 1:5]
                        gi = min(int(cx * W), W - 1)
                        gj = min(int(cy * H), H - 1)

                        # Classification (BCE)
                        cls_target = torch.zeros(self.nc, device=device)
                        if 0 <= cls_id < self.nc:
                            cls_target[cls_id] = 1.0
                        total_cls += self.bce(pred_cls[b, :, gj, gi], cls_target)

                        # CIoU box loss
                        pred_cxcywh = torch.sigmoid(pred_box[b, :, gj, gi])
                        target_cxcywh = targets[b, t, 1:5].to(device)
                        total_box += self._ciou(pred_cxcywh, target_cxcywh)

        # Normalize by total positive target count (computed once, not per-scale)
        total_box = total_box / N_pos
        total_cls = total_cls / N_pos

        # Weighted sum
        total_loss = (self.box_weight * total_box +
                      self.cls_weight * total_cls +
                      self.obj_weight * total_obj)

        return total_loss, {
            'box': total_box.item(),
            'cls': total_cls.item(),
            'obj': total_obj.item(),
            'total': total_loss.item(),
        }


def apply_yolo_batchnorm_defaults(model):
    """
    Apply YOLO-standard BatchNorm parameters.
    All official YOLO models use eps=1e-3, momentum=0.03
    instead of PyTorch defaults (eps=1e-5, momentum=0.1).
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eps = 1e-3
            m.momentum = 0.03


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

    # Set deterministic training for reproducibility
    set_seed(args.seed)
    print(f"  Seed:       {args.seed} (deterministic training enabled)")

    # Settings
    epochs = 5 if args.quick else args.epochs
    batch = args.batch or train_cfg['batch']
    device = args.device or train_cfg['device']
    data_name = args.data or TASK_DATA_MAP.get(args.task, 'coco128.yaml')

    # --- Load dataset config first (to get nc) ---
    print(f"\n  Loading dataset...")
    data_dict = load_dataset_config(data_name)
    train_dir = data_dict.get('train', '')
    val_dir = data_dict.get('val', '')
    nc = data_dict.get('nc', 80 if args.task not in ('pose',) else 1)

    # Validate that train and val directories are different (prevent data leakage)
    train_str = str(train_dir[0]) if isinstance(train_dir, (list, tuple)) else str(train_dir)
    val_str = str(val_dir[0]) if isinstance(val_dir, (list, tuple)) else str(val_dir)
    if val_str == train_str:
        print(f"  [WARN] Validation dir same as training dir — possible data leakage!")
        print(f"  [WARN] train: {train_str}")
        print(f"  [WARN] val:   {val_str}")
        print(f"  [INFO] Consider using --data with a YAML containing separate train/val paths.")

    # Build model with correct nc for this dataset
    model, model_info = build_model(task=args.task, variant=args.variant, nc=nc)

    params = count_parameters(model)
    print(f"  Parameters: {params['total_M']}M")

    try:
        flops = estimate_flops(model, imgsz, 'cpu')
        print(f"  GFLOPs:     {flops.get('flops_G', 'N/A')}")
    except Exception:
        pass

    print(f"  Device:     {device}")
    print(f"  Batch:      {batch}")
    print(f"  Epochs:     {epochs}")
    print(f"  Data:       {data_name} (nc={nc})")

    # Results directory
    results_dir = PROJECT_ROOT / 'experiments' / 'results' / model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Display train path(s)
    if isinstance(train_dir, (list, tuple)):
        print(f"  Train images: {len(train_dir)} directories")
        for d in train_dir:
            print(f"    - {d}")
    else:
        print(f"  Train images: {train_dir}")

    # SimpleDetectionDataset handles both single path and list of paths
    base_dataset = SimpleDetectionDataset(train_dir, imgsz=imgsz, augment=True)
    # Wrap with mosaic augmentation (disabled for quick tests and last 10% of epochs)
    use_mosaic = not args.quick and epochs > 10
    mosaic_disable_epoch = int(epochs * 0.9) if use_mosaic else 0
    dataset = MosaicDataset(base_dataset, imgsz=imgsz, enable=use_mosaic)
    dataloader = DataLoader(
        dataset, batch_size=batch, shuffle=True,
        num_workers=min(train_cfg['workers'], 4),
        pin_memory=(device != 'cpu'), drop_last=True,
    )
    print(f"  Dataset: {len(dataset)} images, {len(dataloader)} batches")
    if use_mosaic:
        print(f"  Mosaic:  ON (disabled after epoch {mosaic_disable_epoch})")

    # --- Setup training ---
    model = model.to(device)

    # Apply YOLO-standard BatchNorm (eps=1e-3, momentum=0.03)
    apply_yolo_batchnorm_defaults(model)

    model.train()

    # Separate weight decay groups (following pfeatherstone/tinyyolo)
    # Weights (dim >= 2) get weight decay, biases/BN params don't
    wd_params = [p for p in model.parameters() if p.dim() >= 2]
    no_wd_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = optim.AdamW([
        {'params': wd_params, 'weight_decay': 1e-4},
        {'params': no_wd_params, 'weight_decay': 0.0},
    ], lr=args.lr, betas=(0.9, 0.999))

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.lr * 0.01)

    # Warmup configuration
    warmup_epochs = args.warmup
    warmup_bias_lr = 0.1  # Bias LR during warmup (higher for stable start)
    base_lr = args.lr
    print(f"  Warmup:     {warmup_epochs} epochs (linear)")

    # Use AMP if available
    use_amp = (device != 'cpu' and train_cfg.get('amp', False))
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    loss_fn = DetectionLoss(nc=nc)

    # EMA (Exponential Moving Average)
    ema_decay = 0.9999
    ema_model = {k: v.clone() for k, v in model.state_dict().items()}

    # Evaluation imports
    from tinyYOLO.utils.postprocess import decode_predictions, non_max_suppression, decode_targets
    from tinyYOLO.utils.metrics import (
        DetectionMetrics, print_metrics_report,
        plot_training_curves, generate_full_report,
    )

    # Eval frequency: every N epochs (and always on last epoch)
    eval_every = max(1, epochs // 10) if epochs > 5 else 1

    # Validation dataloader — uses SEPARATE val directory (no data leakage)
    if not val_dir:
        print(f"  [WARN] No validation directory specified, using training dir (not recommended)")
        val_dir = train_dir
    val_dataset = SimpleDetectionDataset(val_dir, imgsz=imgsz, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False,
                            num_workers=min(train_cfg['workers'], 4),
                            pin_memory=(device != 'cpu'))
    print(f"  Val Dataset: {len(val_dataset)} images")

    # --- Training loop ---
    history = []
    best_loss = float('inf')
    best_map = 0.0

    print(f"\n  {'Epoch':>6} {'Box':>8} {'Cls':>8} {'Obj':>8} {'Total':>8} "
          f"{'P':>6} {'R':>6} {'F1':>6} {'mAP50':>7} {'LR':>10} {'Time':>6}")
    print(f"  {'-'*88}")

    for epoch in range(epochs):
        model.train()
        epoch_losses = {'box': 0, 'cls': 0, 'obj': 0, 'total': 0}
        n_batches = 0
        t0 = time.time()

        # Disable mosaic for final 10% of training (improves fine-grained learning)
        if use_mosaic and epoch >= mosaic_disable_epoch:
            dataset.set_mosaic(False)

        n_total_iters = len(dataloader)

        # Wrap with tqdm for real-time batch progress
        batch_iter = enumerate(dataloader)
        if tqdm is not None:
            pbar = tqdm(
                batch_iter, total=len(dataloader),
                desc=f"  Epoch {epoch+1}/{epochs}",
                bar_format='{l_bar}{bar:30}{r_bar}',
                leave=False, file=sys.stdout
            )
        else:
            pbar = batch_iter

        for batch_idx, (images, targets) in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # --- Linear LR warmup ---
            if epoch < warmup_epochs:
                # Compute global warmup progress [0, 1]
                warmup_iters = warmup_epochs * n_total_iters
                cur_iter = epoch * n_total_iters + batch_idx
                warmup_progress = cur_iter / max(warmup_iters, 1)
                # Scale learning rate linearly from ~0 to base_lr
                for pg in optimizer.param_groups:
                    pg['lr'] = base_lr * warmup_progress

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
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

            for k in epoch_losses:
                epoch_losses[k] += loss_dict.get(k, 0)
            n_batches += 1

            # Update tqdm description with running loss
            if tqdm is not None and isinstance(pbar, tqdm):
                avg_loss = epoch_losses['total'] / n_batches
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'}, refresh=True)

        scheduler.step()

        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        # Update EMA
        with torch.no_grad():
            for k, v in model.state_dict().items():
                ema_model[k] = ema_decay * ema_model[k] + (1 - ema_decay) * v

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        # --- Evaluation ---
        epoch_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'mAP50': 0, 'mAP50_95': 0}
        is_eval_epoch = ((epoch + 1) % eval_every == 0) or (epoch + 1 == epochs)

        if is_eval_epoch:
            model.eval()
            det_metrics = DetectionMetrics(nc=nc, conf_thresh=0.15)

            with torch.no_grad():
                for val_images, val_targets in val_loader:
                    val_images = val_images.to(device, non_blocking=True)
                    val_targets = val_targets.to(device, non_blocking=True)

                    val_outputs = model(val_images)
                    if isinstance(val_outputs, tuple):
                        val_outputs = val_outputs[0]

                    # Decode predictions → [N, 6] (x1,y1,x2,y2,conf,cls)
                    pred_list = decode_predictions(val_outputs, imgsz, conf_thresh=0.15, nc=nc)
                    pred_list = non_max_suppression(pred_list, iou_thresh=0.45)

                    # Decode ground truth → [M, 5] (x1,y1,x2,y2,cls)
                    gt_list = decode_targets(val_targets, imgsz)

                    det_metrics.update(pred_list, gt_list)

            epoch_metrics = det_metrics.compute()
            model.train()

        # Print row
        p = epoch_metrics.get('precision', 0)
        r = epoch_metrics.get('recall', 0)
        f1 = epoch_metrics.get('f1', 0)
        m50 = epoch_metrics.get('mAP50', 0)

        # Close tqdm bar before printing epoch summary
        if tqdm is not None and isinstance(pbar, tqdm):
            pbar.close()

        print(f"  {epoch+1:>4}/{epochs} "
              f"{epoch_losses['box']:>8.4f} {epoch_losses['cls']:>8.4f} "
              f"{epoch_losses['obj']:>8.4f} {epoch_losses['total']:>8.4f} "
              f"{p:>6.3f} {r:>6.3f} {f1:>6.3f} {m50:>7.4f} "
              f"{lr:>10.6f} {elapsed:>5.1f}s", flush=True)

        history.append({
            'epoch': epoch + 1,
            'lr': lr,
            'time': round(elapsed, 1),
            **epoch_losses,
            'precision': round(p, 4),
            'recall': round(r, 4),
            'f1': round(f1, 4),
            'mAP50': round(m50, 4),
            'mAP50_95': round(epoch_metrics.get('mAP50_95', 0), 4),
        })

        # Save best (by mAP if available, else by loss)
        def _clean_state_dict(sd):
            """Strip thop profiler keys from state dict."""
            return {k: v for k, v in sd.items()
                    if not k.endswith(('total_ops', 'total_params'))}

        if is_eval_epoch and m50 > best_map:
            best_map = m50
            torch.save(_clean_state_dict(model.state_dict()), results_dir / 'best.pt')
        elif epoch_losses['total'] < best_loss:
            best_loss = epoch_losses['total']
            if not is_eval_epoch:
                torch.save(_clean_state_dict(model.state_dict()), results_dir / 'best.pt')

    # --- Save final results ---
    torch.save(_clean_state_dict(model.state_dict()), results_dir / 'last.pt')
    torch.save(ema_model, results_dir / 'ema.pt')

    # Final evaluation with full report
    print(f"\n  Running final evaluation...")
    model.eval()
    final_metrics_calc = DetectionMetrics(nc=nc, conf_thresh=0.15)

    with torch.no_grad():
        for val_images, val_targets in val_loader:
            val_images = val_images.to(device, non_blocking=True)
            val_targets = val_targets.to(device, non_blocking=True)

            val_outputs = model(val_images)
            if isinstance(val_outputs, tuple):
                val_outputs = val_outputs[0]

            pred_list = decode_predictions(val_outputs, imgsz, conf_thresh=0.15, nc=nc)
            pred_list = non_max_suppression(pred_list, iou_thresh=0.45)
            gt_list = decode_targets(val_targets, imgsz)
            final_metrics_calc.update(pred_list, gt_list)

    final_metrics = final_metrics_calc.compute()
    print_metrics_report(final_metrics)

    # Save config with final metrics
    config = {
        'name': model_name,
        'task': args.task,
        'variant': args.variant,
        'nc': nc,
        'imgsz': imgsz,
        'epochs': epochs,
        'batch': batch,
        'lr': args.lr,
        'device': str(device),
        'amp': use_amp,
        'params_M': params['total_M'],
        'best_loss': round(best_loss, 4),
        'best_mAP50': round(best_map, 4),
        'final_lr': lr,
        'platform': env['platform'],
        'timestamp': datetime.now().isoformat(),
        'seed': args.seed,
        'warmup_epochs': warmup_epochs,
        'mosaic': use_mosaic,
        'mosaic_disable_epoch': mosaic_disable_epoch if use_mosaic else None,
        'augmentation': {
            'resize': imgsz,
            'color_jitter': {'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.4, 'hue': 0.015},
            'random_grayscale': 0.1,
            'horizontal_flip': 0.5,
            'random_perspective': {'distortion_scale': 0.15, 'p': 0.3},
            'mosaic': {'enabled': use_mosaic, 'disabled_after_epoch': mosaic_disable_epoch if use_mosaic else None},
        },
        'loss': {
            'type': 'CIoU + BCE',
            'box_weight': 2.0,
            'cls_weight': 1.0,
            'obj_weight': 1.0,
        },
        'optimizer': {
            'type': 'AdamW',
            'lr': args.lr,
            'betas': [0.9, 0.999],
            'weight_decay': '1e-4 (weights only, biases/BN excluded)',
        },
        'batchnorm': {
            'eps': 1e-3,
            'momentum': 0.03,
        },
        'scheduler': {
            'type': 'CosineAnnealingLR',
            'T_max': epochs,
            'eta_min': args.lr * 0.01,
        },
        'warmup': {
            'type': 'linear',
            'epochs': warmup_epochs,
        },
        'final_metrics': {
            'precision': final_metrics['precision'],
            'recall': final_metrics['recall'],
            'f1': final_metrics['f1'],
            'mAP50': final_metrics['mAP50'],
            'mAP50_95': final_metrics['mAP50_95'],
        },
    }

    with open(results_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    with open(results_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Generate full report (curves, confusion matrix, per-class)
    try:
        generate_full_report(final_metrics, history, results_dir)
    except Exception as e:
        print(f"  [WARN] Report generation error: {e}")
        # Fallback: basic loss curves
        try:
            plot_training_curves(history, results_dir / 'training_curves.png')
        except Exception:
            pass

    print(f"\n  Results saved to: {results_dir}")
    print(f"  Best mAP@50:      {best_map:.4f}")
    print(f"  Best total loss:   {best_loss:.4f}")
    print(f"  Outputs:")
    print(f"    best.pt, last.pt, ema.pt      — Model checkpoints")
    print(f"    config.json                    — Hyperparameters & augmentation report")
    print(f"    history.json                   — Per-epoch losses + metrics")
    print(f"    metrics.json                   — Final P/R/F1/mAP/confusion matrix")
    print(f"    training_curves.png            — Loss + accuracy curves")
    print(f"    confusion_matrix.png           — Confusion matrix heatmap")
    print(f"    per_class_report.txt           — Per-class P/R/F1/AP breakdown")

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
