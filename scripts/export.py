"""
TinyYOLO Export Script
========================
Export trained models to multiple deployment formats.

Usage:
    python scripts/export.py --weights path/to/model.pt --task det --formats onnx,tflite
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tinyYOLO.models import build_model
from tinyYOLO.utils.env import detect_environment


def parse_args():
    parser = argparse.ArgumentParser(description='TinyYOLO Export')
    parser.add_argument('--weights', type=str, required=True, help='Model weights path')
    parser.add_argument('--task', type=str, default='det', help='Task type')
    parser.add_argument('--variant', type=str, default='standard', help='Variant')
    parser.add_argument('--imgsz', type=int, default=320, help='Input image size')
    parser.add_argument('--formats', type=str, default='onnx',
                        help='Export formats (comma-separated): onnx,tflite,coreml,torchscript')
    parser.add_argument('--fp16', action='store_true', help='Export with FP16')
    parser.add_argument('--int8', action='store_true', help='Export with INT8 quantization')
    return parser.parse_args()


def export_onnx(model, imgsz, output_path, fp16=False):
    """Export to ONNX format."""
    import torch

    model.eval()
    dummy = torch.randn(1, 3, imgsz, imgsz)

    torch.onnx.export(
        model, dummy, str(output_path),
        opset_version=18,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={'images': {0: 'batch'}, 'output': {0: 'batch'}},
    )
    print(f"  Exported ONNX: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


def export_torchscript(model, imgsz, output_path):
    """Export to TorchScript format."""
    import torch

    model.eval()
    dummy = torch.randn(1, 3, imgsz, imgsz)
    traced = torch.jit.trace(model, dummy)
    traced.save(str(output_path))
    print(f"  Exported TorchScript: {output_path}")


def main():
    args = parse_args()
    import torch

    env = detect_environment()
    formats = [f.strip() for f in args.formats.split(',')]

    # Build model and load weights
    model, info = build_model(task=args.task, variant=args.variant)
    weights_path = Path(args.weights)

    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location='cpu')
        # Filter out thop profiler keys (total_ops/total_params) injected by GFLOPs calculation
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.endswith(('total_ops', 'total_params'))}
        model.load_state_dict(state_dict)
        print(f"  Loaded weights: {weights_path}")
    else:
        print(f"  [WARN] Weights not found, exporting untrained model")

    model.eval()
    output_dir = weights_path.parent / 'exports'
    output_dir.mkdir(exist_ok=True)

    name = weights_path.stem

    for fmt in formats:
        if fmt == 'onnx':
            export_onnx(model, args.imgsz, output_dir / f'{name}.onnx', args.fp16)
        elif fmt == 'torchscript':
            export_torchscript(model, args.imgsz, output_dir / f'{name}.torchscript')
        else:
            print(f"  [SKIP] Format '{fmt}' — use Ultralytics for TFLite/CoreML export")

    print(f"\n  Exports saved to: {output_dir}")


if __name__ == '__main__':
    main()
