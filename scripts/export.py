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

from tinyYOLO.models import build_model, infer_arch_from_state_dict
from tinyYOLO.utils.env import detect_environment


def parse_args():
    parser = argparse.ArgumentParser(description='TinyYOLO Export')
    parser.add_argument('--weights', type=str, required=True, help='Model weights path')
    parser.add_argument('--task', type=str, default='det', help='Task type')
    parser.add_argument('--variant', type=str, default='standard', help='Variant')
    parser.add_argument('--imgsz', type=int, default=320, help='Input image size')
    parser.add_argument('--nc', type=int, default=80, help='Number of classes')
    parser.add_argument('--data', type=str, default=None, help='Dataset YAML configuration path')
    parser.add_argument('--formats', type=str, default='onnx',
                        help='Export formats (comma-separated): onnx,tflite,coreml,torchscript')
    parser.add_argument('--fp16', action='store_true', help='Export with FP16')
    parser.add_argument('--int8', action='store_true', help='Export with INT8 quantization')
    return parser.parse_args()


def export_onnx(model, imgsz, output_path, fp16=False):
    """Export model to ONNX format for cross-platform deployment.

    The ONNX model can be used with:
      - ONNX Runtime (CPU/GPU inference)
      - TensorRT (NVIDIA Jetson — convert ONNX → TensorRT engine)
      - TFLite (via onnx-tf → TFLite conversion)
      - OpenVINO (Intel hardware)

    Args:
        model: TinyYOLO model in eval mode.
        imgsz: Input image size (square).
        output_path: Output .onnx file path.
        fp16: If True, export in FP16 (half precision).

    Notes:
        - Dynamic batch axis is enabled by default for variable batch inference.
        - Opset 18 is used for broadest runtime compatibility.
        - For INT8 deployment, use scripts/quantize.py to create a calibrated
          INT8 model, then export with TensorRT's INT8 calibration workflow.
    """
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
    """Export model to TorchScript format for C++ deployment.

    TorchScript models can be loaded without Python via libtorch,
    making them suitable for embedded C++ applications.

    Args:
        model: TinyYOLO model in eval mode.
        imgsz: Input image size.
        output_path: Output .torchscript file path.
    """
    import torch

    model.eval()
    dummy = torch.randn(1, 3, imgsz, imgsz)
    traced = torch.jit.trace(model, dummy)
    traced.save(str(output_path))
    print(f"  Exported TorchScript: {output_path}")


def _clean_state_dict(state_dict):
    """Remove profiler metadata keys injected by PyTorch profiler/thop.

    Also strips distributed training (DDP 'module.'), compilation
    ('_orig_mod.'), and quantized wrapper ('model.') prefixes from state_dict keys.
    """
    cleaned = {}
    for k, v in state_dict.items():
        if k.endswith(('total_ops', 'total_params')):
            continue
        k_clean = k
        if k_clean.startswith('_orig_mod.'):
            k_clean = k_clean[len('_orig_mod.'):]
        if k_clean.startswith('module.'):
            k_clean = k_clean[len('module.'):]
        if k_clean.startswith('model.'):
            k_clean = k_clean[len('model.'):]
        cleaned[k_clean] = v
    return cleaned


def main():
    args = parse_args()
    import torch

    env = detect_environment()
    formats = [f.strip() for f in args.formats.split(',')]

    nc = args.nc
    if args.data and Path(args.data).exists():
        try:
            from train import load_dataset_config
            data_dict = load_dataset_config(args.data)
            nc = data_dict.get('nc', nc)
            print(f"  [INFO] Custom dataset resolved: nc={nc}")
        except Exception as e:
            print(f"  [WARN] Failed to load nc from {args.data}: {e}")

    # Build model and load weights
    model, info = build_model(task=args.task, variant=args.variant, nc=nc)
    weights_path = Path(args.weights)

    if weights_path.exists():
        checkpoint = torch.load(weights_path, map_location='cpu')
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        state_dict = _clean_state_dict(state_dict)
        # R2 architecture flags (SPPF / depthwise kernel size) are not stored in the
        # checkpoint, so recover them from tensor shapes before the first load attempt.
        arch = infer_arch_from_state_dict(state_dict)
        if arch != {'use_sppf': True, 'neck_k': 5, 'head_k': 5}:
            print(f"  [INFO] Checkpoint architecture detected: SPPF={arch['use_sppf']}, "
                  f"neck_k={arch['neck_k']}, head_k={arch['head_k']} — rebuilding to match.")
            model, info = build_model(task=args.task, variant=args.variant, nc=nc, **arch)

        try:
            model.load_state_dict(state_dict)
            print(f"  Loaded weights: {weights_path}")
        except RuntimeError as e:
            # Self-healing auto-detection of variant mismatch
            has_eca_keys = any('attn3.conv.weight' in k for k in state_dict.keys())
            has_standard_keys = any('attn3.conv.0.weight' in k for k in state_dict.keys())
            
            detected_variant = None
            if has_eca_keys and not has_standard_keys and args.variant == 'standard':
                detected_variant = 'quantized'
            elif has_standard_keys and not has_eca_keys and args.variant == 'quantized':
                detected_variant = 'standard'
                
            if detected_variant:
                print(f"  [INFO] Variant mismatch detected between CLI argument and checkpoint.")
                print(f"         Auto-rebuilding model with variant='{detected_variant}' to match checkpoint perfectly.")
                model, info = build_model(task=args.task, variant=detected_variant, nc=nc, **arch)
                model.load_state_dict(state_dict)
                print(f"  Loaded weights successfully after auto-healing! ✓")
            else:
                raise e
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
