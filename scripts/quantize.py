"""
TinyYOLO Quantization Pipeline
================================
Quantization-Aware Training (QAT) and Post-Training Quantization (PTQ)
for deploying TinyYOLO on INT8 edge accelerators.

QAT inserts fake quantization nodes during training so the model learns
to compensate for quantization noise. PTQ calibrates a pre-trained model
on a representative dataset without retraining.

Usage:
    # QAT from a pre-trained checkpoint
    python scripts/quantize.py --mode qat --weights best.pt --data voc.yaml --epochs 20

    # PTQ with calibration
    python scripts/quantize.py --mode ptq --weights best.pt --data voc.yaml --n-calib 500

    # Export INT8 ONNX
    python scripts/quantize.py --mode ptq --weights best.pt --export onnx
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.utils.data import DataLoader

from tinyYOLO.models import build_model
from tinyYOLO.utils.env import detect_environment


class DequantizedHeadWrapper(nn.Module):
    """Wrapper that dequantizes inputs before feeding them to the task-specific head, preserving FP32 precision."""
    def __init__(self, head):
        super().__init__()
        import torch.ao.quantization as ao_quant
        self.dequant = ao_quant.DeQuantStub()
        self.head = head

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            # Handle multiple scales from LitePAN neck
            return self.head([self.dequant(t) for t in x])
        return self.head(self.dequant(x))


class QuantizedWrapper(nn.Module):
    """Wrapper that inserts QuantStub at the input to convert images from Float32 to quantized INT8."""
    def __init__(self, model):
        super().__init__()
        import torch.ao.quantization as ao_quant
        self.quant = ao_quant.QuantStub()
        
        # Wrap the head to dequantize neck features back to Float32 before predictions
        if hasattr(model, 'head') and not isinstance(model.head, DequantizedHeadWrapper):
            model.head = DequantizedHeadWrapper(model.head)
            
        self.model = model

    def forward(self, x):
        x = self.quant(x)
        return self.model(x)


def parse_args():
    parser = argparse.ArgumentParser(description='TinyYOLO Quantization')
    parser.add_argument('--mode', type=str, default='ptq',
                        choices=['qat', 'ptq'], help='Quantization mode')
    parser.add_argument('--weights', type=str, default=None,
                        help='Pre-trained weights path')
    parser.add_argument('--task', type=str, default='det', help='Task type')
    parser.add_argument('--variant', type=str, default='quantized',
                        help='Architecture variant (should be quantized for INT8)')
    parser.add_argument('--data', type=str, default='voc.yaml',
                        help='Dataset YAML for calibration/fine-tuning')
    parser.add_argument('--imgsz', type=int, default=416, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Fine-tuning epochs for QAT')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for QAT (lower than full training)')
    parser.add_argument('--n-calib', type=int, default=500,
                        help='Number of calibration batches for PTQ')
    parser.add_argument('--backend', type=str, default='qnnpack',
                        choices=['fbgemm', 'qnnpack'],
                        help='Quantization backend (qnnpack for ARM, fbgemm for x86)')
    parser.add_argument('--export', type=str, default=None,
                        choices=['onnx', 'torchscript', None],
                        help='Export format after quantization')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: alongside weights)')
    return parser.parse_args()


def _load_model_and_weights(args):
    """Build model and load pre-trained weights."""
    nc = 80
    if args.data and Path(args.data).exists():
        try:
            from train import load_dataset_config
            data_dict = load_dataset_config(args.data)
            nc = data_dict.get('nc', 80)
            print(f"  [INFO] Custom dataset resolved: nc={nc}")
        except Exception as e:
            print(f"  [WARN] Failed to load nc from {args.data}: {e}")

    model, info = build_model(task=args.task, variant=args.variant, nc=nc)

    if args.weights and Path(args.weights).exists():
        checkpoint = torch.load(args.weights, map_location='cpu')
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        
        # Strip '_orig_mod.' (compile), 'module.' (DDP), and 'model.' (quant wrapper) prefixes precisely
        # Filter out thop profiler keys
        cleaned_state_dict = {}
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
            cleaned_state_dict[k_clean] = v
            
        model.load_state_dict(cleaned_state_dict)
        print(f"  Loaded weights: {args.weights}")
    else:
        print(f"  [WARN] No weights loaded — quantizing untrained model")

    return model, info


def _get_calibration_loader(args):
    """Create a data loader for calibration/fine-tuning."""
    from train import load_dataset_config, SimpleDetectionDataset

    data_dict = load_dataset_config(args.data)
    train_dir = data_dict.get('train', '')

    dataset = SimpleDetectionDataset(train_dir, imgsz=args.imgsz, augment=False)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True,
        num_workers=2, pin_memory=False, drop_last=False,
    )
    return loader


def apply_ptq(model, calibration_loader, n_batches=500, backend='qnnpack'):
    """Post-Training Quantization with MinMax observer calibration.

    PTQ calibrates a pre-trained model by collecting activation statistics
    over a representative dataset, then maps weights and activations to INT8.

    Args:
        model: Pre-trained TinyYOLO model (should use quantized variant with ReLU6).
        calibration_loader: DataLoader with representative data.
        n_batches: Number of batches for calibration statistics.
        backend: Quantization backend ('qnnpack' for ARM, 'fbgemm' for x86).

    Returns:
        model_int8: INT8 quantized model.
        calibration_stats: Dict with calibration metrics.
    """
    torch.backends.quantized.engine = backend

    wrapped_model = QuantizedWrapper(model)

    # Set quantization configuration
    # Per-channel symmetric for weights, per-tensor asymmetric for activations
    if backend == 'qnnpack':
        wrapped_model.qconfig = quant.QConfig(
            activation=quant.observer.MinMaxObserver.with_args(
                dtype=torch.quint8, qscheme=torch.per_tensor_affine
            ),
            weight=quant.observer.MinMaxObserver.with_args(
                dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
            ),
        )
    else:
        wrapped_model.qconfig = quant.get_default_qconfig(backend)

    # Exclude task-specific regression/class heads from INT8 quantization to preserve bbox/class accuracy,
    # but keep DequantizedHeadWrapper active so that its DeQuantStub gets converted.
    if hasattr(model, 'head'):
        if isinstance(model.head, DequantizedHeadWrapper):
            model.head.head.qconfig = None
        else:
            model.head.qconfig = None

    # Prepare model for calibration (inserts observer modules)
    wrapped_model.eval()
    quant.prepare(wrapped_model, inplace=True)

    # Run calibration forward passes
    print(f"\n  Running PTQ calibration ({n_batches} batches)...")
    n_images = 0
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= n_batches:
                break
            wrapped_model(images)
            n_images += images.shape[0]
            if (i + 1) % 100 == 0:
                print(f"    Calibrated {i + 1}/{n_batches} batches ({n_images} images)")

    print(f"  Calibration complete: {n_images} images processed")

    # Convert to quantized model
    model_int8 = quant.convert(wrapped_model.eval(), inplace=False)

    # Compute model size
    fp32_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    int8_params = 0
    for name, param in model_int8.named_parameters():
        int8_params += param.numel() * param.element_size()
    for name, buf in model_int8.named_buffers():
        int8_params += buf.numel() * buf.element_size()
    int8_size = int8_params / 1e6

    stats = {
        'method': 'PTQ',
        'backend': backend,
        'calibration_images': n_images,
        'calibration_batches': min(i + 1, n_batches),
        'fp32_size_mb': round(fp32_size, 3),
        'int8_size_mb': round(int8_size, 3),
        'compression_ratio': round(fp32_size / max(int8_size, 1e-6), 2),
    }

    return model_int8, stats


def apply_qat(model, train_loader, epochs=10, lr=1e-4, backend='qnnpack'):
    """Quantization-Aware Training with fake quantization nodes.

    QAT simulates INT8 quantization during training by inserting fake
    quantization nodes (round + clamp) in the forward pass. The model
    learns to produce weights and activations robust to quantization noise,
    typically achieving higher INT8 accuracy than PTQ.

    Args:
        model: Pre-trained TinyYOLO model.
        train_loader: Training data loader.
        epochs: Number of QAT fine-tuning epochs.
        lr: Learning rate (should be lower than full training, e.g., 1e-4).
        backend: Quantization backend.

    Returns:
        model_int8: INT8 quantized model.
        training_stats: Dict with QAT training metrics.
    """
    from train import DetectionLoss

    torch.backends.quantized.engine = backend

    wrapped_model = QuantizedWrapper(model)
    wrapped_model.train()
    wrapped_model.qconfig = quant.get_default_qat_qconfig(backend)

    # Exclude task-specific regression/class heads from INT8 quantization to preserve bbox/class accuracy,
    # but keep DequantizedHeadWrapper active so that its DeQuantStub gets converted.
    if hasattr(model, 'head'):
        if isinstance(model.head, DequantizedHeadWrapper):
            model.head.head.qconfig = None
        else:
            model.head.qconfig = None

    quant.prepare_qat(wrapped_model, inplace=True)

    optimizer = torch.optim.AdamW(wrapped_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Infer nc from wrapped_model (look for cls_preds output channels)
    nc = 80
    for name, module in wrapped_model.named_modules():
        if hasattr(module, 'nc'):
            nc = module.nc
            break
    loss_fn = DetectionLoss(nc=nc)

    print(f"\n  Starting QAT fine-tuning ({epochs} epochs, lr={lr})...")
    history = []

    for epoch in range(epochs):
        wrapped_model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, targets in train_loader:
            optimizer.zero_grad()
            outputs = wrapped_model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss, loss_dict = loss_fn(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss_dict['total']
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        lr_now = scheduler.get_last_lr()[0]
        history.append({'epoch': epoch + 1, 'loss': round(avg_loss, 4), 'lr': round(lr_now, 8)})
        print(f"    Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, lr={lr_now:.6f}")

        # Freeze observer and batch norm stats in last 25% of QAT
        if epoch >= int(epochs * 0.75):
            wrapped_model.apply(quant.disable_observer)
            wrapped_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    # Convert to INT8
    model_int8 = quant.convert(wrapped_model.eval(), inplace=False)

    stats = {
        'method': 'QAT',
        'backend': backend,
        'epochs': epochs,
        'final_loss': history[-1]['loss'] if history else 0,
        'history': history,
    }

    return model_int8, stats


def export_quantized(model, imgsz, output_path, fmt='onnx'):
    """Export quantized model to deployment format.

    Args:
        model: Quantized model (INT8).
        imgsz: Input image size.
        output_path: Output file path.
        fmt: Export format ('onnx' or 'torchscript').
    """
    if fmt == 'torchscript':
        model.eval()
        dummy = torch.randn(1, 3, imgsz, imgsz)
        traced = torch.jit.trace(model, dummy)
        traced.save(str(output_path))
        size_mb = output_path.stat().st_size / 1e6
        print(f"  Exported TorchScript: {output_path} ({size_mb:.2f} MB)")
    elif fmt == 'onnx':
        try:
            model.eval()
            dummy = torch.randn(1, 3, imgsz, imgsz)
            try:
                # Force legacy tracing-based exporter explicitly (dynamo=False)
                # to bypass packed C++ parameter trace exceptions in modern PyTorch versions
                torch.onnx.export(
                    model, dummy, str(output_path),
                    opset_version=18,
                    input_names=['images'],
                    output_names=['output'],
                    dynamic_axes={'images': {0: 'batch'}, 'output': {0: 'batch'}},
                    dynamo=False
                )
            except TypeError:
                # Backward compatibility fallback for older PyTorch versions
                torch.onnx.export(
                    model, dummy, str(output_path),
                    opset_version=18,
                    input_names=['images'],
                    output_names=['output'],
                    dynamic_axes={'images': {0: 'batch'}, 'output': {0: 'batch'}},
                )
            size_mb = output_path.stat().st_size / 1e6
            print(f"  Exported ONNX: {output_path} ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"\n  [WARN] Native PyTorch ONNX export is not supported for eager-mode quantized CPU models.")
            print(f"         Reason: {e}")
            print(f"         (Note: PyTorch eager quantization uses proprietary C++ packed parameters that cannot be directly mapped to ONNX).")
            print(f"         Your calibrated INT8 PyTorch checkpoint (.pt) was successfully created and saved!")
            print(f"\n  [INFO] Recommended Production Deployment Path:")
            print(f"         To deploy a high-performance quantized INT8 model on ONNX Runtime, use the standard ONNX quantization workflow:")
            print(f"         1. Export the Float32 model to ONNX:")
            print(f"            python scripts/export.py --weights experiments/results/crime-detection-yolo-run/best.pt --imgsz {imgsz}")
            print(f"         2. Quantize the ONNX model to INT8 using ONNX Runtime:")
            print(f"            pip install onnxruntime")
            print(f"            python -c \"import onnxruntime.quantization as q; q.quantize_dynamic('experiments/results/crime-detection-yolo-run/exports/best.onnx', '{output_path}')\"")
    else:
        print(f"  [SKIP] Unknown format: {fmt}")


def main():
    args = parse_args()
    import json

    print(f"\n{'='*60}")
    print(f"  TinyYOLO Quantization Pipeline")
    print(f"  Mode:    {args.mode.upper()}")
    print(f"  Variant: {args.variant}")
    print(f"  Backend: {args.backend}")
    print(f"{'='*60}")

    if args.variant != 'quantized':
        print(f"\n  [WARN] Using '{args.variant}' variant. The 'quantized' variant")
        print(f"         (ReLU6 + ECA) is specifically designed for INT8 and will")
        print(f"         produce better accuracy retention under quantization.")

    # Load model
    model, info = _load_model_and_weights(args)
    print(f"  Model:   {info['total_params_M']}M params")

    # Get calibration/training data
    cal_loader = _get_calibration_loader(args)
    print(f"  Data:    {len(cal_loader.dataset)} images, {len(cal_loader)} batches")

    # Run quantization
    if args.mode == 'ptq':
        model_int8, stats = apply_ptq(
            model, cal_loader,
            n_batches=args.n_calib,
            backend=args.backend,
        )
    elif args.mode == 'qat':
        model_int8, stats = apply_qat(
            model, cal_loader,
            epochs=args.epochs,
            lr=args.lr,
            backend=args.backend,
        )

    # Report
    print(f"\n  Quantization Results:")
    for k, v in stats.items():
        if k != 'history':
            print(f"    {k}: {v}")

    # Save
    output_dir = Path(args.output) if args.output else Path(args.weights).parent / 'quantized'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save INT8 model
    int8_path = output_dir / f'model_int8_{args.mode}.pt'
    torch.save(model_int8.state_dict(), int8_path)
    print(f"\n  Saved INT8 model: {int8_path}")

    # Save stats
    stats_path = output_dir / f'quantization_stats_{args.mode}.json'
    # Filter non-serializable items
    save_stats = {k: v for k, v in stats.items()}
    with open(stats_path, 'w') as f:
        json.dump(save_stats, f, indent=2)
    print(f"  Saved stats: {stats_path}")

    # Export if requested
    if args.export:
        export_path = output_dir / f'model_int8.{args.export}'
        export_quantized(model_int8, args.imgsz, export_path, args.export)

    print(f"\n  All outputs saved to: {output_dir}")
    print(f"  Quantization complete! ✓")


if __name__ == '__main__':
    main()
