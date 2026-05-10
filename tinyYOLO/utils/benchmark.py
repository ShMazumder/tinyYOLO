"""
Model Benchmarking Utilities
==============================
Parameter counting, FLOPs estimation, and latency measurement.
"""

import time
import torch
import torch.nn as nn


def count_parameters(model, trainable_only=True):
    """
    Count model parameters.

    Returns:
        dict with total, trainable, non_trainable counts and formatted strings.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    result = {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable,
        'total_M': round(total / 1e6, 2),
        'trainable_M': round(trainable / 1e6, 2),
    }
    return result


def estimate_flops(model, imgsz=320, device='cpu'):
    """
    Estimate model FLOPs using a dummy input.
    Requires thop library (pip install thop).

    Args:
        model: PyTorch model.
        imgsz: Input image size.
        device: Device to run on.

    Returns:
        dict with flops, flops_G (in GFLOPs), and params.
    """
    try:
        from thop import profile, clever_format
    except ImportError:
        # Fallback: rough estimation
        params = count_parameters(model)
        return {
            'flops': None,
            'flops_G': None,
            'params_M': params['total_M'],
            'note': 'Install thop for FLOPs: pip install thop',
        }

    model = model.to(device).eval()
    dummy = torch.randn(1, 3, imgsz, imgsz).to(device)

    with torch.no_grad():
        flops, params = profile(model, inputs=(dummy,), verbose=False)

    return {
        'flops': int(flops),
        'flops_G': round(flops / 1e9, 2),
        'params': int(params),
        'params_M': round(params / 1e6, 2),
    }


def measure_latency(model, imgsz=320, device='cpu', warmup=10, runs=100):
    """
    Measure inference latency.

    Args:
        model: PyTorch model.
        imgsz: Input image size.
        device: Device ('cpu', 'cuda:0', 'mps').
        warmup: Number of warmup iterations.
        runs: Number of timed iterations.

    Returns:
        dict with mean_ms, std_ms, fps.
    """
    model = model.to(device).eval()
    dummy = torch.randn(1, 3, imgsz, imgsz).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)

    # Sync CUDA
    if device.startswith('cuda'):
        torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            model(dummy)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    mean_ms = sum(times) / len(times)
    std_ms = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5

    return {
        'mean_ms': round(mean_ms, 2),
        'std_ms': round(std_ms, 2),
        'fps': round(1000 / mean_ms, 1),
        'device': device,
        'imgsz': imgsz,
    }


def full_benchmark(model, imgsz_list=None, device='cpu'):
    """
    Run full benchmark across multiple resolutions.

    Args:
        model: PyTorch model.
        imgsz_list: List of image sizes to benchmark.
        device: Device.

    Returns:
        List of benchmark dicts.
    """
    if imgsz_list is None:
        imgsz_list = [160, 224, 320, 416, 640]

    results = []
    for imgsz in imgsz_list:
        params = count_parameters(model)
        flops = estimate_flops(model, imgsz, device)
        latency = measure_latency(model, imgsz, device, warmup=5, runs=50)

        results.append({
            'imgsz': imgsz,
            'params_M': params['total_M'],
            'flops_G': flops.get('flops_G'),
            **latency,
        })

    return results


def print_benchmark_table(results):
    """Print benchmark results as a formatted table."""
    print(f"\n{'ImgSz':>6} | {'Params(M)':>10} | {'GFLOPs':>8} | {'ms':>8} | {'FPS':>8}")
    print("-" * 52)
    for r in results:
        flops = f"{r['flops_G']:.2f}" if r.get('flops_G') else "N/A"
        print(f"{r['imgsz']:>6} | {r['params_M']:>10.2f} | {flops:>8} | "
              f"{r['mean_ms']:>7.1f} | {r['fps']:>7.1f}")
