"""
Environment Auto-Detection
============================
Detects training environment (local, Colab, Kaggle, RunPod, etc.)
and returns optimal training settings.
"""

import os
import platform
import sys


def detect_environment():
    """
    Auto-detect training environment and return optimal settings.

    Returns:
        dict with keys: platform, gpu_available, gpu_name, gpu_count,
        gpu_memory_gb, recommended_batch_size, recommended_workers,
        recommended_device, recommended_imgsz, fp16_available.
    """
    env = {
        'platform': 'local',
        'os': platform.system(),
        'python': sys.version.split()[0],
        'gpu_available': False,
        'gpu_name': None,
        'gpu_count': 0,
        'gpu_memory_gb': 0,
        'recommended_batch_size': 8,
        'recommended_workers': 2,
        'recommended_device': 'cpu',
        'fp16_available': False,
        'data_dir': './datasets',
    }

    # --- Detect GPU ---
    try:
        import torch
        env['torch_version'] = torch.__version__

        if torch.cuda.is_available():
            env['gpu_available'] = True
            env['gpu_count'] = torch.cuda.device_count()
            env['gpu_name'] = torch.cuda.get_device_name(0)
            # Get GPU memory — multiple fallback methods
            mem = 0
            try:
                mem = torch.cuda.mem_get_info(0)[1]  # (free, total)
            except Exception:
                try:
                    props = torch.cuda.get_device_properties(0)
                    for attr in ('total_memory', 'total_mem', 'totalGlobalMem'):
                        if hasattr(props, attr):
                            mem = getattr(props, attr)
                            break
                except Exception:
                    pass
            env['gpu_memory_gb'] = round(mem / (1024 ** 3), 1) if mem else 0
            env['recommended_device'] = 'cuda:0'
            env['fp16_available'] = True
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            env['gpu_available'] = True
            env['gpu_name'] = 'Apple Silicon (MPS)'
            env['recommended_device'] = 'mps'
            env['fp16_available'] = True
    except ImportError:
        env['torch_version'] = 'not installed'

    # --- Detect Platform ---
    if os.environ.get('COLAB_GPU') or os.environ.get('COLAB_RELEASE_TAG'):
        env['platform'] = 'colab'
        env['data_dir'] = '/content/datasets'
        env['recommended_workers'] = 4  # Colab has 2 vCPUs but benefits from 4 workers
    elif os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        env['platform'] = 'kaggle'
        env['data_dir'] = '/kaggle/working/datasets'
        env['recommended_workers'] = 4
    elif 'runpod' in os.environ.get('HOSTNAME', '').lower() or os.environ.get('RUNPOD_POD_ID'):
        env['platform'] = 'runpod'
        env['data_dir'] = '/workspace/datasets'
        env['recommended_workers'] = 8
    elif os.path.exists('/workspace') and os.environ.get('VAST_CONTAINERLABEL'):
        env['platform'] = 'vast.ai'
        env['data_dir'] = '/workspace/datasets'
        env['recommended_workers'] = 8
    else:
        env['platform'] = 'local'
        env['recommended_workers'] = min(os.cpu_count() or 4, 8)

    # --- Batch Size Recommendation ---
    # TinyYOLO is only 0.21M params — GPU memory is rarely the bottleneck.
    # Larger batches improve GPU utilization by reducing data-loading stalls.
    mem = env['gpu_memory_gb']
    if mem >= 40:       # A100 80GB, A100 40GB
        env['recommended_batch_size'] = 256
    elif mem >= 20:     # A5000, 3090, 4090
        env['recommended_batch_size'] = 128
    elif mem >= 10:     # T4 (15GB), 3080, 2080Ti
        env['recommended_batch_size'] = 64
    elif mem >= 6:      # 3060, 2060
        env['recommended_batch_size'] = 32
    elif env['gpu_available']:
        env['recommended_batch_size'] = 16
    else:
        env['recommended_batch_size'] = 4  # CPU

    return env


def get_training_config(env=None, imgsz=320):
    """
    Generate training configuration based on detected environment.

    Args:
        env: Environment dict from detect_environment(). Auto-detects if None.
        imgsz: Target image size.

    Returns:
        dict with training parameters.
    """
    if env is None:
        env = detect_environment()

    # Scale batch size with resolution (quadratic relationship)
    scale = (320 / imgsz) ** 2
    batch = max(1, int(env['recommended_batch_size'] * scale))

    config = {
        'device': env['recommended_device'],
        'batch': batch,
        'workers': env['recommended_workers'],
        'imgsz': imgsz,
        'amp': env['fp16_available'],  # Automatic Mixed Precision
        'cache': env['platform'] in ('colab', 'kaggle'),  # Cache dataset in RAM on cloud
        'data_dir': env['data_dir'],
    }
    return config


def print_env_report(env=None):
    """Print a formatted environment report."""
    if env is None:
        env = detect_environment()

    print("=" * 60)
    print("  TinyYOLO Environment Report")
    print("=" * 60)
    print(f"  Platform:    {env['platform'].upper()}")
    print(f"  OS:          {env['os']}")
    print(f"  Python:      {env['python']}")
    print(f"  PyTorch:     {env.get('torch_version', 'N/A')}")
    print(f"  GPU:         {env['gpu_name'] or 'None (CPU only)'}")
    if env['gpu_available']:
        print(f"  GPU Memory:  {env['gpu_memory_gb']} GB")
        print(f"  GPU Count:   {env['gpu_count']}")
        print(f"  FP16:        {'Yes' if env['fp16_available'] else 'No'}")
    print(f"  Device:      {env['recommended_device']}")
    print(f"  Batch Size:  {env['recommended_batch_size']} (recommended)")
    print(f"  Workers:     {env['recommended_workers']}")
    print(f"  Data Dir:    {env['data_dir']}")
    print("=" * 60)
