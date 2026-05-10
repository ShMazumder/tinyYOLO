"""
TinyYOLO Benchmark Script
============================
Benchmark all model variants across all resolutions.

Usage:
    python scripts/benchmark_models.py
    python scripts/benchmark_models.py --tasks det seg --variants standard --imgsz 320,640
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tinyYOLO.models import build_model
from tinyYOLO.utils.benchmark import (
    count_parameters, estimate_flops, measure_latency, print_benchmark_table,
)
from tinyYOLO.utils.env import detect_environment, print_env_report


def parse_args():
    parser = argparse.ArgumentParser(description='TinyYOLO Benchmarking')
    parser.add_argument('--tasks', type=str, default='det,seg,pose,cls,obb',
                        help='Tasks to benchmark (comma-separated)')
    parser.add_argument('--variants', type=str, default='standard,quantized',
                        help='Variants to benchmark')
    parser.add_argument('--imgsz', type=str, default='160,224,320,416,640',
                        help='Image sizes to benchmark')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for latency measurement')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    return parser.parse_args()


def main():
    args = parse_args()

    env = detect_environment()
    print_env_report(env)

    tasks = [t.strip() for t in args.tasks.split(',')]
    variants = [v.strip() for v in args.variants.split(',')]
    imgsz_list = [int(s.strip()) for s in args.imgsz.split(',')]

    all_results = []

    for task in tasks:
        for variant in variants:
            print(f"\n{'='*60}")
            print(f"  Model: tinyYOLO-{task} ({variant})")
            print(f"{'='*60}")

            try:
                model, info = build_model(task=task, variant=variant)
                params = count_parameters(model)
                print(f"  Parameters: {params['total_M']}M")

                results = []
                for imgsz in imgsz_list:
                    try:
                        flops = estimate_flops(model, imgsz, 'cpu')
                        latency = measure_latency(model, imgsz, args.device, warmup=5, runs=30)

                        result = {
                            'task': task,
                            'variant': variant,
                            'imgsz': imgsz,
                            'params_M': params['total_M'],
                            'flops_G': flops.get('flops_G'),
                            **latency,
                        }
                        results.append(result)
                        all_results.append(result)
                    except Exception as e:
                        print(f"  [ERROR] {imgsz}: {e}")

                if results:
                    print_benchmark_table(results)

            except Exception as e:
                print(f"  [ERROR] Failed to build model: {e}")

    # Save results
    output_path = args.output or str(
        PROJECT_ROOT / 'experiments' / 'results' / f'benchmark_{datetime.now():%Y%m%d_%H%M%S}.json'
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == '__main__':
    main()
