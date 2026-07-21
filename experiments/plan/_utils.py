"""
Shared helpers for the TinyYOLO experiment plan notebooks.

Every experiment drives the *real, tested* training path (`scripts/train.py`)
via subprocess, then reads back the artifacts each run writes to
`experiments/results/<name>/`. Nothing here reimplements training or metrics —
that keeps results trustworthy and avoids a second source of bugs.

Usage from a notebook cell:
    from _utils import run_train, load_metrics, summarize, print_table, REPO_ROOT
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from statistics import mean, pstdev

# --- Repo layout -----------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent            # experiments/plan -> repo root
TRAIN = REPO_ROOT / "scripts" / "train.py"
QUANTIZE = REPO_ROOT / "scripts" / "quantize.py"
RESULTS = REPO_ROOT / "experiments" / "results"


def run_train(name, task="det", variant="quantized", imgsz=416, epochs=100,
              seed=42, data=None, extra=None, dry_run=False, timeout=None):
    """Run scripts/train.py for one experiment. Returns the results dir.

    Args:
        name: experiment name -> experiments/results/<name>/
        task, variant, imgsz, epochs, seed: forwarded flags.
        data: dataset yaml (e.g. 'voc.yaml'); None lets train.py auto-pick.
        extra: list[str] of additional flags (e.g. ['--attention','eca']).
        dry_run: if True, only print the command and return None.
        timeout: seconds before the subprocess is killed.
    """
    cmd = [
        sys.executable, str(TRAIN),
        "--task", str(task),
        "--variant", str(variant),
        "--imgsz", str(imgsz),
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--name", str(name),
    ]
    if data:
        cmd += ["--data", str(data)]
    if extra:
        cmd += list(extra)

    print("  $", " ".join(cmd))
    if dry_run:
        return None

    t0 = time.time()
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, timeout=timeout)
    print(f"  done in {time.time()-t0:.0f}s")
    return RESULTS / name


def load_metrics(name):
    """Read experiments/results/<name>/metrics.json (or None if missing)."""
    p = RESULTS / name / "metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def summarize(names, keys=("mAP50", "mAP50_95", "precision", "recall", "f1")):
    """Aggregate metrics across a list of run names -> {key: (mean, std)}."""
    out = {}
    for k in keys:
        vals = []
        for n in names:
            m = load_metrics(n)
            if m and k in m and m[k] is not None:
                vals.append(float(m[k]))
        if vals:
            out[k] = (mean(vals), pstdev(vals) if len(vals) > 1 else 0.0)
    return out, len([n for n in names if load_metrics(n)])


def print_table(rows, headers):
    """Minimal fixed-width table printer (no deps)."""
    cols = list(zip(*([headers] + rows))) if rows else [[h] for h in headers]
    widths = [max(len(str(c)) for c in col) for col in cols]
    line = "  " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("  " + "-+-".join("-" * w for w in widths))
    for r in rows:
        print("  " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)))


# Canonical seed set for statistical runs (matches the manuscript).
SEEDS_5 = [42, 123, 256, 512, 1024]
SEEDS_3 = [42, 123, 256]
