# %% [markdown]
# # Stage 1 — COCO128 smoke test (GATE)
#
# End-to-end pipeline check on the tiny 128-image COCO subset (auto-downloads).
# Confirms the full train->eval->artifact path works and that boxes localize
# after the R1.4 fix, before spending real GPU-hours on Stage 2.
#
# **Pass:** mAP@50 clearly > 0 and predictions are not sprayed (FP not in the
# hundreds-of-thousands). The broken-decode VOC run had 644k predictions / mAP 0.001.
#
# ## Gate caveat (R1.5)
# coco128 has 929 instances across 80 classes = ~11.6 per class. An 80-way
# classifier cannot be learned from that at any step count. A run can have a
# perfectly correct box+objectness path and still score mAP50 ~0.01 here purely
# because the class head is starved. Diagnose with the per-class table: if the
# box path is healthy, box loss sits near 0.6 (mean CIoU ~0.35 at positives)
# while >90% of predictions collapse onto one class id.
#
# Use `--nc1` to collapse all labels to a single "object" class. That isolates
# the box + objectness path from classification and is the *real* structural
# gate. Expect mAP50 > 0.4 on a 128-image train==val set; anything less means a
# genuine bug in assignment, loss, or decode.

# %%
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import run_train, load_metrics, print_table

# %%
# Step budget matters more than epoch count. 128 imgs / batch B = (128//B)
# steps per epoch:
#     batch 128 -> 1 step/ep    (the auto batch; ~100 steps total: hopeless)
#     batch 16  -> 8 steps/ep   (800 total at 100 ep: still far too few)
#     batch 8   -> 16 steps/ep  (9600 total at 600 ep: a fair test)
p = argparse.ArgumentParser(description="Stage 1 coco128 smoke test")
p.add_argument("--epochs", type=int, default=600)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--imgsz", type=int, default=320)
p.add_argument("--lr", type=float, default=None, help="override train.py lr (default 1e-3)")
p.add_argument("--variant", default="quantized")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--name", default=None, help="run name (auto-derived if unset)")
p.add_argument("--nc1", action="store_true",
               help="collapse all classes to one -> isolates box+obj path from cls")
p.add_argument("--dry-run", action="store_true", help="print the command only")
# parse_known_args so this stays importable from a notebook without arg juggling
args, _unknown = p.parse_known_args()

name = args.name or (
    f"s1_coco128_{'nc1' if args.nc1 else 'q'}_{args.imgsz}_b{args.batch}_e{args.epochs}"
)

extra = ["--batch", str(args.batch)]
if args.lr is not None:
    extra += ["--lr", str(args.lr)]
if args.nc1:
    extra += ["--single-cls"]

run_train(name, task="det", variant=args.variant, imgsz=args.imgsz,
          epochs=args.epochs, seed=args.seed, data="coco128.yaml",
          extra=extra, dry_run=args.dry_run)

# %%
m = load_metrics(name)
if m:
    print_table(
        [[f"{m['mAP50']:.4f}", f"{m['mAP50_95']:.4f}", f"{m['precision']:.4f}",
          f"{m['recall']:.4f}", m.get("n_predictions", "?"), m.get("fp", "?")]],
        ["mAP50", "mAP50-95", "P", "R", "n_pred", "FP"],
    )
    # nc1 isolates the structural path, so it gets the strict gate. The 80-class
    # run cannot clear 0.05 for data reasons, not code reasons -> advisory only.
    if args.nc1:
        gate = m["mAP50"] > 0.40
        print("  STAGE 1 (nc=1) PASS — box + objectness path is correct"
              if gate else
              "  STAGE 1 (nc=1) FAIL — real bug in assigner/loss/decode, fix before Stage 2")
    else:
        gate = m["mAP50"] > 0.05
        print("  STAGE 1 PASS" if gate else
              "  STAGE 1 INCONCLUSIVE — 80 classes over 929 instances is not learnable.\n"
              "  Re-run with --nc1 for the structural gate before blaming the code.")
else:
    print("  no metrics.json — run on a GPU box (drop --dry-run)")
