# %% [markdown]
# # Stage 7 — Multi-task validation
#
# Train each non-detection head and report its task metric. **Only claim a task
# in the paper once it has an artifact here.** Current manuscript is inconsistent:
# README says all 5 tasks "validated"; report §12 says Cls/OBB pending. This
# resolves it with real runs.
#
# Note: seg/pose/obb losses in `MultiTaskLoss` are partly placeholders — verify
# each task's loss is implemented before trusting its metric (see scripts/train.py
# `MultiTaskLoss`). Detection is the validated path; others may need loss work.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import run_train, load_metrics, print_table

DRY_RUN = False

TASKS = [
    # (task, variant, imgsz, epochs, data, note)
    ("seg",  "quantized", 320, 200, None, "COCO instance masks"),
    ("pose", "quantized", 320, 200, None, "COCO person keypoints"),
    ("cls",  "quantized", 224, 100, None, "ImageNet-1k top-1"),
    ("obb",  "quantized", 416, 200, None, "DOTA oriented boxes"),
]

# %%
for task, variant, imgsz, epochs, data, note in TASKS:
    name = f"mt_{task}_{variant}_{imgsz}"
    print(f"\n  --- {task} ({note}) ---")
    run_train(name, task=task, variant=variant, imgsz=imgsz, epochs=epochs,
              seed=42, data=data, dry_run=DRY_RUN)

# %%
rows = []
for task, variant, imgsz, epochs, data, note in TASKS:
    m = load_metrics(f"mt_{task}_{variant}_{imgsz}")
    if m:
        rows.append([task, imgsz, f"{m.get('mAP50', float('nan')):.4f}", note])
    else:
        rows.append([task, imgsz, "no artifact", note])
print_table(rows, ["task", "imgsz", "primary metric", "note"])
print("\n  Reconcile README (§ 'all 5 validated') and report §12 with THIS table.")
