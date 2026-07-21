# %% [markdown]
# # Confidence diagnostic — why 0 predictions?
#
# Loads the trained checkpoint and reports the actual objectness / class / joint
# confidence magnitudes on real val images, plus how many cells clear each
# threshold. Runs in seconds, no retrain. Decides the next fix:
#   - joint max >> 1e-3 but eval gave 0  -> inference/threshold bug
#   - joint max <  1e-3 everywhere       -> confidences genuinely not learned
#   - obj max ~0.5 but cls max ~0.01     -> classifier is the weak link (or vice-versa)

# %%
import sys, glob
from pathlib import Path
import numpy as np, cv2, torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from tinyYOLO.models import build_model
from tinyYOLO.utils.postprocess import decode_predictions

CKPT = ROOT / "experiments/results/s1_coco128_q_320/best.pt"
IMG_DIR = ROOT / "datasets/coco128/images/train2017"
IMGSZ = 320

# %%
model, _ = build_model(task="det", variant="quantized", nc=80)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
if isinstance(sd, dict):
    for k in ("model", "model_state_dict", "state_dict", "ema"):
        if k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
missing = model.load_state_dict(sd, strict=False)
model.eval()
print(f"  loaded {CKPT.name}; missing={len(missing.missing_keys)} unexpected={len(missing.unexpected_keys)}")

# %%
paths = sorted(glob.glob(str(IMG_DIR / "*.jpg")))[:8]
xs = []
for p in paths:
    im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (IMGSZ, IMGSZ))
    xs.append(torch.from_numpy(im).permute(2, 0, 1).float() / 255.0)
x = torch.stack(xs)
print(f"  {len(paths)} images -> input {tuple(x.shape)}")

with torch.no_grad():
    outs = model(x)

# %% [markdown]
# ## Per-scale confidence stats

# %%
for si, o in enumerate(outs):
    obj = torch.sigmoid(o[:, 4:5])                 # [B,1,H,W]
    cls = torch.sigmoid(o[:, 5:])                  # [B,nc,H,W]
    joint = obj.squeeze(1) * cls.max(dim=1).values # [B,H,W]
    H, W = o.shape[-2:]
    print(f"  scale{si} {H}x{W}: "
          f"obj[max {obj.max():.3f} mean {obj.mean():.4f}] "
          f"cls[max {cls.max():.3f} mean {cls.mean():.4f}] "
          f"joint[max {joint.max():.4f}] "
          f"cells>1e-3 {int((joint>1e-3).sum())}  >1e-2 {int((joint>1e-2).sum())}  >5e-2 {int((joint>5e-2).sum())}")

# %% [markdown]
# ## Raw detections at conf=0 (are the boxes even plausible?)

# %%
dets = decode_predictions(outs, IMGSZ, conf_thresh=0.0, nc=80)
for b, d in enumerate(dets):
    if len(d):
        top = d[d[:, 4].argsort(descending=True)][:3]
        print(f"  img{b}: {len(d)} raw dets | top scores {[round(float(s),4) for s in top[:,4]]} "
              f"| top box(px) {[round(float(v),1) for v in top[0,:4]]}")
    else:
        print(f"  img{b}: 0 raw dets even at conf=0")

print("\n  READ: if joint max < 1e-3 everywhere -> confidences not learned (train longer / raise LR /"
      " obj pos_weight). If joint max > 1e-2 -> eval threshold/path bug, not training.")
