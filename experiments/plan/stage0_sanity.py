# %% [markdown]
# # Stage 0 — Sanity (HARD GATE)
#
# Fast checks that the R1.4 box-decode fix actually made localization learnable.
# Runs on CPU in seconds. **If S0.1 does not overfit, stop — do not spend GPU
# hours.** No dataset required; uses synthetic data.
#
# Tests:
# - S0.3 decode round-trip (encode a box -> decode_boxes -> recover it)
# - S0.1 overfit one batch (loss must drop toward 0 on a fixed synthetic batch)
# - S0.2 gradient flow (all params get finite, non-zero grads)

# %%
import sys, importlib.util
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tinyYOLO.models import build_model
from tinyYOLO.utils.boxcodec import decode_boxes

# Load DetectionLoss from scripts/train.py without triggering its __main__.
_spec = importlib.util.spec_from_file_location("tyt_train", REPO_ROOT / "scripts" / "train.py")
tyt_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tyt_train)
DetectionLoss = tyt_train.DetectionLoss

torch.manual_seed(0)

# %% [markdown]
# ## S0.3 — Decode round-trip
# Build raw logits that should decode to a known box in cell (gi, gj), then
# confirm `decode_boxes` recovers it. This directly tests the codec math.

# %%
def encode_box(cx, cy, w, h, gi, gj, W, H):
    """Inverse of decode_boxes for a target box at cell (gi,gj)."""
    sx = (cx * W - gi + 0.5) / 2.0
    sy = (cy * H - gj + 0.5) / 2.0
    sx = min(max(sx, 1e-4), 1 - 1e-4)
    sy = min(max(sy, 1e-4), 1 - 1e-4)
    tx = torch.logit(torch.tensor(sx))
    ty = torch.logit(torch.tensor(sy))
    tw = torch.log(torch.tensor(w * W))
    th = torch.log(torch.tensor(h * H))
    return torch.stack([tx, ty, tw, th])

W = H = 10
targets = [(0.23, 0.61, 0.15, 0.30), (0.80, 0.20, 0.10, 0.10)]
ok = True
for (cx, cy, w, h) in targets:
    gi, gj = int(cx * W), int(cy * H)
    raw = encode_box(cx, cy, w, h, gi, gj, W, H).unsqueeze(0)          # [1,4]
    dec = decode_boxes(raw, torch.tensor([gi]), torch.tensor([gj]), W, H)[0]
    err = (dec - torch.tensor([cx, cy, w, h])).abs().max().item()
    print(f"  target={cx,cy,w,h}  decoded={tuple(round(v,3) for v in dec.tolist())}  max_err={err:.4f}")
    ok &= err < 1e-3
print("  S0.3 PASS" if ok else "  S0.3 FAIL")

# %% [markdown]
# ## S0.1 / S0.2 — Overfit one synthetic batch + gradient flow
# One fixed batch, a handful of GT boxes, many steps. Loss must fall sharply.
# With the old `sigmoid*imgsz` decode this stays flat; with the fix it drops.

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
model, info = build_model(task="det", variant="quantized")
model = model.to(device).train()
print(f"  model: {info.get('total_params_M','?')}M params on {device}")

B, imgsz, nc = 4, 320, 80
x = torch.randn(B, 3, imgsz, imgsz, device=device)

# targets: [B, max_obj, 5] = (cls, cx, cy, w, h) normalized; pad with zeros
max_obj = 3
tgt = torch.zeros(B, max_obj, 5, device=device)
for b in range(B):
    for o in range(max_obj):
        tgt[b, o] = torch.tensor([float((b + o) % nc), 0.3 + 0.1 * o, 0.4 + 0.1 * b,
                                  0.15, 0.20], device=device)

loss_fn = DetectionLoss(nc=nc).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

first = last = None
for step in range(300):
    opt.zero_grad()
    out = model(x)
    loss, parts = loss_fn(out, tgt)
    loss.backward()
    if step == 0:
        first = parts["total"]
        gnorms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        dead = sum(1 for g in gnorms if g == 0.0)
        nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
        print(f"  S0.2 grads: {len(gnorms)} tensors, {dead} zero, nan={nan}")
    opt.step()
    if step % 50 == 0:
        print(f"  step {step:3d}  total={parts['total']:.4f}  box={parts['box']:.4f}")
    last = parts["total"]

drop = (first - last) / max(first, 1e-9)
print(f"  S0.1 loss {first:.3f} -> {last:.3f}  ({drop*100:.0f}% drop)")
print("  S0.1 PASS" if drop > 0.5 else "  S0.1 FAIL (localization path still broken)")
