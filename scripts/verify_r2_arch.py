#!/usr/bin/env python3
"""
R2 architecture verification
============================
Checks the Step-1 / Step-1b changes (SPPF + depthwise k=5) before any GPU time
is spent on them:

  1. every task/variant builds and does a forward pass with correct output shapes
  2. SPPF actually changes P5 (it is not silently an Identity)
  3. parameter cost of each change is what the design doc claims
  4. a pre-R2 checkpoint still loads, via infer_arch_from_state_dict()
  5. k=5 weights really are 5x5 in both neck and head

Run:  python scripts/verify_r2_arch.py
Exit code 0 = all pass.
"""

import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyYOLO.models import build_model, infer_arch_from_state_dict  # noqa: E402
from tinyYOLO.utils.benchmark import count_parameters  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)
    return cond


def section(t):
    print(f"\n{'=' * 68}\n  {t}\n{'=' * 68}")


# --------------------------------------------------------------------------
section("1. Build + forward, all tasks x variants")

EXPECTED_EXTRA = {'det': 0, 'seg': 32, 'obb': 1}  # extra channels appended per scale
BOX_CH = 4  # R2 head: 4 box + nc cls, no objectness channel

for task in ('det', 'seg', 'pose', 'cls', 'obb'):
    for variant in ('standard', 'quantized'):
        try:
            nc = {'cls': 10, 'pose': 1}.get(task, 20)
            model, info = build_model(task=task, variant=variant, nc=nc)
            model.eval()
            with torch.no_grad():
                out = model(torch.randn(2, 3, 320, 320))

            if task == 'cls':
                ok = tuple(out.shape) == (2, nc)
                check(f"{task}/{variant} forward", ok, f"out {tuple(out.shape)}")
            else:
                dets = out[0] if isinstance(out, tuple) else out
                exp_c = nc + BOX_CH + EXPECTED_EXTRA.get(task, 0)
                shapes = [tuple(d.shape) for d in dets]
                ok = (len(dets) == 3
                      and all(d.shape[0] == 2 for d in dets)
                      and all(d.shape[1] == exp_c for d in dets)
                      and [d.shape[2] for d in dets] == [40, 20, 10])
                check(f"{task}/{variant} forward", ok, f"{shapes}")
        except Exception:
            check(f"{task}/{variant} forward", False, traceback.format_exc(limit=1).strip())

# --------------------------------------------------------------------------
section("2. SPPF is live, not a no-op")

m_on, _ = build_model(task='det', variant='quantized', nc=20, use_sppf=True)
m_off, _ = build_model(task='det', variant='quantized', nc=20, use_sppf=False)

check("sppf module present when use_sppf=True",
      any('sppf.cv1' in n for n, _ in m_on.named_parameters()))
check("sppf module absent when use_sppf=False",
      not any('sppf' in n for n, _ in m_off.named_parameters()))

# SPPF must actually alter the P5 tensor
bb = m_on.backbone
bb.eval()
with torch.no_grad():
    x = torch.randn(1, 3, 320, 320)
    y = bb.stem(x); y = bb.stage1(y)
    p3 = bb.stage2(y); p4 = bb.attn3(bb.stage3(p3)); s4 = bb.stage4(p4)
    p5_no_sppf = s4
    p5_sppf = bb.sppf(s4)
check("SPPF changes P5 values", not torch.allclose(p5_no_sppf, p5_sppf, atol=1e-6))
check("SPPF preserves P5 channel count", p5_sppf.shape == p5_no_sppf.shape,
      f"{tuple(p5_sppf.shape)}")

# --------------------------------------------------------------------------
section("3. Parameter cost of each change (nc=80, det, quantized)")

def total(**kw):
    m, _ = build_model(task='det', variant='quantized', nc=80, **kw)
    return count_parameters(m)['total']

pre = total(use_sppf=False, neck_k=3, head_k=3)
sppf_only = total(use_sppf=True, neck_k=3, head_k=3)
k5_only = total(use_sppf=False, neck_k=5, head_k=5)
r2 = total(use_sppf=True, neck_k=5, head_k=5)

print(f"    pre-R2 (sppf off, k=3)      : {pre/1e6:.4f} M")
print(f"    + SPPF only                 : {sppf_only/1e6:.4f} M  (+{(sppf_only-pre)/1e3:.1f} k)")
print(f"    + k=5 only                  : {k5_only/1e6:.4f} M  (+{(k5_only-pre)/1e3:.1f} k)")
print(f"    R2 (both)                   : {r2/1e6:.4f} M  (+{(r2-pre)/1e3:.1f} k, "
      f"{100*(r2-pre)/pre:.1f}%)")

check("pre-R2 total matches the 0.218M baseline", abs(pre - 218_000) < 6_000,
      f"{pre}")
check("SPPF costs ~32k params", 25_000 < (sppf_only - pre) < 40_000,
      f"{sppf_only - pre}")
check("k=3->5 is cheap relative to SPPF", (k5_only - pre) < (sppf_only - pre),
      f"k5 +{k5_only - pre}, sppf +{sppf_only - pre}")

# --------------------------------------------------------------------------
section("4. Backward compatibility — pre-R2 checkpoint reload")

old_model, _ = build_model(task='det', variant='quantized', nc=20,
                           use_sppf=False, neck_k=3, head_k=3)
old_sd = old_model.state_dict()

old_model_legacy, _ = build_model(task='det', variant='quantized', nc=20,
                                  use_sppf=False, neck_k=3, head_k=3,
                                  use_obj=True, box_mode='exp')
old_sd = old_model_legacy.state_dict()
arch = infer_arch_from_state_dict(old_sd)
check("detects use_sppf=False", arch['use_sppf'] is False, str(arch))
check("detects neck_k=3", arch['neck_k'] == 3, str(arch))
check("detects head_k=3", arch['head_k'] == 3, str(arch))
check("detects use_obj=True", arch['use_obj'] is True, str(arch))
check("detects box_mode='exp'", arch['box_mode'] == 'exp', str(arch))

rebuilt, _ = build_model(task='det', variant='quantized', nc=20, **arch)
try:
    rebuilt.load_state_dict(old_sd, strict=True)
    check("pre-R2 checkpoint loads into rebuilt model", True)
except Exception as e:
    check("pre-R2 checkpoint loads into rebuilt model", False, str(e)[:160])

# a default (R2) model must NOT silently accept a pre-R2 checkpoint
default_model, _ = build_model(task='det', variant='quantized', nc=20)
try:
    default_model.load_state_dict(old_sd, strict=True)
    check("R2 model correctly rejects pre-R2 checkpoint (strict)", False,
          "loaded when it should have failed")
except Exception:
    check("R2 model correctly rejects pre-R2 checkpoint (strict)", True)

# and the R2 round-trip must work
r2_model, _ = build_model(task='det', variant='quantized', nc=20)
r2_arch = infer_arch_from_state_dict(r2_model.state_dict())
check("detects R2 arch correctly",
      r2_arch == {'use_sppf': True, 'neck_k': 5, 'head_k': 5,
                  'use_obj': False, 'box_mode': 'ltrb'}, str(r2_arch))

# Wrapped heads (seg/pose/obb) nest the detect head one level deeper, so they take
# the fallback lookup path. That branch is where the first version of this function
# crashed on tensor truth-testing — exercise every task, not just 'det'.
for task in ('det', 'seg', 'pose', 'obb'):
    nc_t = 1 if task == 'pose' else 20
    for want in ({'use_sppf': True, 'neck_k': 5, 'head_k': 5,
                  'use_obj': False, 'box_mode': 'ltrb'},
                 {'use_sppf': False, 'neck_k': 3, 'head_k': 3,
                  'use_obj': True, 'box_mode': 'exp'}):
        try:
            m, _ = build_model(task=task, variant='quantized', nc=nc_t, **want)
            got = infer_arch_from_state_dict(m.state_dict())
            check(f"infer_arch round-trip {task} obj={want['use_obj']} {want['box_mode']}",
                  got == want, f"got {got}")
        except Exception as e:
            check(f"infer_arch round-trip {task} obj={want['use_obj']} {want['box_mode']}",
                  False, f"{type(e).__name__}: {str(e)[:110]}")

# Mixed config: the two kernel sizes are independent knobs and must be read back
# independently, not inferred from one another.
m_mixed, _ = build_model(task='det', variant='quantized', nc=20,
                         use_sppf=True, neck_k=3, head_k=5)
got_mixed = infer_arch_from_state_dict(m_mixed.state_dict())
check("infer_arch reads neck_k and head_k independently",
      got_mixed == {'use_sppf': True, 'neck_k': 3, 'head_k': 5,
                    'use_obj': False, 'box_mode': 'ltrb'}, str(got_mixed))

# --------------------------------------------------------------------------
section("5. Kernel sizes are physically 5x5")

sd = r2_model.state_dict()
neck_w = sd['neck.td_conv4.dw.conv.weight']
head_w = sd['head.cls_convs.0.0.dw.conv.weight']
check("neck depthwise weight is 5x5", tuple(neck_w.shape[-2:]) == (5, 5), str(tuple(neck_w.shape)))
check("head depthwise weight is 5x5", tuple(head_w.shape[-2:]) == (5, 5), str(tuple(head_w.shape)))

down_w = sd['neck.bu_down3.dw.conv.weight']
check("neck stride-2 downsample also 5x5", tuple(down_w.shape[-2:]) == (5, 5),
      str(tuple(down_w.shape)))

# --------------------------------------------------------------------------
section("6. ltrb codec — reach and round-trip")

from tinyYOLO.utils.boxcodec import decode_boxes, make_grid  # noqa: E402

# The bug ltrb exists to fix: under 'exp' a cell's centre reach is (-0.5,+1.5)
# cells, so an edge cell of a large object cannot emit that object's centre.
# Under 'ltrb' any in-box cell can represent the box exactly.
W = H = 10                       # P5 grid at 320px
gt = torch.tensor([0.50, 0.50, 0.50, 0.50])   # centred box spanning 5x5 cells
gi = torch.tensor([2.0])                       # cell 2.5 cells left of GT centre
gj = torch.tensor([5.0])

def best_iou(mode, iters=400):
    """Fit one cell's raw output to the GT and report the IoU it can reach."""
    raw = torch.zeros(1, 4, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=0.1)
    for _ in range(iters):
        opt.zero_grad()
        d = decode_boxes(raw, gi, gj, W, H, mode=mode)[0]
        px1, py1 = d[0] - d[2] / 2, d[1] - d[3] / 2
        px2, py2 = d[0] + d[2] / 2, d[1] + d[3] / 2
        tx1, ty1 = gt[0] - gt[2] / 2, gt[1] - gt[3] / 2
        tx2, ty2 = gt[0] + gt[2] / 2, gt[1] + gt[3] / 2
        inter = ((torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
                 * (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0))
        union = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0) + gt[2] * gt[3] - inter
        loss = 1 - inter / (union + 1e-9)
        loss.backward()
        opt.step()
    return float(1 - loss.item())

iou_exp = best_iou('exp')
iou_ltrb = best_iou('ltrb')
print(f"    edge cell fitted to a 5x5-cell GT:  exp -> IoU {iou_exp:.3f}   "
      f"ltrb -> IoU {iou_ltrb:.3f}")
check("ltrb lets an edge cell reach the target (IoU > 0.95)", iou_ltrb > 0.95,
      f"{iou_ltrb:.3f}")
check("exp cannot (reproduces the observed loss floor)", iou_exp < 0.85,
      f"{iou_exp:.3f}")

# decode must be finite and positive-sized everywhere
raw = torch.randn(2, 4, 10, 10) * 5
from tinyYOLO.utils.boxcodec import decode_grid as _dg  # noqa: E402
for mode in ('ltrb', 'exp'):
    cx, cy, w, h = _dg(raw, mode=mode)
    check(f"{mode} decode finite", bool(torch.isfinite(torch.stack([cx, cy, w, h])).all()))
    check(f"{mode} decode positive w/h", bool((w > 0).all() and (h > 0).all()))

# --------------------------------------------------------------------------
section("7. Loss — dense soft targets, gradient flow, obj removal")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location(
    "tytrain", Path(__file__).resolve().parent / "train.py")
tytrain = importlib.util.module_from_spec(spec)
sys.modules["tytrain"] = tytrain
spec.loader.exec_module(tytrain)

NC = 4
model_r2, _ = build_model(task='det', variant='quantized', nc=NC)
loss_r2 = tytrain.MultiTaskLoss(task='det', nc=NC, use_obj=False, box_mode='ltrb')

# two images, two objects each
tgt = torch.zeros(2, 8, 5)
tgt[0, 0] = torch.tensor([0.0, 0.50, 0.50, 0.40, 0.40])
tgt[0, 1] = torch.tensor([1.0, 0.20, 0.25, 0.10, 0.10])
tgt[1, 0] = torch.tensor([2.0, 0.70, 0.30, 0.25, 0.30])

out = model_r2(torch.randn(2, 3, 320, 320))
check("R2 head emits 4+nc channels", out[0].shape[1] == NC + 4, f"{out[0].shape[1]}")

total, parts = loss_r2(out, tgt)
print(f"    loss parts: {parts}")
check("loss is finite", bool(torch.isfinite(total)))
check("obj term is exactly zero when objectness removed", parts['obj'] == 0.0,
      f"obj={parts['obj']}")
check("cls term is non-zero", parts['cls'] > 0)
check("box term is non-zero", parts['box'] > 0)

total.backward()
gz = {n: (p.grad is not None and bool(p.grad.abs().sum() > 0))
      for n, p in model_r2.named_parameters()}
check("cls_preds receive gradient", all(v for n, v in gz.items() if 'cls_preds' in n))
check("reg_preds receive gradient", all(v for n, v in gz.items() if 'reg_preds' in n))
check("backbone receives gradient", any(v for n, v in gz.items() if 'backbone' in n))
check("sppf receives gradient", any(v for n, v in gz.items() if 'sppf' in n))

# The decisive property of 2a: background cells must now be supervised. Under the
# old positives-only loss, a cell with no assignment produced no cls gradient at
# all, so its logits drifted upward and the confidence signal was worthless.
bg_probe, _ = build_model(task='det', variant='quantized', nc=NC)
o = bg_probe(torch.randn(1, 3, 320, 320))
empty = torch.zeros(1, 8, 5)          # no objects at all -> every cell is background
t_empty, parts_empty = tytrain.MultiTaskLoss(task='det', nc=NC)(o, empty)
check("empty image still produces a cls loss (background supervised)",
      parts_empty['cls'] > 0, f"cls={parts_empty['cls']:.4f}")
check("empty image produces no box loss", parts_empty['box'] == 0.0,
      f"box={parts_empty['box']}")

# legacy path must still work
model_lg, _ = build_model(task='det', variant='quantized', nc=NC,
                          use_obj=True, box_mode='exp')
loss_lg = tytrain.MultiTaskLoss(task='det', nc=NC, use_obj=True, box_mode='exp')
out_lg = model_lg(torch.randn(2, 3, 320, 320))
check("legacy head emits 5+nc channels", out_lg[0].shape[1] == NC + 5, f"{out_lg[0].shape[1]}")
tl, pl = loss_lg(out_lg, tgt)
check("legacy loss finite and has obj term", bool(torch.isfinite(tl)) and pl['obj'] > 0,
      f"{pl}")

# --------------------------------------------------------------------------
section("8. Postprocess infers layout from channel count")

from tinyYOLO.utils.postprocess import decode_predictions  # noqa: E402

with torch.no_grad():
    o_r2 = model_r2(torch.randn(2, 3, 320, 320))
    o_lg = model_lg(torch.randn(2, 3, 320, 320))

d_r2 = decode_predictions(o_r2, imgsz=320, conf_thresh=0.001, nc=NC, box_mode='ltrb')
d_lg = decode_predictions(o_lg, imgsz=320, conf_thresh=0.001, nc=NC, box_mode='exp')
check("decode_predictions handles no-obj head", len(d_r2) == 2 and d_r2[0].shape[1] == 6,
      f"{[tuple(d.shape) for d in d_r2]}")
check("decode_predictions handles legacy head", len(d_lg) == 2 and d_lg[0].shape[1] == 6,
      f"{[tuple(d.shape) for d in d_lg]}")
check("predicted class ids within range",
      bool((d_r2[0][:, 5] < NC).all() and (d_r2[0][:, 5] >= 0).all()))

# --------------------------------------------------------------------------
section("Result")
if FAILURES:
    print(f"  {len(FAILURES)} FAILED:")
    for f in FAILURES:
        print(f"    - {f}")
    sys.exit(1)
print("  All checks passed.")
sys.exit(0)
