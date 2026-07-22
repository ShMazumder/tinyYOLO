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
                exp_c = nc + 5 + EXPECTED_EXTRA.get(task, 0)
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

arch = infer_arch_from_state_dict(old_sd)
check("detects use_sppf=False", arch['use_sppf'] is False, str(arch))
check("detects neck_k=3", arch['neck_k'] == 3, str(arch))
check("detects head_k=3", arch['head_k'] == 3, str(arch))

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
      r2_arch == {'use_sppf': True, 'neck_k': 5, 'head_k': 5}, str(r2_arch))

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
section("Result")
if FAILURES:
    print(f"  {len(FAILURES)} FAILED:")
    for f in FAILURES:
        print(f"    - {f}")
    sys.exit(1)
print("  All checks passed.")
sys.exit(0)
