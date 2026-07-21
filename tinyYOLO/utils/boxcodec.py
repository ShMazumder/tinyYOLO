"""
TinyYOLO Box Codec
==================
Canonical, grid-anchored, anchor-free box decode shared by BOTH the training
loss and inference post-processing. Keeping a single source of truth prevents
the train/inference parametrization from silently diverging.

Parametrization (per detection scale with grid W x H):

    cx = (gi + 2*sigmoid(tx) - 0.5) / W        # center x, image fraction [0,1]
    cy = (gj + 2*sigmoid(ty) - 0.5) / H        # center y, image fraction [0,1]
    w  = exp(clamp(tw, max=wh_max)) / W         # width,  image fraction
    h  = exp(clamp(th, max=wh_max)) / H         # height, image fraction

Rationale:
  * The center is ANCHORED to its grid cell (gi, gj). A convolutional head is
    translation-equivariant, so it cannot regress an absolute image-space center
    from identical local features. Adding the cell index (gi, gj) is what makes
    localization learnable. The old `sigmoid(t) * imgsz` decode omitted this and
    could only place boxes near the image centre -> mAP collapsed to ~0.
  * `2*sigmoid(t) - 0.5` gives a center range of (-0.5, +1.5) cells (YOLOv5
    style), letting a cell claim its immediate neighbours — useful for
    multi-positive assignment.
  * Size uses exp relative to cell size (anchor-free). At init (t~0) w,h ~= one
    cell, a sensible prior. `clamp(max=wh_max)` keeps it numerically stable.

All outputs are normalized to the [0,1] image fraction. Multiply by `imgsz`
for pixel coordinates.
"""

import os
import torch

WH_MAX = 4.0  # exp(4)=54.6 cells max side; clamp guards against blow-up


def _legacy_decode():
    """A1-ablation switch. When TINYYOLO_LEGACY_DECODE is set, the codec reverts
    to the broken pre-R1.4 `sigmoid` parametrization (no grid anchoring). This
    exists ONLY to reproduce the decode ablation (grid-anchored vs old) as an
    A/B — it is the bug that zeroed mAP; never enable it for a real run."""
    return os.environ.get("TINYYOLO_LEGACY_DECODE", "0") not in ("0", "", "false", "False")


def decode_boxes(raw, gi, gj, W, H, wh_max=WH_MAX):
    """Decode raw head outputs at given grid cells into normalized cxcywh.

    Args:
        raw: [..., 4] raw regression outputs (tx, ty, tw, th).
        gi:  [...] grid x-index (column) of each cell, float or long.
        gj:  [...] grid y-index (row) of each cell, float or long.
        W, H: grid width / height for this scale.
        wh_max: clamp ceiling on tw/th before exp (stability).

    Returns:
        [..., 4] boxes as (cx, cy, w, h) in [0,1] image fraction.
    """
    tx, ty, tw, th = raw[..., 0], raw[..., 1], raw[..., 2], raw[..., 3]

    if _legacy_decode():
        # Broken pre-R1.4 decode (A1 ablation only): absolute normalized center,
        # no grid index -> a conv head cannot localize -> mAP collapses.
        return torch.stack([torch.sigmoid(tx), torch.sigmoid(ty),
                            torch.sigmoid(tw), torch.sigmoid(th)], dim=-1)

    gi = gi.to(raw.dtype)
    gj = gj.to(raw.dtype)
    cx = (gi + 2.0 * torch.sigmoid(tx) - 0.5) / W
    cy = (gj + 2.0 * torch.sigmoid(ty) - 0.5) / H
    w = torch.exp(torch.clamp(tw, max=wh_max)) / W
    h = torch.exp(torch.clamp(th, max=wh_max)) / H
    return torch.stack([cx, cy, w, h], dim=-1)


def make_grid(W, H, device, dtype=torch.float32):
    """Return (gi, gj) meshgrids of shape [H, W] with cell indices."""
    ys = torch.arange(H, device=device, dtype=dtype)
    xs = torch.arange(W, device=device, dtype=dtype)
    gj, gi = torch.meshgrid(ys, xs, indexing='ij')  # gj: rows(y), gi: cols(x)
    return gi, gj


def decode_grid(raw_bchw, imgsz=None, wh_max=WH_MAX):
    """Decode a full [B, 4, H, W] regression map over its whole grid.

    Args:
        raw_bchw: [B, 4, H, W] raw (tx,ty,tw,th) map.
        imgsz: if given, outputs are scaled to pixels; else [0,1] fraction.
        wh_max: clamp ceiling on tw/th.

    Returns:
        cx, cy, w, h each [B, H, W].
    """
    B, _, H, W = raw_bchw.shape
    raw = raw_bchw.permute(0, 2, 3, 1)  # [B, H, W, 4]
    gi, gj = make_grid(W, H, raw.device, raw.dtype)  # [H, W]
    gi = gi.unsqueeze(0).expand(B, H, W)
    gj = gj.unsqueeze(0).expand(B, H, W)
    dec = decode_boxes(raw, gi, gj, W, H, wh_max=wh_max)  # [B,H,W,4] in [0,1]
    cx, cy, w, h = dec[..., 0], dec[..., 1], dec[..., 2], dec[..., 3]
    if imgsz is not None:
        cx, cy, w, h = cx * imgsz, cy * imgsz, w * imgsz, h * imgsz
    return cx, cy, w, h
