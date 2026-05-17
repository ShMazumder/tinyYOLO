"""
TinyYOLO Shared Building Blocks
================================
All fundamental modules used across backbone, neck, and heads.
Two activation profiles:
  - Standard: SiLU (best accuracy, FP32/FP16)
  - Quantized: ReLU6 (INT8-safe, edge deployment)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    """Auto-compute padding for 'same' output when stride=1."""
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


def get_activation(act='silu'):
    """Return activation module by name string."""
    if act == 'silu':
        return nn.SiLU(inplace=True)
    elif act == 'relu6':
        return nn.ReLU6(inplace=True)
    elif act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif act is None or act == 'none' or act is False:
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation: {act}")


# ---------------------------------------------------------------------------
# Core Convolution Blocks
# ---------------------------------------------------------------------------

class ConvBNAct(nn.Module):
    """Standard Conv2d + BatchNorm + Activation."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act='silu'):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d),
                              groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Forward after fusing Conv+BN (inference optimization)."""
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Separable Convolution = Depthwise + Pointwise."""

    def __init__(self, c1, c2, k=3, s=1, act='silu'):
        super().__init__()
        self.dw = ConvBNAct(c1, c1, k, s, g=c1, act=act)
        self.pw = ConvBNAct(c1, c2, 1, 1, act=act)

    def forward(self, x):
        return self.pw(self.dw(x))


# ---------------------------------------------------------------------------
# Ghost Modules (from GhostNet — cheap feature generation)
# ---------------------------------------------------------------------------

class GhostConv(nn.Module):
    """
    Ghost Convolution: generate feature maps cheaply.
    Uses a primary conv for half the channels, then cheap linear
    transforms (depthwise conv) to generate the rest.
    """

    def __init__(self, c1, c2, k=1, s=1, dw_k=3, ratio=2, act='silu'):
        super().__init__()
        init_ch = c2 // ratio
        new_ch = init_ch * (ratio - 1)
        self.primary = ConvBNAct(c1, init_ch, k, s, act=act)
        self.cheap = ConvBNAct(init_ch, new_ch, dw_k, 1, g=init_ch, act=act)
        
        # Resilient quantized concatenation block
        try:
            import torch.ao.quantization as ao_quant
            self.cat_op = ao_quant.QFunctional()
        except (ImportError, AttributeError):
            try:
                import torch.nn.quantized as nn_quant
                self.cat_op = nn_quant.FloatFunctional()
            except (ImportError, AttributeError):
                self.cat_op = None

    def forward(self, x):
        y = self.primary(x)
        features = [y, self.cheap(y)]
        if self.cat_op is not None and (y.is_quantized or features[1].is_quantized):
            return self.cat_op.cat(features, dim=1)
        return torch.cat(features, dim=1)


class GhostBottleneck(nn.Module):
    """
    Ghost Bottleneck: two GhostConv layers with optional DW stride
    and optional squeeze-excitation.
    """

    def __init__(self, c1, c2, k=3, s=1, use_se=False, act='silu'):
        super().__init__()
        mid = c2 // 2
        self.ghost1 = GhostConv(c1, mid, 1, 1, act=act)

        # Depthwise conv for stride > 1
        if s > 1:
            self.dw = ConvBNAct(mid, mid, k, s, g=mid, act='none')
        else:
            self.dw = None

        # Optional SE attention
        self.se = SEBlock(mid) if use_se else None

        # Second ghost conv (no activation before residual add)
        self.ghost2 = GhostConv(mid, c2, 1, 1, act='none')

        # Shortcut path
        if s > 1 or c1 != c2:
            self.shortcut = nn.Sequential(
                ConvBNAct(c1, c1, k, s, g=c1, act='none'),
                ConvBNAct(c1, c2, 1, 1, act='none'),
            )
        else:
            self.shortcut = nn.Identity()

        # Resilient quantized addition block
        try:
            import torch.ao.quantization as ao_quant
            self.add = ao_quant.QFunctional()
        except (ImportError, AttributeError):
            try:
                import torch.nn.quantized as nn_quant
                self.add = nn_quant.FloatFunctional()
            except (ImportError, AttributeError):
                self.add = None

    def forward(self, x):
        residual = self.shortcut(x)
        y = self.ghost1(x)
        if self.dw is not None:
            y = self.dw(y)
        if self.se is not None:
            y = self.se(y)
        y = self.ghost2(y)
        if self.add is not None:
            return self.add.add(y, residual)
        return y + residual


# ---------------------------------------------------------------------------
# Attention Modules
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Channel attention via global avg pool → FC → ReLU → FC → Sigmoid.
    INT8-safe: uses ReLU + Sigmoid (both quantize well).
    """

    def __init__(self, c, ratio=4):
        super().__init__()
        mid = max(c // ratio, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, c, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ECABlock(nn.Module):
    """
    Efficient Channel Attention (INT8-safe).
    Uses 1D conv instead of FC → much lighter than SE.
    Adaptive kernel size based on channel count.
    """

    def __init__(self, c, k=None):
        super().__init__()
        if k is None:
            t = int(abs(math.log2(c) + 1) / 2)
            k = t if t % 2 else t + 1
            k = max(k, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, 1, c)
        y = self.sigmoid(self.conv(y)).view(b, c, 1, 1)
        return x * y


class LightSpatialAttn(nn.Module):
    """
    Lightweight Spatial Attention.
    Concatenates channel-wise avg and max → 7x7 conv → sigmoid mask.
    NOT INT8-safe (uses channel reduction ops).
    Standard architecture only.
    """

    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, 1, 3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg, mx], dim=1))
        return x * attn


# ---------------------------------------------------------------------------
# CSP Blocks
# ---------------------------------------------------------------------------

class C3Ghost(nn.Module):
    """CSP Bottleneck with Ghost convolutions (3 convs)."""

    def __init__(self, c1, c2, n=1, shortcut=True, act='silu'):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = ConvBNAct(c1, c_, 1, 1, act=act)
        self.cv2 = ConvBNAct(c1, c_, 1, 1, act=act)
        self.cv3 = ConvBNAct(2 * c_, c2, 1, 1, act=act)
        self.m = nn.Sequential(
            *[GhostBottleneck(c_, c_, act=act) for _ in range(n)]
        )
        
        # Resilient quantized concatenation block
        try:
            import torch.ao.quantization as ao_quant
            self.cat_op = ao_quant.QFunctional()
        except (ImportError, AttributeError):
            try:
                import torch.nn.quantized as nn_quant
                self.cat_op = nn_quant.FloatFunctional()
            except (ImportError, AttributeError):
                self.cat_op = None

    def forward(self, x):
        feats = [self.m(self.cv1(x)), self.cv2(x)]
        if self.cat_op is not None and any(t.is_quantized for t in feats):
            return self.cv3(self.cat_op.cat(feats, dim=1))
        return self.cv3(torch.cat(feats, dim=1))


# ---------------------------------------------------------------------------
# Utility: Concat (Ultralytics-compatible)
# ---------------------------------------------------------------------------

class Concat(nn.Module):
    """Concatenate tensors along a dimension."""

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
        
        # Resilient quantized concatenation block
        try:
            import torch.ao.quantization as ao_quant
            self.cat_op = ao_quant.QFunctional()
        except (ImportError, AttributeError):
            try:
                import torch.nn.quantized as nn_quant
                self.cat_op = nn_quant.FloatFunctional()
            except (ImportError, AttributeError):
                self.cat_op = None

    def forward(self, x):
        if self.cat_op is not None and any(t.is_quantized for t in x if isinstance(t, torch.Tensor)):
            return self.cat_op.cat(x, self.dim)
        return torch.cat(x, self.dim)
