"""
TinyYOLO Neck — LitePAN
========================
Lightweight PAN+FPN using depthwise separable convolutions.
Takes 3 feature scales from backbone (P3, P4, P5) and
produces 3 fused feature maps for the detection head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tinyYOLO.modules.common import ConvBNAct, DWConv


class LitePAN(nn.Module):
    """
    Lightweight PAN (Path Aggregation Network) + FPN.

    Uses depthwise separable convolutions throughout for efficiency.
    Compatible with both standard and quantized variants.

    Args:
        in_channels: List of 3 channel counts from backbone [P3, P4, P5].
        out_channel: Unified output channel count for all scales.
        act: Activation function ('silu' for standard, 'relu6' for quantized).
    """

    def __init__(self, in_channels=None, out_channel=64, act='silu'):
        super().__init__()
        if in_channels is None:
            in_channels = [40, 80, 160]
        c3, c4, c5 = in_channels

        # --- Lateral convolutions (channel reduction) ---
        self.lat5 = ConvBNAct(c5, out_channel, 1, 1, act=act)
        self.lat4 = ConvBNAct(c4, out_channel, 1, 1, act=act)
        self.lat3 = ConvBNAct(c3, out_channel, 1, 1, act=act)

        # --- Top-down path (FPN: P5 → P4 → P3) ---
        self.td_merge4 = ConvBNAct(out_channel * 2, out_channel, 1, 1, act=act)
        self.td_conv4 = DWConv(out_channel, out_channel, 3, 1, act=act)
        self.td_merge3 = ConvBNAct(out_channel * 2, out_channel, 1, 1, act=act)
        self.td_conv3 = DWConv(out_channel, out_channel, 3, 1, act=act)

        # --- Bottom-up path (PAN: P3 → P4 → P5) ---
        self.bu_down3 = DWConv(out_channel, out_channel, 3, 2, act=act)
        self.bu_merge4 = ConvBNAct(out_channel * 2, out_channel, 1, 1, act=act)
        self.bu_conv4 = DWConv(out_channel, out_channel, 3, 1, act=act)
        self.bu_down4 = DWConv(out_channel, out_channel, 3, 2, act=act)
        self.bu_merge5 = ConvBNAct(out_channel * 2, out_channel, 1, 1, act=act)
        self.bu_conv5 = DWConv(out_channel, out_channel, 3, 1, act=act)

        # Store for head
        self.out_channels = [out_channel, out_channel, out_channel]

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

    def _cat(self, x, y):
        feats = [x, y]
        if self.cat_op is not None and (x.is_quantized or y.is_quantized):
            return self.cat_op.cat(feats, dim=1)
        return torch.cat(feats, dim=1)

    def forward(self, features):
        """
        Args:
            features: List of [P3, P4, P5] from backbone.
        Returns:
            List of [F3, N4, N5] fused feature maps.
        """
        p3, p4, p5 = features

        # Lateral projections
        l5 = self.lat5(p5)
        l4 = self.lat4(p4)
        l3 = self.lat3(p3)

        # Top-down (FPN)
        up5 = F.interpolate(l5, size=l4.shape[2:], mode='nearest')
        f4 = self.td_conv4(self.td_merge4(self._cat(up5, l4)))

        up4 = F.interpolate(f4, size=l3.shape[2:], mode='nearest')
        f3 = self.td_conv3(self.td_merge3(self._cat(up4, l3)))

        # Bottom-up (PAN)
        d3 = self.bu_down3(f3)
        n4 = self.bu_conv4(self.bu_merge4(self._cat(d3, f4)))

        d4 = self.bu_down4(n4)
        n5 = self.bu_conv5(self.bu_merge5(self._cat(d4, l5)))

        return [f3, n4, n5]

    def get_out_channels(self):
        """Return output channels for head input."""
        return self.out_channels
