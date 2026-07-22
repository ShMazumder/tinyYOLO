"""
TinyYOLO Backbone
==================
Ghost-based backbone returning 3 feature scales (P3, P4, P5).

Two variants:
  - Standard:  SiLU + LightSpatialAttn + SEBlock  (best accuracy)
  - Quantized: ReLU6 + ECABlock only               (INT8-safe)
"""

import torch
import torch.nn as nn
from tinyYOLO.modules.common import (
    ConvBNAct, GhostBottleneck, SEBlock, ECABlock, LightSpatialAttn, SPPF,
)


class TinyBackbone(nn.Module):
    """
    TinyYOLO Backbone with Ghost modules.

    Returns features at 3 scales: P3(/8), P4(/16), P5(/32).

    Args:
        channels: Channel counts for [stem, stage1, stage2, stage3, stage4].
        depths:   Number of GhostBottleneck repeats per stage.
        variant:  'standard' (SiLU + spatial attn) or 'quantized' (ReLU6 + ECA).
        attention: Override attention type independently of variant.
                   Options: 'default' (use variant's default), 'se', 'eca', 'none'.
                   When 'default', standard uses LightSpatial+SE, quantized uses ECA+ECA.
        use_sppf: Insert an SPPF pyramid-pooling block after stage4. Every YOLO
                  since v4 has an SPP-family module; this backbone previously had
                  none, leaving it with no global context at any layer. Costs
                  ~32k params at the default width. Set False to reproduce the
                  pre-R2 architecture.
        sppf_k: SPPF maxpool kernel (5 → effective receptive fields 5/9/13).
    """

    STANDARD_CHANNELS = [16, 24, 40, 80, 160]
    STANDARD_DEPTHS = [1, 1, 2, 3, 2]

    def __init__(self, channels=None, depths=None, variant='standard', attention='default',
                 use_sppf=True, sppf_k=5):
        super().__init__()
        channels = channels or self.STANDARD_CHANNELS
        depths = depths or self.STANDARD_DEPTHS
        act = 'silu' if variant == 'standard' else 'relu6'

        # Stem: 3 → channels[0], stride 2
        self.stem = ConvBNAct(3, channels[0], 3, 2, act=act)

        # Build stages
        self.stage1 = self._make_stage(channels[0], channels[1], depths[1], 2, act)
        self.stage2 = self._make_stage(channels[1], channels[2], depths[2], 2, act)
        self.stage3 = self._make_stage(channels[2], channels[3], depths[3], 2, act)
        self.stage4 = self._make_stage(channels[3], channels[4], depths[4], 2, act)

        # Attention — decoupled from variant to support cross-ablation experiments.
        # 'default' preserves original behavior; explicit values override it.
        if attention == 'default':
            if variant == 'standard':
                self.attn3 = LightSpatialAttn(channels[3])
                self.attn4 = SEBlock(channels[4])
            else:  # quantized
                self.attn3 = ECABlock(channels[3])
                self.attn4 = ECABlock(channels[4])
        elif attention == 'eca':
            self.attn3 = ECABlock(channels[3])
            self.attn4 = ECABlock(channels[4])
        elif attention == 'se':
            self.attn3 = LightSpatialAttn(channels[3])
            self.attn4 = SEBlock(channels[4])
        elif attention == 'none':
            self.attn3 = nn.Identity()
            self.attn4 = nn.Identity()
        else:
            raise ValueError(f"Unknown attention type: {attention}. Use 'default', 'eca', 'se', or 'none'.")

        # Pyramid pooling on the deepest map. Placed BEFORE the P5 attention so the
        # channel recalibration operates on context-enriched features. Channel count
        # is preserved (c -> c), so the neck is unaffected either way.
        self.use_sppf = use_sppf
        self.sppf = SPPF(channels[4], channels[4], k=sppf_k, act=act) if use_sppf else nn.Identity()

        # Store output channel info for neck
        self.out_channels = [channels[2], channels[3], channels[4]]

    @staticmethod
    def _make_stage(c1, c2, n, stride, act):
        """Build a stage with n GhostBottleneck blocks."""
        layers = [GhostBottleneck(c1, c2, s=stride, act=act)]
        for _ in range(n - 1):
            layers.append(GhostBottleneck(c2, c2, act=act))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)            # /2
        x = self.stage1(x)          # /4
        p3 = self.stage2(x)         # /8
        p4 = self.attn3(self.stage3(p3))   # /16
        p5 = self.attn4(self.sppf(self.stage4(p4)))   # /32
        return [p3, p4, p5]

    def get_out_channels(self):
        """Return output channels for neck input."""
        return self.out_channels


def build_backbone(variant='standard', width_mult=1.0, attention='default', use_sppf=True):
    """
    Factory function for building backbone with width multiplier.

    Args:
        variant: 'standard' or 'quantized'
        width_mult: Channel width multiplier (0.25, 0.5, 0.75, 1.0)
        attention: Override attention type ('default', 'eca', 'se', 'none')
        use_sppf: Insert SPPF after stage4 (default True as of R2).
    """
    base_channels = [16, 24, 40, 80, 160]
    channels = [max(8, int(c * width_mult) // 8 * 8) for c in base_channels]
    return TinyBackbone(channels=channels, variant=variant, attention=attention,
                        use_sppf=use_sppf)
