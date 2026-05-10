"""
TinyYOLO Detection Heads
=========================
Task-specific heads for all 5 tasks:
  - TinyDetect:   Object detection (anchor-free, NMS-free capable)
  - TinySegment:  Instance segmentation (detection + proto-masks)
  - TinyPose:     Pose estimation (detection + keypoints)
  - TinyClassify: Image classification (global pool + FC)
  - TinyOBB:      Oriented bounding box detection

All detection-based heads use a decoupled design (separate cls/reg branches).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tinyYOLO.modules.common import ConvBNAct, DWConv


class TinyDetect(nn.Module):
    """
    Anchor-free detection head (decoupled classification + regression).

    No DFL (Distribution Focal Loss) — direct bbox regression for
    lighter deployment. Compatible with NMS-free training via
    consistent dual assignment.

    Args:
        nc: Number of classes.
        in_channels: List of input channels from neck (one per scale).
        reg_max: Max regression range (0 = no DFL, direct regression).
    """

    def __init__(self, nc=80, in_channels=None, reg_max=0):
        super().__init__()
        if in_channels is None:
            in_channels = [64, 64, 64]

        self.nc = nc
        self.nl = len(in_channels)  # Number of detection layers (scales)
        self.reg_max = reg_max
        self.no = nc + 4  # Outputs per anchor: nc classes + 4 bbox coords

        # Per-scale decoupled heads
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        for ch in in_channels:
            # Classification branch
            self.cls_convs.append(nn.Sequential(
                DWConv(ch, ch, 3, 1, act='silu'),
                DWConv(ch, ch, 3, 1, act='silu'),
            ))
            self.cls_preds.append(
                nn.Conv2d(ch, nc, 1, bias=True)
            )
            # Regression branch
            self.reg_convs.append(nn.Sequential(
                DWConv(ch, ch, 3, 1, act='silu'),
                DWConv(ch, ch, 3, 1, act='silu'),
            ))
            self.reg_preds.append(
                nn.Conv2d(ch, 4, 1, bias=True)
            )

        self._init_bias()

    def _init_bias(self):
        """Initialize classification bias for stable early training."""
        for cls_pred in self.cls_preds:
            # Bias init: -log((1 - prior) / prior) where prior = 0.01
            nn.init.constant_(cls_pred.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, features):
        """
        Args:
            features: List of feature maps from neck [F3, N4, N5].
        Returns:
            List of (cls_pred, reg_pred) tuples per scale.
        """
        outputs = []
        for i, feat in enumerate(features):
            cls_feat = self.cls_convs[i](feat)
            reg_feat = self.reg_convs[i](feat)
            cls_out = self.cls_preds[i](cls_feat)  # [B, nc, H, W]
            reg_out = self.reg_preds[i](reg_feat)  # [B, 4, H, W]
            outputs.append(torch.cat([reg_out, cls_out], dim=1))  # [B, 4+nc, H, W]
        return outputs


class TinySegment(nn.Module):
    """
    Instance segmentation head = TinyDetect + proto-mask branch.

    Generates prototype masks and per-detection mask coefficients.

    Args:
        nc: Number of classes.
        in_channels: List of input channels from neck.
        nm: Number of proto-mask channels.
    """

    def __init__(self, nc=80, in_channels=None, nm=32):
        super().__init__()
        if in_channels is None:
            in_channels = [64, 64, 64]
        self.nm = nm
        self.detect = TinyDetect(nc=nc, in_channels=in_channels)

        # Mask coefficient prediction (appended to detection)
        self.mask_preds = nn.ModuleList()
        for ch in in_channels:
            self.mask_preds.append(nn.Conv2d(ch, nm, 1, bias=True))

        # Proto-mask branch (from highest-res feature)
        self.proto = nn.Sequential(
            ConvBNAct(in_channels[0], in_channels[0], 3, 1, act='silu'),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBNAct(in_channels[0], in_channels[0] // 2, 3, 1, act='silu'),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBNAct(in_channels[0] // 2, nm, 1, 1, act='silu'),
        )

    def forward(self, features):
        """Returns detection outputs + mask coefficients + proto-masks."""
        det_out = self.detect(features)

        # Append mask coefficients to each scale
        for i in range(len(features)):
            mask_coeff = self.mask_preds[i](features[i])
            det_out[i] = torch.cat([det_out[i], mask_coeff], dim=1)

        # Proto-masks from P3 (highest resolution)
        proto = self.proto(features[0])  # [B, nm, H*4, W*4]

        return det_out, proto


class TinyPose(nn.Module):
    """
    Pose estimation head = TinyDetect + keypoint regression.

    Args:
        nc: Number of classes (typically 1 for person).
        in_channels: List of input channels from neck.
        nk: Number of keypoints (17 for COCO).
        ndim: Dimensions per keypoint (2 for x,y or 3 for x,y,visibility).
    """

    def __init__(self, nc=1, in_channels=None, nk=17, ndim=3):
        super().__init__()
        if in_channels is None:
            in_channels = [64, 64, 64]
        self.nk = nk
        self.ndim = ndim
        self.detect = TinyDetect(nc=nc, in_channels=in_channels)

        # Keypoint prediction branch
        self.kpt_preds = nn.ModuleList()
        for ch in in_channels:
            self.kpt_preds.append(nn.Sequential(
                DWConv(ch, ch, 3, 1, act='silu'),
                nn.Conv2d(ch, nk * ndim, 1, bias=True),
            ))

    def forward(self, features):
        """Returns detection outputs + keypoint predictions."""
        det_out = self.detect(features)
        kpt_out = []
        for i, feat in enumerate(features):
            kpt = self.kpt_preds[i](feat)  # [B, nk*ndim, H, W]
            kpt_out.append(kpt)
        return det_out, kpt_out


class TinyClassify(nn.Module):
    """
    Image classification head.
    Global average pooling + dropout + linear classifier.

    Args:
        in_channel: Input channels (from last backbone stage).
        nc: Number of classes.
        dropout: Dropout rate.
    """

    def __init__(self, in_channel=160, nc=1000, dropout=0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(in_channel, nc)

    def forward(self, features):
        """
        Args:
            features: List of [P3, P4, P5] — only P5 is used.
        """
        x = features[-1] if isinstance(features, list) else features
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)


class TinyOBB(nn.Module):
    """
    Oriented Bounding Box detection head = TinyDetect + angle regression.

    Args:
        nc: Number of classes.
        in_channels: List of input channels from neck.
    """

    def __init__(self, nc=80, in_channels=None):
        super().__init__()
        if in_channels is None:
            in_channels = [64, 64, 64]
        self.detect = TinyDetect(nc=nc, in_channels=in_channels)

        # Angle prediction (1 value per anchor)
        self.angle_preds = nn.ModuleList()
        for ch in in_channels:
            self.angle_preds.append(nn.Sequential(
                DWConv(ch, ch, 3, 1, act='silu'),
                nn.Conv2d(ch, 1, 1, bias=True),
            ))

    def forward(self, features):
        """Returns detection outputs + angle predictions."""
        det_out = self.detect(features)
        for i in range(len(features)):
            angle = self.angle_preds[i](features[i])
            det_out[i] = torch.cat([det_out[i], angle], dim=1)
        return det_out
