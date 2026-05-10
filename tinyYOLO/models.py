"""
TinyYOLO Model Builder
========================
Assembles complete models from backbone + neck + head for any task/variant.
"""

import torch.nn as nn
from tinyYOLO.modules.backbone import TinyBackbone
from tinyYOLO.modules.neck import LitePAN
from tinyYOLO.modules.heads import (
    TinyDetect, TinySegment, TinyPose, TinyClassify, TinyOBB,
)


class TinyYOLOModel(nn.Module):
    """
    Complete TinyYOLO model: Backbone → Neck → Head.

    Args:
        backbone: TinyBackbone instance.
        neck: LitePAN instance (None for classification).
        head: Task-specific head instance.
        task: Task name string.
    """

    def __init__(self, backbone, neck, head, task='det'):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.task = task

    def forward(self, x):
        features = self.backbone(x)

        if self.task == 'cls':
            return self.head(features)

        fused = self.neck(features)
        return self.head(fused)


HEAD_MAP = {
    'det': TinyDetect,
    'seg': TinySegment,
    'pose': TinyPose,
    'cls': TinyClassify,
    'obb': TinyOBB,
}

HEAD_KWARGS = {
    'det': lambda nc, ch: {'nc': nc, 'in_channels': ch},
    'seg': lambda nc, ch: {'nc': nc, 'in_channels': ch, 'nm': 32},
    'pose': lambda nc, ch: {'nc': 1, 'in_channels': ch, 'nk': 17, 'ndim': 3},
    'cls': lambda nc, ch: {'in_channel': 160, 'nc': nc},
    'obb': lambda nc, ch: {'nc': nc, 'in_channels': ch},
}


def build_model(task='det', variant='standard', nc=80, width_mult=1.0):
    """
    Build a complete TinyYOLO model.

    Args:
        task: One of 'det', 'seg', 'pose', 'cls', 'obb'.
        variant: 'standard' or 'quantized'.
        nc: Number of classes.
        width_mult: Width multiplier for backbone channels.

    Returns:
        (model, info_dict)
    """
    # Build backbone
    base_channels = [16, 24, 40, 80, 160]
    channels = [max(8, int(c * width_mult) // 8 * 8) for c in base_channels]
    backbone = TinyBackbone(channels=channels, variant=variant)

    # Build neck (skip for classification)
    neck = None
    neck_out = None
    if task != 'cls':
        in_ch = backbone.get_out_channels()
        neck_ch = max(8, int(64 * width_mult) // 8 * 8)
        act = 'silu' if variant == 'standard' else 'relu6'
        neck = LitePAN(in_channels=in_ch, out_channel=neck_ch, act=act)
        neck_out = neck.get_out_channels()

    # Build head
    head_cls = HEAD_MAP[task]
    head_kwargs = HEAD_KWARGS[task](nc, neck_out)

    # For classification, adjust input channel based on backbone
    if task == 'cls':
        head_kwargs['in_channel'] = channels[-1]

    head = head_cls(**head_kwargs)

    # Assemble
    model = TinyYOLOModel(backbone, neck, head, task)

    # Info
    total_params = sum(p.numel() for p in model.parameters())
    info = {
        'task': task,
        'variant': variant,
        'nc': nc,
        'width_mult': width_mult,
        'backbone_channels': channels,
        'neck_channel': neck_ch if neck else None,
        'total_params': total_params,
        'total_params_M': round(total_params / 1e6, 2),
    }

    return model, info
