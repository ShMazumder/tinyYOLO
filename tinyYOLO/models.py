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

    def load_state_dict(self, state_dict, strict=True, *args, **kwargs):
        """
        Overridden to automatically strip compilation (_orig_mod.) and 
        DDP (module.) prefixes recursively from state dict keys.
        """
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.endswith(('total_ops', 'total_params')):
                continue
            k_clean = k
            while k_clean.startswith('_orig_mod.') or k_clean.startswith('module.'):
                if k_clean.startswith('_orig_mod.'):
                    k_clean = k_clean[len('_orig_mod.'):]
                if k_clean.startswith('module.'):
                    k_clean = k_clean[len('module.'):]
            if k_clean.startswith('model.'):
                k_clean = k_clean[len('model.'):]
            cleaned_state_dict[k_clean] = v
        return super().load_state_dict(cleaned_state_dict, strict=strict, *args, **kwargs)

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
    'det': lambda nc, ch, act, k: {'nc': nc, 'in_channels': ch, 'act': act, 'k': k},
    'seg': lambda nc, ch, act, k: {'nc': nc, 'in_channels': ch, 'nm': 32, 'act': act, 'k': k},
    'pose': lambda nc, ch, act, k: {'nc': 1, 'in_channels': ch, 'nk': 17, 'ndim': 3,
                                    'act': act, 'k': k},
    'cls': lambda nc, ch, act, k: {'in_channel': 160, 'nc': nc},
    'obb': lambda nc, ch, act, k: {'nc': nc, 'in_channels': ch, 'act': act, 'k': k},
}


def infer_arch_from_state_dict(state_dict):
    """Recover R2 architecture flags from a checkpoint's tensor shapes.

    Checkpoints are saved as bare state_dicts with no config attached, so a model
    built with R2 defaults (SPPF on, k=5) cannot load a pre-R2 checkpoint. Rather
    than version the checkpoint format, detect the architecture the same way
    `export.py` already auto-detects `variant` — from the keys and shapes.

    Args:
        state_dict: A (prefix-cleaned) model state dict.

    Returns:
        dict with 'use_sppf', 'neck_k', 'head_k' — pass straight to build_model().
    """
    keys = set(state_dict.keys())

    def _find(suffix):
        for k in keys:
            if k.endswith(suffix):
                return state_dict[k]
        return None

    use_sppf = any('sppf.cv1' in k for k in keys)

    # Depthwise kernel size is literally the spatial extent of the dw weight.
    neck_w = _find('neck.td_conv4.dw.conv.weight')
    head_w = (_find('head.cls_convs.0.0.dw.conv.weight')
              or _find('head.detect.cls_convs.0.0.dw.conv.weight'))

    return {
        'use_sppf': use_sppf,
        'neck_k': int(neck_w.shape[-1]) if neck_w is not None else 3,
        'head_k': int(head_w.shape[-1]) if head_w is not None else 3,
    }


def build_model(task='det', variant='standard', nc=80, width_mult=1.0, attention='default',
                use_sppf=True, neck_k=5, head_k=5):
    """
    Build a complete TinyYOLO model.

    Args:
        task: One of 'det', 'seg', 'pose', 'cls', 'obb'.
        variant: 'standard' or 'quantized'.
        nc: Number of classes.
        width_mult: Width multiplier for backbone channels.
        attention: Override attention type ('default', 'eca', 'se', 'none').
                   When 'default', uses variant's built-in attention config.
        use_sppf: Insert SPPF after backbone stage4. R2 default: True.
        neck_k: Depthwise kernel size in LitePAN. R2 default: 5.
        head_k: Depthwise kernel size in the detection head. R2 default: 5.

    Note:
        The three R2 flags default to the new architecture. To rebuild a pre-R2
        checkpoint, pass `**infer_arch_from_state_dict(sd)`.

    Returns:
        (model, info_dict)
    """
    # Build backbone
    base_channels = [16, 24, 40, 80, 160]
    channels = [max(8, int(c * width_mult) // 8 * 8) for c in base_channels]
    backbone = TinyBackbone(channels=channels, variant=variant, attention=attention,
                            use_sppf=use_sppf)

    # Variant-appropriate activation — used consistently across all components
    act = 'silu' if variant == 'standard' else 'relu6'

    # Build neck (skip for classification)
    neck = None
    neck_out = None
    neck_ch = None
    if task != 'cls':
        in_ch = backbone.get_out_channels()
        neck_ch = max(8, int(64 * width_mult) // 8 * 8)
        neck = LitePAN(in_channels=in_ch, out_channel=neck_ch, act=act, k=neck_k)
        neck_out = neck.get_out_channels()

    # Build head — pass variant-appropriate activation
    head_cls = HEAD_MAP[task]
    head_kwargs = HEAD_KWARGS[task](nc, neck_out, act, head_k)

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
        'attention': attention,
        'nc': nc,
        'width_mult': width_mult,
        'backbone_channels': channels,
        'neck_channel': neck_ch,
        'use_sppf': use_sppf,
        'neck_k': neck_k,
        'head_k': head_k,
        'total_params': total_params,
        'total_params_M': round(total_params / 1e6, 2),
    }

    return model, info
