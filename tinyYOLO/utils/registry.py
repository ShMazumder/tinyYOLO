"""
TinyYOLO Module Registry
==========================
Registers custom modules with the Ultralytics framework so they can be
referenced by name in YAML model configuration files.

Usage:
    from tinyYOLO.utils.registry import register_all
    register_all()  # Call once before YOLO(config.yaml)
"""

import importlib


def register_all():
    """
    Register all tinyYOLO custom modules with Ultralytics.
    Must be called before loading a model from a tinyYOLO YAML config.
    """
    from tinyYOLO.modules.common import (
        ConvBNAct, DWConv, GhostConv, GhostBottleneck,
        SEBlock, ECABlock, LightSpatialAttn, C3Ghost, Concat,
    )
    from tinyYOLO.modules.backbone import TinyBackbone
    from tinyYOLO.modules.neck import LitePAN
    from tinyYOLO.modules.heads import (
        TinyDetect, TinySegment, TinyPose, TinyClassify, TinyOBB,
    )

    custom_modules = {
        'ConvBNAct': ConvBNAct,
        'DWConv': DWConv,
        'GhostConv': GhostConv,
        'GhostBottleneck': GhostBottleneck,
        'SEBlock': SEBlock,
        'ECABlock': ECABlock,
        'LightSpatialAttn': LightSpatialAttn,
        'C3Ghost': C3Ghost,
        'TinyBackbone': TinyBackbone,
        'LitePAN': LitePAN,
        'TinyDetect': TinyDetect,
        'TinySegment': TinySegment,
        'TinyPose': TinyPose,
        'TinyClassify': TinyClassify,
        'TinyOBB': TinyOBB,
    }

    # Inject into ultralytics namespace
    try:
        import ultralytics.nn.modules as ul_modules
        for name, cls in custom_modules.items():
            setattr(ul_modules, name, cls)
            if hasattr(ul_modules, '__all__'):
                if name not in ul_modules.__all__:
                    ul_modules.__all__.append(name)
        print(f"[tinyYOLO] Registered {len(custom_modules)} custom modules with Ultralytics")
    except ImportError:
        print("[tinyYOLO] Ultralytics not found — modules available as standalone PyTorch")

    return custom_modules
