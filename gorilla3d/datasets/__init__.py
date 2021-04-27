# Copyright (c) Gorilla-Lab. All rights reserved.
from .utils import (elastic, pc_aug, pc_jitter, pc_flipper, pc_rotator)
from .scannetv2 import (ScanNetV2Inst, ScanNetV2InstTrainVal, ScanNetV2InstTest,
                        visualize_instance_mask, visualize_instance_mask_lite)
from .s3dis import S3DISInst
from .shapenet import ShapeNetPartNormal, ShapenetImplicitRecon
from .kitti import KittiSem, KittiSemRV
from .modelnet import ModelNetCls

import torch
from gorilla import DATASETS, auto_registry

auto_registry(DATASETS, globals(), torch.utils.data.Dataset)

__all__ = [k for k in globals().keys() if not k.startswith("_")]

