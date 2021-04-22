# -*- coding:utf-8 -*-
# author: Xinge
# @file: cylinder_spconv_3d.py
from typing import List

import gorilla
import torch
from torch import nn

@gorilla.MODELS.register_module()
class CylinderAsym(nn.Module):
    def __init__(self,
                 cylin_model_cfg,
                 segmentator_spconv_cfg,
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.cylinder_3d_generator = gorilla.build_module(cylin_model_cfg)
        self.cylinder_3d_spconv_seg = gorilla.build_module(segmentator_spconv_cfg)

    def forward(self,
                pt_features: torch.Tensor,
                voxels: torch.Tensor):
        r"""
        Cylinder forward

        Args:
            pt_features (List[torch.Tensor, [N, 9]]): list of point-wise features
            voxels (List[torch.Tensor, [N, 4]]): list of point-wise indices
        """
        # coords: [num_valid_vxoel, 4]
        # features_3d: [num_valid_vxoel, C], features of each non-empty voxel
        coords, features_3d = self.cylinder_3d_generator(pt_features, voxels)

        spatial_features = self.cylinder_3d_spconv_seg(features_3d, coords)

        return spatial_features
