# Copyright (c) Gorilla-Lab. All rights reserved.
from typing import List

import numpy as np


class PointCloudTransfromer(object):
    def __init__(self,
                 rotate_aug: bool = False,
                 flip_aug: bool = False,
                 scale_aug: bool = False,
                 transform: bool = False,
                 trans_std: List[float] = [0.1, 0.1, 0.1]):
        super().__init__()
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.transform = transform
        self.trans_std = trans_std

    def __call__(self, point: np.array):
        assert point.shape[
            1] >= 3, "the dimension of xyz must be larger than or equal to 3"
        xyz = point[:, :3]
        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 90) - np.pi / 4
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]
        # convert coordinate into polar coordinates

        if self.transform:
            noise_translate = np.array([
                np.random.normal(0, self.trans_std[0], 1),
                np.random.normal(0, self.trans_std[1], 1),
                np.random.normal(0, self.trans_std[2], 1)
            ]).T

            xyz[:, 0:3] += noise_translate

        point[:, :3] = xyz

        return point
