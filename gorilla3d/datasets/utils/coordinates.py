# Copyright (c) Gorilla-Lab. All rights reserved.
from typing import List

import numpy as np
import numba as nb

import gorilla


class GridProcesses(object):
    def __init__(self,
                 num_class: int = 20,
                 grid_size: List[int] = [480, 360, 32],
                 fixed_volume_space: bool = False,
                 min_volume_space: List[float] = [50, -np.pi, -4],
                 max_volume_space: List[float] = [50, np.pi, 2],
                 ignore_label: int = 255,
                 **kwargs):
        super().__init__()
        self.num_class = num_class
        self.grid_size = np.asarray(grid_size)
        self.fixed_volume_space = fixed_volume_space
        self.min_volume_space = min_volume_space
        self.max_volume_space = max_volume_space
        self.ignore_label = ignore_label

    def __call__(self, xyz: np.ndarray, labels: np.ndarray):
        xyz = xyz[:, :3]  # [N, 3]

        max_bound = np.percentile(xyz, 100, axis=0)
        min_bound = np.percentile(xyz, 0, axis=0)

        if self.fixed_volume_space:
            min_bound = np.asarray(self.min_volume_space)
            max_bound = np.asarray(self.max_volume_space)

        # get grid index
        crop_range = max_bound - min_bound  # [3]
        intervals = crop_range / (self.grid_size - 1
                                  )  # [3] equal to voxel size
        if (intervals == 0).any():
            print("Zero interval!")

        grid_ind = (np.floor(
            (np.clip(xyz, min_bound, max_bound) - min_bound) /
            intervals)).astype(np.int)  # [N, 3] equal to voxel indices

        # initialize voxel and get the coordinate of each grid in Cartesian coordinates
        voxel_position = np.zeros(self.grid_size, dtype=np.float32)  # [3]
        dim_array = np.ones(len(self.grid_size) + 1, int)  # [4]
        dim_array[0] = -1
        voxel_position = np.indices(
            self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(
                dim_array)  # [3, H, W, L]

        # process labels
        processed_label = np.ones(self.grid_size,
                                  dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort(
            (grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label),
                                           label_voxel_pair)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) +
                         0.5) * intervals + min_bound  # [N, 3]
        return_xyz = xyz - voxel_centers  # [N, 3] realate coordinate for points in their voxels
        return_xyz = np.concatenate((return_xyz, xyz, xyz[:, :2]),
                                    axis=1)  # [N, 8]

        return voxel_position, processed_label, grid_ind, labels, return_xyz


class PolarProcesses(object):
    def __init__(self,
                 num_class: int = 20,
                 grid_size: List[int] = [480, 360, 32],
                 fixed_volume_space: bool = False,
                 min_volume_space: List[float] = [50, -np.pi, -4],
                 max_volume_space: List[float] = [50, np.pi, 2],
                 ignore_label: int = 255,
                 use_voxel_center: bool = False,
                 **kwargs):
        super().__init__()
        self.num_class = num_class
        self.grid_size = np.asarray(grid_size)
        self.fixed_volume_space = fixed_volume_space
        self.min_volume_space = min_volume_space
        self.max_volume_space = max_volume_space
        self.ignore_label = ignore_label
        self.use_voxel_center = use_voxel_center

    def __call__(self, xyz: np.ndarray, labels: np.ndarray):
        # convert the coordinate from Cartesian coordinates to polar coordinates
        xyz_pol = self.cart2polar(xyz)  # [N, 3]

        # get the bound of xyz in polar system
        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
        max_bound = np.max(xyz_pol[:, 1:], axis=0)  # [2]
        min_bound = np.min(xyz_pol[:, 1:], axis=0)  # [2]
        max_bound = np.concatenate(([max_bound_r], max_bound))  # [3]
        min_bound = np.concatenate(([min_bound_r], min_bound))  # [3]
        if self.fixed_volume_space:
            min_bound = np.asarray(self.min_volume_space)
            max_bound = np.asarray(self.max_volume_space)
        # get grid index
        crop_range = max_bound - min_bound  # [3]
        intervals = crop_range / (self.grid_size - 1
                                  )  # [3] equal to voxel size
        if (intervals == 0).any():
            print("Zero interval!")
        grid_ind = (np.floor(
            (np.clip(xyz_pol, min_bound, max_bound) - min_bound) /
            intervals)).astype(np.int)  # [N, 3] equal to voxel indices

        # initialize voxel and get the coordinate of each grid in Cartesian coordinates
        voxel_position = np.zeros(self.grid_size, dtype=np.float32)  # [3]
        dim_array = np.ones(len(self.grid_size) + 1, int)  # [4]
        dim_array[0] = -1
        voxel_position = np.indices(
            self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(
                dim_array)  # [3, H, W, L]
        # NOTE: the polar2cat operation consumes around 0.2~0.3s for batch=2 in my traing machine
        #       and this value maybe not used, so we close it as default
        if self.use_voxel_center:
            voxel_position = self.polar2cat(voxel_position)  # [3, H, W, L]

        # process labels
        processed_label = np.ones(self.grid_size,
                                  dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort(
            (grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label),
                                           label_voxel_pair)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) +
                         0.5) * intervals + min_bound  # [N, 3]
        return_xyz = xyz_pol - voxel_centers  # [N, 3] realate coordinate for points in their voxels
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]),
                                    axis=1)  # [N, 8]

        return voxel_position, processed_label, grid_ind, labels, return_xyz

    # core transformation of cylindrical
    # transformation between Cartesian coordinates and polar coordinates
    @staticmethod
    def cart2polar(input_xyz: np.ndarray):
        rho = np.sqrt(input_xyz[:, 0]**2 + input_xyz[:, 1]**2)  # [N]
        phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])  # [N]
        return np.stack((rho, phi, input_xyz[:, 2]), axis=1)  # [N, 3]

    # transformation between polar coordinates and Cartesian coordinates
    @staticmethod
    def polar2cat(input_xyz_polar: np.ndarray):
        x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])  # [H, W, L]
        y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])  # [H, W, L]
        return np.stack((x, y, input_xyz_polar[2]), axis=0)  # [3, H, W, L]


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])',
        nopython=True,
        cache=True,
        parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size, ), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1],
                            cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size, ), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1],
                    cur_sear_ind[2]] = np.argmax(counter)
    return processed_label
