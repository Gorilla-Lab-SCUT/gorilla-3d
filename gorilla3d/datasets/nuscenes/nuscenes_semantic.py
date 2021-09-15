# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import glob
from typing import Dict, List

import gorilla
import numpy as np
import numba as nb
import torch
from torch.utils.data import Dataset

from ..utils import PointCloudTransfromer, PolarProcesses, GridProcesses


# modify from https://github.com/xinge008/Cylinder3D/blob/master/dataloader/dataset_semantickitti.py
class NuscenesSem(Dataset):
    def __init__(self,
                 data_root: str,
                 info_file: str,
                 task: str = "train",
                 label_mapping: str = "nuscenes.yaml",
                 return_ref: bool = False,
                 transform_cfg: Dict = dict(rotate_aug=False,
                                            flip_aug=False,
                                            scale_aug=False,
                                            transform=False),
                 grid_cfg: Dict = dict(
                     type="PolarProcesses",
                     num_class=20,
                     grid_size=[480, 360, 32],
                     fixed_volume_space=False,
                     min_volume_space=[0, -np.pi, -4],
                     max_volume_space=[50, np.pi, 2],
                     use_voxel_center=False,
                 ),
                 **kwargs):
        self.data_root = data_root
        self.logger = gorilla.derive_logger(__name__)
        self.return_ref = return_ref
        self.data_infos = gorilla.load(info_file)

        self.nuscenesyaml = gorilla.load(label_mapping)
        self.learning_map = self.nuscenesyaml["learning_map"]
        self.label_mapper = np.vectorize(self.learning_map.__getitem__)
        self.task = task

        self.logger.info("Using {} scans for {}".format(
            len(self.data_infos), self.task))

        self.pc_transformer = PointCloudTransfromer(**transform_cfg)
        processer_caller = globals()[grid_cfg["type"]]
        self.processer = processer_caller(**grid_cfg)

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.data_infos)

    def __getitem__(self, index):
        info = self.data_infos[index]
        lidar_path = info["lidar_path"][16:]
        lidar_sd_token = info["token"]["data"]["LIDAR_TOP"]
        lidarseg_labels_filename = os.path.join(self.data_root,
                                                lidar_sd_token["filename"])

        point_labels = np.fromfile(lidarseg_labels_filename,
                                   dtype=np.uint8).reshape([-1, 1])  # [N, 1]
        point_labels = self.label_mapper(point_labels)  # [N, 1]
        points = np.fromfile(os.path.join(self.data_root, lidar_path),
                             dtype=np.float32,
                             count=-1).reshape([-1, 5])  # [N, 5]

        if self.task == "train":
            points[:, :3] = self.pc_transformer(points[:, :3])

        voxel_position, processed_label, grid_ind, labels, processed_xyz = self.processer(
            points, point_labels)
        xyz = points[:, :3]
        if self.return_ref:
            sig = points[:, 3]
            if len(sig.shape) == 2:
                sig = np.squeeze(sig)
            return_fea = np.concatenate((processed_xyz, sig[..., np.newaxis]),
                                        axis=1)

        data_tuple = (voxel_position, processed_label, grid_ind, labels, xyz,
                      return_fea)
        return data_tuple

    @property
    def label_name(self):
        nuscenes_label_name = {}
        for i in sorted(list(self.learning_map.keys()))[::-1]:
            nuscenes_label_name[
                self.learning_map[i]] = self.nuscenesyaml["labels_16"][i]
        return nuscenes_label_name

    @staticmethod
    def collate_fn(batch):
        voxel_centers = []
        voxel_labels = []
        grid_inds = []
        point_labels = []
        point_xyzs = []
        point_features = []
        for i, b in enumerate(batch):
            voxel_centers.append(torch.from_numpy(b[0]).float())
            voxel_labels.append(torch.from_numpy(b[1]).long())
            grid_inds.append(
                torch.cat([
                    torch.LongTensor(b[2].shape[0], 1).fill_(i),
                    torch.from_numpy(b[2]).long()
                ], 1))
            point_labels.append(torch.from_numpy(b[3]))
            point_xyzs.append(
                torch.cat([
                    torch.FloatTensor(b[4].shape[0], 1).fill_(i),
                    torch.from_numpy(b[4])
                ], 1))
            point_features.append(torch.from_numpy(b[5]).float())

        voxel_centers = torch.stack(voxel_centers)  # [B, H, W, D, 3]
        voxel_labels = torch.stack(voxel_labels)  # [B, H, W, D]
        grid_inds = torch.cat(grid_inds, 0)  # [N, 4]
        point_labels = torch.cat(point_labels, 0)  # [N]
        point_xyzs = torch.cat(point_xyzs, 0)  # [N, 3]
        point_features = torch.cat(point_features, 0)  # [N, C]

        return {
            "voxel_centers": voxel_centers,
            "voxel_labels": voxel_labels,
            "grid_inds": grid_inds,
            "point_labels": point_labels,
            "point_xyzs": point_xyzs,
            "point_features": point_features
        }


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
