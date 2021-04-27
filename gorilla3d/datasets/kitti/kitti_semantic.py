# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import glob
from typing import Dict, List

import gorilla
import numpy as np
import numba as nb
import torch
from torch.utils.data import Dataset

from .utils import SemLaserScan

# modify from https://github.com/xinge008/Cylinder3D/blob/master/dataloader/dataset_semantickitti.py
class KittiSem(Dataset):
    def __init__(self,
                 data_root: str,
                 task: str="train",
                 label_mapping: str="semantic-kitti.yaml",
                 return_ref: bool=False,
                 return_test: bool=False,
                 transform_cfg: Dict=dict(
                    rotate_aug=False,
                    flip_aug=False,
                    scale_aug=False,
                    transform=False),
                 grid_cfg: Dict=dict(
                    type="PolarProcesses",
                    num_class=20,
                    grid_size=[480, 360, 32],
                    fixed_volume_space=False,
                    min_volume_space=[0, -np.pi, -4],
                    max_volume_space=[50, np.pi, 2],
                    use_voxel_center=False,
                 ),
                 **kwargs):
        self.logger = gorilla.derive_logger(__name__)
        self.return_ref = return_ref
        self.return_test = return_test
        self.semkittiyaml = gorilla.load(label_mapping)
        self.learning_map = self.semkittiyaml["learning_map"]
        self.label_mapper = np.vectorize(self.learning_map.__getitem__)
        assert task in ["train", "val", "test"], f"`task` must be in ['train', 'val', 'test'], but got {task}"
        self.sequences = self.semkittiyaml["split"][task]
        self.task = task

        self.data_files = []
        for i_folder in self.sequences:
            i_folder = f"{i_folder:0>2}"
            self.data_files += glob.glob(os.path.join(data_root, f"{i_folder:0>2}", "velodyne", "*"))

        self.data_files.sort()
        self.data_files = self.data_files

        self.logger.info("Using {} scans from sequences {}".format(len(self.data_files), self.sequences))

        self.pc_transformer = PointCloudTransfromer(**transform_cfg)
        processer_caller = globals()[grid_cfg["type"]]
        self.processer = processer_caller(**grid_cfg)

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.data_files)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.data_files[index], dtype=np.float32).reshape((-1, 4)) # [N, 4]
        if self.task == "test":
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.data_files[index].replace("velodyne", "labels").replace(".bin", ".label"),
                                         dtype=np.int32).reshape((-1, 1)) # [N, 1]
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = self.label_mapper(annotated_data) # annotated id map

        if self.task == "train":
            raw_data = self.pc_transformer(raw_data)
        voxel_position, processed_label, grid_ind, labels, processed_xyz = self.processer(raw_data, annotated_data)
        xyz = raw_data[:, :3]
        if self.return_ref:
            sig = raw_data[:, 3]
            if len(sig.shape) == 2:
                sig = np.squeeze(sig)
            return_fea = np.concatenate((processed_xyz, sig[..., np.newaxis]), axis=1)

        data_tuple = (voxel_position, processed_label, grid_ind, labels, xyz, return_fea)
        return data_tuple

    @property
    def label_name(self):
        SemKITTI_label_name = {}
        for i in sorted(list(self.learning_map.keys()))[::-1]:
            SemKITTI_label_name[self.learning_map[i]] = self.semkittiyaml["labels"][i]
        return SemKITTI_label_name

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
            grid_inds.append(torch.cat([torch.LongTensor(b[2].shape[0], 1).fill_(i), torch.from_numpy(b[2]).long()], 1))
            point_labels.append(torch.from_numpy(b[3]))
            point_xyzs.append(torch.cat([torch.FloatTensor(b[4].shape[0], 1).fill_(i), torch.from_numpy(b[4])], 1))
            point_features.append(torch.from_numpy(b[5]).float())
        
        voxel_centers = torch.stack(voxel_centers) # [B, H, W, D, 3]
        voxel_labels = torch.stack(voxel_labels) # [B, H, W, D]
        grid_inds = torch.cat(grid_inds, 0) # [N, 4]
        point_labels = torch.cat(point_labels, 0) # [N]
        point_xyzs = torch.cat(point_xyzs, 0) # [N, 3]
        point_features = torch.cat(point_features, 0) # [N, C]

        return {
            "voxel_centers": voxel_centers,
            "voxel_labels": voxel_labels,
            "grid_inds": grid_inds,
            "point_labels": point_labels,
            "point_xyzs": point_xyzs,
            "point_features": point_features
        }


class GridProcesses(object):
    def __init__(self,
                 num_class: int=20,
                 grid_size: List[int]=[480, 360, 32],
                 fixed_volume_space: bool=False,
                 min_volume_space: List[float]=[50, -np.pi, -4],
                 max_volume_space: List[float]=[50, np.pi, 2],
                 ignore_label: int=255,
                 **kwargs):
        super().__init__()
        self.num_class = num_class
        self.grid_size = np.asarray(grid_size)
        self.fixed_volume_space = fixed_volume_space
        self.min_volume_space = min_volume_space
        self.max_volume_space = max_volume_space
        self.ignore_label = ignore_label
        
    def __call__(self,
                 xyz: np.ndarray,
                 labels: np.ndarray):
        xyz = xyz[:, :3] # [N, 3]

        max_bound = np.percentile(xyz, 100, axis=0)
        min_bound = np.percentile(xyz, 0, axis=0)

        if self.fixed_volume_space:
            min_bound = np.asarray(self.min_volume_space)
            max_bound = np.asarray(self.max_volume_space)

        # get grid index
        crop_range = max_bound - min_bound # [3]
        intervals = crop_range / (self.grid_size - 1) # [3] equal to voxel size
        if (intervals == 0).any():
            print("Zero interval!")

        grid_ind = (np.floor((np.clip(xyz, min_bound, max_bound) - min_bound) / intervals)).astype(np.int) # [N, 3] equal to voxel indices

        # initialize voxel and get the coordinate of each grid in Cartesian coordinates
        voxel_position = np.zeros(self.grid_size, dtype=np.float32) # [3]
        dim_array = np.ones(len(self.grid_size) + 1, int) # [4]
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array) # [3, H, W, L]

        # process labels
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound # [N, 3]
        return_xyz = xyz - voxel_centers # [N, 3] realate coordinate for points in their voxels
        return_xyz = np.concatenate((return_xyz, xyz, xyz[:, :2]), axis=1) # [N, 8]

        return voxel_position, processed_label, grid_ind, labels, return_xyz


class PolarProcesses(object):
    def __init__(self,
                 num_class: int=20,
                 grid_size: List[int]=[480, 360, 32],
                 fixed_volume_space: bool=False,
                 min_volume_space: List[float]=[50, -np.pi, -4],
                 max_volume_space: List[float]=[50, np.pi, 2],
                 ignore_label: int=255,
                 use_voxel_center: bool=False,
                 **kwargs):
        super().__init__()
        self.num_class = num_class
        self.grid_size = np.asarray(grid_size)
        self.fixed_volume_space = fixed_volume_space
        self.min_volume_space = min_volume_space
        self.max_volume_space = max_volume_space
        self.ignore_label = ignore_label
        self.use_voxel_center = use_voxel_center
    
    def __call__(self,
                 xyz: np.ndarray,
                 labels: np.ndarray):
        # convert the coordinate from Cartesian coordinates to polar coordinates
        xyz_pol = self.cart2polar(xyz) # [N, 3]

        # get the bound of xyz in polar system
        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
        max_bound = np.max(xyz_pol[:, 1:], axis=0) # [2]
        min_bound = np.min(xyz_pol[:, 1:], axis=0) # [2]
        max_bound = np.concatenate(([max_bound_r], max_bound)) # [3]
        min_bound = np.concatenate(([min_bound_r], min_bound)) # [3]
        if self.fixed_volume_space:
            min_bound = np.asarray(self.min_volume_space)
            max_bound = np.asarray(self.max_volume_space)
        # get grid index
        crop_range = max_bound - min_bound # [3]
        intervals = crop_range / (self.grid_size - 1) # [3] equal to voxel size
        if (intervals == 0).any():
            print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int) # [N, 3] equal to voxel indices

        # initialize voxel and get the coordinate of each grid in Cartesian coordinates
        voxel_position = np.zeros(self.grid_size, dtype=np.float32) # [3]
        dim_array = np.ones(len(self.grid_size) + 1, int) # [4]
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array) # [3, H, W, L]
        # NOTE: the polar2cat operation consumes around 0.2~0.3s for batch=2 in my traing machine
        #       and this value maybe not used, so we close it as default
        if self.use_voxel_center:
            voxel_position = self.polar2cat(voxel_position) # [3, H, W, L]

        # process labels
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound # [N, 3]
        return_xyz = xyz_pol - voxel_centers # [N, 3] realate coordinate for points in their voxels
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1) # [N, 8]

        return voxel_position, processed_label, grid_ind, labels, return_xyz

    # core transformation of cylindrical
    # transformation between Cartesian coordinates and polar coordinates
    @staticmethod
    def cart2polar(input_xyz: np.ndarray):
        rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2) # [N]
        phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0]) # [N]
        return np.stack((rho, phi, input_xyz[:, 2]), axis=1) # [N, 3]

    # transformation between polar coordinates and Cartesian coordinates
    @staticmethod
    def polar2cat(input_xyz_polar: np.ndarray):
        x = input_xyz_polar[0] * np.cos(input_xyz_polar[1]) # [H, W, L]
        y = input_xyz_polar[0] * np.sin(input_xyz_polar[1]) # [H, W, L]
        return np.stack((x, y, input_xyz_polar[2]), axis=0) # [3, H, W, L]


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label

class PointCloudTransfromer(object):
    def __init__(self,
                 rotate_aug: bool=False,
                 flip_aug: bool=False,
                 scale_aug: bool=False,
                 transform: bool=False,
                 trans_std: List[float]=[0.1, 0.1, 0.1]):
        super().__init__()
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.transform = transform
        self.trans_std = trans_std

    def __call__(self, point: np.array):
        assert point.shape[1] >= 3, "the dimension of xyz must be larger than or equal to 3"
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
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        point[:, :3] = xyz

        return point


class KittiSemRV(KittiSem):
    def __init__(self,
                 data_root: str, # directory where data is
                 task: str="train",
                 label_mapping: str="semantic-kitti.yaml",
                 max_points: int=150000, # max number of points present in dataset
                 transform: bool=False,
                 sensor: Dict=dict(
                    project=True,
                    height=64,
                    width=2048,
                    fov_up=3,
                    fov_down=-25,
                 ), # sensor to parse scans from
                img_means=[12.12, 10.88, 0.23, -1.04, 0.21],
                img_stds=[12.32, 11.47, 6.91, -0.86, 0.16],
                 **kwargs): # send ground truth?
        # save deats
        super().__init__(data_root,
                         task,
                         label_mapping,
                         **kwargs)

        self.max_points = max_points
        self.transform = transform
        self.scan = SemLaserScan(**sensor)
        self.color_map = self.semkittiyaml["color_map"]
        self.sensor_img_means = np.array(img_means, dtype=np.float32)
        self.sensor_img_stds = np.array(img_stds, dtype=np.float32)

        # self.data_files = self.data_files[:100]

    def collate_fn(self, batch):
        return torch.utils.data._utils.collate.default_collate(batch)

    def __getitem__(self, index):
        gt_flag = self.task != "test"
        # get item in tensor shape
        scan_file = self.data_files[index]
        raw_data = np.fromfile(self.data_files[index], dtype=np.float32).reshape((-1, 4)) # [N, 4]
        if gt_flag:
            label_file = scan_file.replace("velodyne", "labels").replace(".bin", ".label")
            annotated_data = np.fromfile(label_file, dtype=np.int32) # [N]
        else:
            # construct the fake labels
            annotated_data = np.zeros_like(raw_data[:, 0]) # [N, 1]

        # point cloud data augment
        if self.task == "train":
            raw_data = self.pc_transformer(raw_data)

        # reset scan
        self.scan.reset()

        # read points and labels
        self.scan.set_points(raw_data[:, :3], raw_data[:, 3])
        self.scan.set_label(annotated_data)
        self.scan.sem_label_map(self.label_mapper) # map labels

        # get un-projected paramters(origin input)
        npoints = self.scan.points.shape[0]
        num_pad = self.max_points - npoints
        unproj_xyz = np.pad(self.scan.points, ((0, num_pad), (0, 0)), constant_values=(-1.0, -1.0)) # [max_points, 3]
        unproj_range = np.pad(self.scan.unproj_range, (0, num_pad), constant_values=(-1.0, -1.0)) # [max_points]
        unproj_remissions = np.pad(self.scan.remissions, (0, num_pad), constant_values=(-1.0, -1.0)) # [max_points]
        unproj_labels = np.pad(self.scan.sem_label, (0, num_pad), constant_values=(-1, -1)).astype(np.int32) # [max_points]

        # get projected paramters
        proj_range = self.scan.proj_range.copy() # [H, W]
        proj_xyz = self.scan.proj_xyz.copy() # [H, W, 3]
        proj_remission = self.scan.proj_remission.copy() # [H, W]
        proj_mask = self.scan.proj_mask.copy() # [H, W]
        proj_labels = self.scan.proj_sem_label.copy() # [H, W]
        proj_labels = proj_labels * proj_mask # [H, W]
        # model input
        proj = np.concatenate([proj_range[None, :, :].copy(),
                               proj_xyz.transpose(2, 0, 1).copy(),
                               proj_remission[None, :, :].copy()]) # [5, H, W]
        proj = (proj - self.sensor_img_means[:, None, None]) / self.sensor_img_stds[:, None, None] # [5, H, W]
        proj = proj * proj_mask.astype(np.float32)

        # get the projection x and y ids(in pixel coordinates)
        proj_x = np.pad(self.scan.proj_x, (0, num_pad), constant_values=(-1.0, -1.0)).astype(np.int64) # [max_points]
        proj_y = np.pad(self.scan.proj_y, (0, num_pad), constant_values=(-1.0, -1.0)).astype(np.int64) # [max_points]

        # get name and sequence
        path_norm = os.path.normpath(scan_file)
        path_split = path_norm.split(os.sep)
        path_name = path_split[-1].replace(".bin", ".label")

        # return 
        return {
            "proj": proj,
            "proj_mask": proj_mask,
            "proj_labels": proj_labels,
            "unproj_labels": unproj_labels,
            "path_name": path_name,
            "proj_x": proj_x,
            "proj_y": proj_y,
            "proj_range": proj_range,
            "unproj_range": unproj_range,
            "proj_xyz": proj_xyz,
            "unproj_xyz": unproj_xyz,
            "proj_remission": proj_remission,
            "unproj_remissions": unproj_remissions,
            "npoints": npoints
        }

