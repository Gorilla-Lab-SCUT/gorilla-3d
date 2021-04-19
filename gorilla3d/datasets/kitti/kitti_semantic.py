# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import glob
from typing import Dict, List

import gorilla
import numpy as np
import numba as nb
import torch
from torch.utils.data import Dataset

# modify from https://github.com/xinge008/Cylinder3D/blob/master/dataloader/dataset_semantickitti.py
class KittiSem(Dataset):
    def __init__(self,
                 data_root: str,
                 task: str="train",
                 label_mapping: str="semantic-kitti.yaml",
                 return_ref: bool=False,
                 transform_cfg: Dict=dict(
                    rotate_aug=False,
                    flip_aug=False,
                    scale_aug=False,
                    transform=False),
                 grid_cfg: Dict=dict(
                    type="PolarProcesses", # "PolarProcesses" or "GridProcesses"
                    num_class=20,
                    grid_size=[480, 360, 32],
                    fixed_volume_space=False,
                    min_volume_space=[0, -np.pi, -4],
                    max_volume_space=[50, np.pi, 2],
                 ),
                 preload_labels: bool=True,
                 **kwargs):
        self.return_ref = return_ref
        self.semkittiyaml = gorilla.load(label_mapping)
        self.learning_map = self.semkittiyaml["learning_map"]
        assert task in ["train", "val", "test"], f"`task` must be in ['train', 'val', 'test'], but got {task}"
        split = self.semkittiyaml["split"][task]
        self.task = task

        self.data_files = []
        for i_folder in split:
            # TODO: fix it
            i_folder = f"{i_folder:0>2}"
            self.data_files += glob.glob(os.path.join(data_root, f"{i_folder:0>2}", "velodyne", "*"))

        self.pc_transformer = PointCloudTransfromer(**transform_cfg)
        processer_caller = globals()[grid_cfg["type"]]
        self.processer = processer_caller(**grid_cfg)

        # load files
        self.preload_labels = preload_labels
        if self.preload_labels:
            self.load_files()

    def load_files(self):
        """
        load all labels to speed up training process
        """
        label_mapper = np.vectorize(self.learning_map.__getitem__)
        if self.task == "test": return
        self.semantic_label_list = []
        self.instance_label_list = []
        print(f"prepare label files: {len(self.data_files)}")
        for data_file in gorilla.track(self.data_files):
            annotated_data = np.fromfile(data_file.replace("velodyne", "labels").replace(".bin", ".label"),
                                         dtype=np.int32).reshape((-1, 1)) # [N, 1]
            semantic_label = annotated_data & 0xFFFF  # semantic label in lower half
            instance_label = annotated_data >> 16     # instance id in upper half
            semantic_label = label_mapper(semantic_label) # annotated id map
            self.semantic_label_list.append(semantic_label)
            self.instance_label_list.append(instance_label)

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.data_files)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.data_files[index], dtype=np.float32).reshape((-1, 4)) # [N, 4]
        if self.task == "test":
            semantic_label = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
            instance_label = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        elif not self.preload_labels:
            label_mapper = np.vectorize(self.learning_map.__getitem__)
            annotated_data = np.fromfile(self.data_files[index].replace("velodyne", "labels").replace(".bin", ".label"),
                                         dtype=np.int32).reshape((-1, 1)) # [N, 1]
            semantic_label = annotated_data & 0xFFFF  # semantic label in lower half
            instance_label = annotated_data >> 16     # instance id in upper half
            semantic_label = label_mapper(semantic_label) # annotated id map
        else:
            semantic_label = self.semantic_label_list[index]
            instance_label = self.instance_label_list[index]

        # from ipdb import set_trace; set_trace()
        # range_proj = self.range_projection(raw_data, annotated_data)
        # np.save("temp.npy", range_proj)

        if self.task == "train":
            raw_data = self.pc_transformer(raw_data)
        voxel_position, processed_label, processed_count, grid_ind, labels, processed_xyz = self.processer(raw_data, annotated_data)
        xyz = raw_data[:, :3]
        if self.return_ref:
            sig = raw_data[:, 3]
            if len(sig.shape) == 2:
                sig = np.squeeze(sig)
            return_fea = np.concatenate((processed_xyz, sig[..., np.newaxis]), axis=1)

        data_tuple = (voxel_position, processed_label, processed_count, grid_ind, labels, xyz, return_fea)
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
        voxel_label_counts = []
        grid_inds = []
        point_labels = []
        point_xyzs = []
        point_features = []
        for b in batch:
            voxel_centers.append(b[0].astype(np.float32))
            voxel_labels.append(b[1].astype(np.int))
            voxel_label_counts.append(b[2].astype(np.int))
            grid_inds.append(b[3])
            point_labels.append(b[4])
            point_xyzs.append(b[5])
            point_features.append(b[6])
        
        voxel_centers = torch.from_numpy(np.stack(voxel_centers))
        voxel_labels = torch.from_numpy(np.stack(voxel_labels))
        voxel_label_counts = torch.from_numpy(np.stack(voxel_label_counts))

        return {
            "voxel_centers": voxel_centers,
            "voxel_labels": voxel_labels,
            "voxel_label_counts": voxel_label_counts,
            "grid_inds": grid_inds,
            "point_labels": point_labels,
            "point_xyzs": point_xyzs,
            "point_features": point_features
        }
        # return voxel_centers, voxel_labels, voxel_label_counts, grid_inds, point_labels, point_xyzs, point_features

    # modify from https://github.com/PRBonn/semantic-kitti-api/blob/master/auxiliary/laserscan.py
    @staticmethod
    def range_projection(scans: np.ndarray,
                         labels: np.ndarray,
                         H: int=64,
                         W: int=1024,
                         fov_up: float=3.0,
                         fov_down: float=25.0):
        # read scan data
        points = scans[:, :3] # [N, 3]
        remission = scans[:, 3] # [N, 3]
        
        # laser parameters
        fov_up = fov_up / 180.0 * np.pi     # field of view up in rad
        fov_down = fov_down / 180.0 * np.pi # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(points, 2, axis=1)

        # get scan components
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0) # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= W # in [0.0, W]
        proj_y *= H # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]

        # order in decreasing depth
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        points = points[order]
        remission = remission[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # projected range image - [H, W, 6] range (-1 is no data) 5: [x, y, z, depth, remission, label]
        proj_range = np.full((H, W, 6), -1, dtype=np.float32)
        proj_range[proj_y, proj_x, :3] = points
        proj_range[proj_y, proj_x, 3] = depth
        proj_range[proj_y, proj_x, 4] = remission
        proj_range[proj_y, proj_x, 4] = labels
        return proj_range


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

        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label # [H, W, L]
        processed_count = np.zeros(np.append(self.grid_size, self.num_class), dtype=np.uint8) # [H, W, L, num_class]
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1) # [N, 4]
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :] # [N, 4] sort the voxel label pair

        processed_count = nb_process_count(np.copy(processed_count), label_voxel_pair) # [H, W, L, num_class]
        non_empty_ids = (processed_count.max(-1) > 0) # [H, W, L]
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label # [H, W, L]
        processed_label[non_empty_ids] = np.argmax(processed_count[non_empty_ids, :], axis=-1) # [H, W, L]

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound # [N, 3]
        return_xyz = xyz - voxel_centers # [N, 3] realate coordinate for points in their voxels
        return_xyz = np.concatenate((return_xyz, xyz, xyz[:, :2]), axis=1) # [N, 9]

        return voxel_position, processed_label, processed_count, grid_ind, labels, return_xyz


class PolarProcesses(object):
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
        voxel_position = self.polar2cat(voxel_position) # [3, H, W, L]

        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label # [H, W, L]
        processed_count = np.zeros(np.append(self.grid_size, self.num_class), dtype=np.uint8) # [H, W, L, num_class]
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1) # [N, 4]
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :] # [N, 4] sort the voxel label pair

        processed_count = nb_process_count(np.copy(processed_count), label_voxel_pair) # [H, W, L, num_class]
        non_empty_ids = (processed_count.max(-1) > 0) # [H, W, L]
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label # [H, W, L]
        processed_label[non_empty_ids] = np.argmax(processed_count[non_empty_ids, :], axis=-1) # [H, W, L]

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound # [N, 3]
        return_xyz = xyz_pol - voxel_centers # [N, 3] realate coordinate for points in their voxels
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1) # [N, 9]

        return voxel_position, processed_label, processed_count, grid_ind, labels, return_xyz

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
        # print(input_xyz_polar.shape)
        x = input_xyz_polar[0] * np.cos(input_xyz_polar[1]) # [H, W, L]
        y = input_xyz_polar[0] * np.sin(input_xyz_polar[1]) # [H, W, L]
        return np.stack((x, y, input_xyz_polar[2]), axis=0) # [3, H, W, L]


@nb.jit("u1[:,:,:,:](u1[:,:,:,:],i8[:,:])", nopython=True, cache=True, parallel=False)
def nb_process_count(processed_count: np.ndarray,
                        sorted_label_voxel_pair: np.ndarray):
    label_size = processed_count.shape[-1]
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    # TODO: read it
    # traverse label pair
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3] # get the voxel indice
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            # `cur_ind != cur_sear_ind`, means we have traversed all points in `cur_sear_ind` voxle
            # find the label with the most points within the voxel
            # clear the counter and reset the cur_sear_ind
            processed_count[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2], :] = counter
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        # count the the label
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_count[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2], :] = counter
    return processed_count

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
