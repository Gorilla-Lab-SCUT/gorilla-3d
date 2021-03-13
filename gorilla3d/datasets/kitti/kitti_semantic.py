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
                 return_test: bool=False,
                 transform_cfg: Dict=dict(
                    rotate_aug=False,
                    flip_aug=False,
                    scale_aug=False,
                    transform=False),
                 grid_cfg: Dict=dict(
                    grid_size=[480, 360, 32],
                    fixed_volume_space=False,
                    min_volume_space=[0, -np.pi, -4],
                    max_volume_space=[50, np.pi, 2],
                 ),
                 preload_labels: bool=True):
        self.return_ref = return_ref
        self.return_test = return_test
        from ipdb import set_trace; set_trace()
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
        self.polar_processer = PolarProcesses(**grid_cfg)

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
        self.annotated_data_list = []
        print(f"prepare label files: {len(self.data_files)}")
        for data_file in gorilla.track(self.data_files):
            annotated_data = np.fromfile(data_file.replace("velodyne", "labels").replace(".bin", ".label"),
                                         dtype=np.int32).reshape((-1, 1)) # [N, 1]
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = label_mapper(annotated_data) # annotated id map
            self.annotated_data_list.append(annotated_data)

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.data_files)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.data_files[index], dtype=np.float32).reshape((-1, 4)) # [N, 4]
        if self.task == "test":
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        elif not self.preload_labels:
            label_mapper = np.vectorize(self.learning_map.__getitem__)
            annotated_data = np.fromfile(self.data_files[index].replace("velodyne", "labels").replace(".bin", ".label"),
                                         dtype=np.int32).reshape((-1, 1)) # [N, 1]
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = label_mapper(annotated_data) # annotated id map
        else:
            annotated_data = self.annotated_data_list[index]

        if self.task == "train":
            raw_data = self.pc_transformer(raw_data)
        voxel_position, processed_label, grid_ind, labels, processed_xyz = self.polar_processer(raw_data, annotated_data)
        if self.return_ref:
            sig = raw_data[:, 3]
            if len(sig.shape) == 2:
                sig = np.squeeze(sig)
            return_fea = np.concatenate((processed_xyz, sig[..., np.newaxis]), axis=1)

        data_tuple = (voxel_position, processed_label, grid_ind, labels, return_fea)
        if self.return_test:
            data_tuple += (index,)
        return data_tuple

    @property
    def label_name(self):
        SemKITTI_label_name = {}
        for i in sorted(list(self.learning_map.keys()))[::-1]:
            SemKITTI_label_name[self.learning_map[i]] = self.semkittiyaml["labels"][i]
        return SemKITTI_label_name

    @staticmethod
    def collate_fn(batch):
        datas = []
        labels = []
        grid_inds = []
        point_labels = []
        xyzs = []
        for b in batch:
            datas.append(b[0].astype(np.float32))
            labels.append(b[1].astype(np.int))
            grid_inds.append(b[2])
            point_labels.append(b[3])
            xyzs.append(b[4])
        
        datas = torch.from_numpy(np.stack(datas))
        labels = torch.from_numpy(np.stack(labels))

        return datas, labels, grid_inds, point_labels, xyzs


class PolarProcesses(object):
    def __init__(self,
                 grid_size: List[int]=[480, 360, 32],
                 fixed_volume_space: bool=False,
                 min_volume_space: List[float]=[50, -np.pi, -4],
                 max_volume_space: List[float]=[50, np.pi, 2],
                 ignore_label: int=255):
        super().__init__()
        self.grid_size = np.asarray(grid_size)
        self.fixed_volume_space = fixed_volume_space
        self.min_volume_space = min_volume_space
        self.max_volume_space = max_volume_space
        self.ignore_label =ignore_label
    
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
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1) # [N, 4]
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :] # [N, 4] sort the voxel label pair
        processed_label = self.nb_process_label(np.copy(processed_label), label_voxel_pair) # [H, W, L]

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound # [N, 3]
        return_xyz = xyz_pol - voxel_centers # [N, 3] realate coordinate for points in their voxels
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1) # [N, 9]

        return voxel_position, processed_label, grid_ind, labels, return_xyz
    
    @staticmethod
    @nb.jit("u1[:,:,:](u1[:,:,:],i8[:,:])", nopython=True, cache=True, parallel=False)
    def nb_process_label(processed_label: np.ndarray,
                         sorted_label_voxel_pair: np.ndarray):
        label_size = 256
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
                processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
                counter = np.zeros((label_size,), dtype=np.uint16)
                cur_sear_ind = cur_ind
            # count the the label
            counter[sorted_label_voxel_pair[i, 3]] += 1
        processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
        return processed_label

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
