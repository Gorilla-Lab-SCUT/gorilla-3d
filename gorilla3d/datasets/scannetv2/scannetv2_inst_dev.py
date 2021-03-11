# Copyright (c) Gorilla-Lab. All rights reserved.
import math
import json
import glob
import os.path as osp
from abc import ABCMeta, abstractmethod
from typing import List

import gorilla
import numpy as np
import torch
from torch.utils.data import (Dataset, DataLoader)

from ...utils import elastic, pc_aug

try:
    import pointgroup_ops
except:
    pass

class ScanNetV2Inst(Dataset):
    def __init__(self,
                 data_root,
                 full_scale: List[int]=[128, 512],
                 scale: float=50.,
                 max_npoint: int=250000,
                 task: str="train",
                 with_elastic: bool=False,
                 test_mode: bool=False,
                 **kwargs):
        
        # initialize dataset parameters
        self.data_root = data_root
        self.full_scale = full_scale
        self.scale = scale
        self.max_npoint = max_npoint
        self.task = task
        self.with_elastic = with_elastic
        self.test_mode = test_mode
        # load files
        self.load_files()
    
    def load_files(self):
        file_names = sorted(glob.glob(osp.join(self.data_root, self.task, "*.pth")))
        self.files = [torch.load(i) for i in gorilla.track(file_names)]
        print(f"{self.task} samples: {len(self.files)}")
        # load superpoint
        self.superpoints = []
        sub_dir = "scans_test" if "test" in self.task else "scans"
        for file in gorilla.track(self.files):
            scene = file[-1]
            with open(osp.join(self.data_root, sub_dir, scene, scene+"_vh_clean_2.0.010000.segs.json"), "r") as f:
                superpoint = json.load(f)
            self.superpoints.append(np.array(superpoint["segIndices"]))

    def __getitem__(self, index):
        aug_flag = "train" in self.task
        if "test" in self.task:
            xyz_origin, rgb, faces, scene = self.files[index]
            # construct fake label for label-lack testset
            semantic_label = np.zeros(xyz_origin.shape[0], dtype=np.int32)
            instance_label = np.zeros(xyz_origin.shape[0], dtype=np.int32)
        else:
            xyz_origin, rgb, faces, semantic_label, instance_label, coords_shift, scene = self.files[index]

        # read superpoint
        superpoint = self.superpoints[index]

        ### jitter / flip x / rotation
        if aug_flag:
            xyz_middle = pc_aug(xyz_origin, True, True, True)
        else:
            xyz_middle = pc_aug(xyz_origin, False, False, False)

        ### scale
        xyz = xyz_middle * self.scale

        ### elastic distortion
        if self.with_elastic:
            xyz = elastic(xyz, 6 * self.scale // 50, 40 * self.scale / 50)
            xyz = elastic(xyz, 20 * self.scale // 50, 160 * self.scale / 50)

        ### offset
        xyz_offset = xyz.min(0)
        xyz -= xyz_offset

        valid_idxs = range(len(xyz_middle))
        ### crop
        if not self.test_mode:
            xyz, valid_idxs = self.crop(xyz)

        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]

        ### get instance labels
        semantic_label = semantic_label[valid_idxs]
        instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)
        inst_num, inst_infos = self.get_instance_info(xyz_middle, instance_label.astype(np.int32))
        inst_info = inst_infos["instance_info"]  # [n, 9], (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        inst_pointnum = inst_infos["instance_pointnum"]   # [num_inst], list
        
        # input 
        loc = torch.from_numpy(xyz).long()
        loc_offset = torch.from_numpy(xyz_offset).long()
        loc_float = torch.from_numpy(xyz_middle)
        feat = torch.from_numpy(rgb)
        superpoint = torch.from_numpy(superpoint)
        # labels
        semantic_label = torch.from_numpy(semantic_label)
        instance_label = torch.from_numpy(instance_label)
        inst_info = torch.from_numpy(inst_info)

        # training noise
        if aug_flag:
            feat += torch.randn(3) * 0.1

        return scene, loc, loc_offset, loc_float, feat, semantic_label, instance_label, superpoint, inst_num, inst_info, inst_pointnum

    def __len__(self):
        return len(self.files)

    def crop(self, xyz):
        """
        :param xyz: (n, 3) >= 0
        """
        xyz_offset = xyz.copy()
        valid_idxs = (xyz_offset.min(1) >= 0)
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.full_scale[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        # do not crop for testset
        while (valid_idxs.sum() > self.max_npoint):
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs


    def get_instance_info(self, xyz, instance_label):
        """
        :param xyz: (n, 3)
        :param instance_label: (n), int, (0~num_inst-1, -100)
        :return: instance_num, dict
        """
        instance_info = np.ones((xyz.shape[0], 9), dtype=np.float32) * -100.0   # [n, 9], float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_pointnum = []   # [num_inst], int
        instance_num = int(instance_label.max()) + 1
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)

            ### instance_info
            xyz_i = xyz[inst_idx_i]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            instance_info_i = instance_info[inst_idx_i]
            instance_info_i[:, 0:3] = mean_xyz_i
            instance_info_i[:, 3:6] = min_xyz_i
            instance_info_i[:, 6:9] = max_xyz_i
            instance_info[inst_idx_i] = instance_info_i

            ### instance_pointnum
            instance_pointnum.append(inst_idx_i[0].size)

        return instance_num, {"instance_info": instance_info, "instance_pointnum": instance_pointnum}


    def get_cropped_inst_label(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    
    def collate_fn(self, batch):
        locs = []
        loc_offset_list = []
        locs_float = []
        feats = []
        semantic_labels = []
        instance_labels = []

        instance_infos = []  # [N, 9]
        instance_pointnum = []  # [total_num_inst], int

        batch_offsets = [0]
        scene_list = []
        superpoint_list = []
        superpoint_bias = 0

        total_inst_num = 0
        for i, data in enumerate(batch):
            scene, loc, loc_offset, loc_float, feat, semantic_label, instance_label, superpoint, inst_num, inst_info, inst_pointnum = data
            
            scene_list.append(scene)
            superpoint += superpoint_bias
            superpoint_bias += (superpoint.max() + 1)

            invalid_ids = np.where(instance_label != -100)
            instance_label[invalid_ids] += total_inst_num
            total_inst_num += inst_num

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + loc.shape[0])

            locs.append(torch.cat([torch.LongTensor(loc.shape[0], 1).fill_(i), loc], 1))
            loc_offset_list.append(loc_offset)
            locs_float.append(loc_float)
            feats.append(feat)
            semantic_labels.append(semantic_label)
            instance_labels.append(instance_label)
            superpoint_list.append(superpoint)

            instance_infos.append(inst_info)
            instance_pointnum.extend(inst_pointnum)

        ### merge all the scenes in the batchd
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int [B+1]

        locs = torch.cat(locs, 0)                                # long [N, 1 + 3], the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float [N, 3]
        superpoint = torch.cat(superpoint_list, 0).long()               # long[N]
        feats = torch.cat(feats, 0)                              # float [N, C]
        semantic_labels = torch.cat(semantic_labels, 0).long()                     # long [N]
        instance_labels = torch.cat(instance_labels, 0).long()   # long [N]
        locs_offset = torch.stack(loc_offset_list)               # long [B, 3]

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)       # float [N, 9] (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int [total_num_inst]

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None) # long [3]

        ### voxelize
        batch_size = len(batch)
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, batch_size, 4)

        return {"locs": locs, "locs_offset": locs_offset, "voxel_locs": voxel_locs,
                "scene_list": scene_list, "p2v_map": p2v_map, "v2p_map": v2p_map,
                "locs_float": locs_float, "feats": feats,
                "semantic_labels": semantic_labels, "instance_labels": instance_labels,
                "instance_info": instance_infos, "instance_pointnum": instance_pointnum,
                "offsets": batch_offsets, "spatial_shape": spatial_shape, "superpoint": superpoint}

