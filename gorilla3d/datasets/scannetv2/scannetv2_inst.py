"""
ScanNet v2 Dataloader (Modified from SparseConvNet Dataloader)
Written by Li Jiang
"""

from abc import ABCMeta, abstractmethod
import json
import glob
import os.path as osp
import ipdb
from numpy.lib.financial import ipmt

import numpy as np
import torch
from torch.utils.data import (Dataset, DataLoader)

from ...utils import elastic, pc_aug

try:
    from pointgroup import pointgroup_ops
except:
    pass

class ScanNetV2Inst(Dataset, metaclass=ABCMeta):
    def __init__(self, cfg=None, logger=None):
        self.logger = logger

        # dataset parameters
        self.data_root = cfg.data.data_root
        self.dataset = "scannetv2"
        self.filename_suffix = cfg.data.filename_suffix
        self.batch_size = cfg.data.batch_size

        # voxelization parameters
        self.full_scale = cfg.data.full_scale
        self.scale = cfg.data.scale
        self.max_npoint = cfg.data.max_npoint
        self.mode = cfg.data.mode
        self.workers = cfg.data.workers
        # self.workers = 0 # for debug merge

        # special paramters
        self.train_mini = cfg.data.train_mini 
        self.task = cfg.task
        self.with_elastic = cfg.data.with_elastic
        
        # load files
        self.load_files()
    
    def load_files(self):
        file_names = sorted(glob.glob(osp.join(self.data_root, self.dataset, self.task, "*" + self.filename_suffix)))
        self.files = [torch.load(i) for i in file_names]

        self.logger.info("{} samples: {}".format(self.task, len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.getitem(index)
    
    @abstractmethod
    def getitem(self, index):
        pass

    @abstractmethod
    def merge(self):
        pass
    
    @property
    def dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, collate_fn=self.merge, num_workers=self.workers,
                          shuffle=True, sampler=None, drop_last=True, pin_memory=True)

    def crop(self, xyz):
        """
        :param xyz: (n, 3) >= 0
        """
        xyz_offset = xyz.copy()
        valid_idxs = (xyz_offset.min(1) >= 0)
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.full_scale[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while (valid_idxs.sum() > self.max_npoint):
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs


    def get_instance_info(self, xyz, instance_label):
        """
        :param xyz: (n, 3)
        :param instance_label: (n), int, (0~nInst-1, -100)
        :return: instance_num, dict
        """
        instance_info = np.ones((xyz.shape[0], 9), dtype=np.float32) * -100.0   # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_pointnum = []   # (nInst), int
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


class ScanNetV2InstTrainVal(ScanNetV2Inst):
    def __init__(self, cfg=None, logger=None, split="train"):
        ScanNetV2Inst.__init__(self, cfg, logger)
        self.split = split

    def getitem(self, index):
        xyz_origin, rgb, faces, label, instance_label, coords_shift, scene = self.files[index]

        # read overseg
        with open(osp.join(self.data_root, self.dataset, "scans", scene, scene+"_vh_clean_2.0.010000.segs.json"), "r") as f:
            overseg = json.load(f)
        overseg = np.array(overseg["segIndices"])

        ### jitter / flip x / rotation
        if self.split=="train":
            xyz_middle = pc_aug(xyz_origin, True, True, True)
        else:
            xyz_middle = pc_aug(xyz_origin, False, False, False)

        ### scale
        xyz = xyz_middle * self.scale

        ### elastic
        if self.with_elastic:
            xyz = elastic(xyz, 6 * self.scale // 50, 40 * self.scale / 50)
            xyz = elastic(xyz, 20 * self.scale // 50, 160 * self.scale / 50)

        ### offset
        xyz_offset = xyz.min(0)
        xyz -= xyz_offset

        ### crop
        xyz, valid_idxs = self.crop(xyz)

        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]
        label = label[valid_idxs]
        overseg = overseg[valid_idxs]
        _, overseg = np.unique(overseg, return_inverse=True)
        instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)

        ### get instance information
        inst_num, inst_infos = self.get_instance_info(xyz_middle, instance_label.astype(np.int32))
        inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        inst_pointnum = inst_infos["instance_pointnum"]   # (nInst), list
        
        loc = torch.from_numpy(xyz).long()
        loc_offset = torch.from_numpy(xyz_offset).long()
        loc_float = torch.from_numpy(xyz_middle)
        feat = torch.from_numpy(rgb)
        if self.mode == "train":
            feat += torch.randn(3) * 0.1
        label = torch.from_numpy(label)
        instance_label = torch.from_numpy(instance_label)
        overseg = torch.from_numpy(overseg)

        inst_info = torch.from_numpy(inst_info)

        return scene, loc, loc_offset, loc_float, feat, label, instance_label, overseg, inst_num, inst_info, inst_pointnum

    def merge(self, batch):
        locs = []
        loc_offset_list = []
        locs_float = []
        feats = []
        labels = []
        instance_labels = []

        instance_infos = []  # (N, 9)
        instance_pointnum = []  # (total_nInst), int

        batch_offsets = [0]
        scene_list = []
        overseg_list = []
        overseg_bias = 0

        total_inst_num = 0
        for i, data in enumerate(batch):
            scene, loc, loc_offset, loc_float, feat, label, instance_label, overseg, inst_num, inst_info, inst_pointnum = data
            
            scene_list.append(scene)
            overseg += overseg_bias
            overseg_bias += (overseg.max() + 1)

            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + loc.shape[0])

            locs.append(torch.cat([torch.LongTensor(loc.shape[0], 1).fill_(i), loc], 1))
            loc_offset_list.append(loc_offset)
            locs_float.append(loc_float)
            feats.append(feat)
            labels.append(label)
            instance_labels.append(instance_label)
            overseg_list.append(overseg)

            instance_infos.append(inst_info)
            instance_pointnum.extend(inst_pointnum)

        ### merge all the scenes in the batchd
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)                                # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        overseg = torch.cat(overseg_list, 0).long()               # long(N)
        feats = torch.cat(feats, 0)                              # float (N, C)
        labels = torch.cat(labels, 0).long()                     # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()   # long (N)
        locs_offset = torch.stack(loc_offset_list)               # long (B, 3)

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)       # float (N, 9) (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None) # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

        return {"locs": locs, "locs_offset": locs_offset, "voxel_locs": voxel_locs,
                "scene_list": scene_list, "p2v_map": p2v_map, "v2p_map": v2p_map,
                "locs_float": locs_float, "feats": feats, "labels": labels, "instance_labels": instance_labels,
                "instance_info": instance_infos, "instance_pointnum": instance_pointnum,
                "offsets": batch_offsets, "spatial_shape": spatial_shape, "overseg": overseg}


class ScanNetV2InstTest(ScanNetV2Inst):
    def __init__(self, cfg=None, logger=None):
        ScanNetV2Inst.__init__(self, cfg, logger)
        self.batch_size = 1
        self.workers = cfg.data.test_workers

    def getitem(self, index):
        if "val" in self.task or "train" in self.task:
            xyz_origin, rgb, faces, label, instance_label, coords_shift, scene = self.files[index]
            sub_dir = "scans"
        elif self.task == "test":
            xyz_origin, rgb, faces, scene = self.files[index]
            sub_dir = "scans_test"

        # read overseg
        with open(osp.join(self.data_root, self.dataset, sub_dir, scene, scene+"_vh_clean_2.0.010000.segs.json"), "r") as f:
            overseg = json.load(f)
        overseg = np.array(overseg["segIndices"])
        
        # read edge
        edges = np.concatenate([faces[:, :2], faces[:, 1:], faces[:, [0, 2]]]) # (nEdges, 2)
        overseg_edges = overseg[edges]
        overseg_edges = overseg_edges[overseg_edges[:, 0] != overseg_edges[:, 1]]

        xyz_middle = pc_aug(xyz_origin, False, False, False)

        ### scale
        xyz = xyz_middle * self.scale

        ### offset
        xyz_offset = xyz.min(0)
        xyz -= xyz_offset

        _, overseg = np.unique(overseg, return_inverse=True)

        if "val" in self.task or "train" in self.task:
            ### get instance information
            inst_num, inst_infos = self.get_instance_info(xyz_middle, instance_label.astype(np.int32))
            inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            inst_pointnum = inst_infos["instance_pointnum"]   # (nInst), list
        
        loc = torch.from_numpy(xyz).long()
        loc_offset = torch.from_numpy(xyz_offset).long()
        loc_float = torch.from_numpy(xyz_middle)
        feat = torch.from_numpy(rgb)
        overseg = torch.from_numpy(overseg)
        
        if "val" in self.task or "train" in self.task:
            label = torch.from_numpy(label)
            instance_label = torch.from_numpy(instance_label)
            inst_info = torch.from_numpy(inst_info)

        if "val" in self.task or "train" in self.task:
            return scene, loc, loc_offset, loc_float, feat, label, instance_label, overseg, inst_num, inst_info, inst_pointnum
        else:
            return scene, loc, loc_offset, loc_float, feat, overseg

    def merge(self, batch):
        locs = []
        loc_offset_list = []
        locs_float = []
        feats = []

        batch_offsets = [0]
        scene_list = []
        overseg_list = []
        overseg_bias = 0

        # with gt
        total_inst_num = 0
        labels = []
        instance_labels = []
        instance_infos = []  # (N, 9)
        instance_pointnum = []  # (total_nInst), int

        for i, data in enumerate(batch):
            if "val" in self.task or "train" in self.task:
                scene, loc, loc_offset, loc_float, feat, label, instance_label, overseg, inst_num, inst_info, inst_pointnum = data
            else:
                scene, loc, loc_offset, loc_float, feat, overseg = data
            
            scene_list.append(scene)
            overseg += overseg_bias
            overseg_bias += (overseg.max() + 1)

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + loc.shape[0])

            locs.append(torch.cat([torch.LongTensor(loc.shape[0], 1).fill_(i), loc], 1))
            loc_offset_list.append(loc_offset)
            locs_float.append(loc_float)
            feats.append(feat)
            overseg_list.append(overseg)

            if "val" in self.task or "train" in self.task:
                instance_label[np.where(instance_label != -100)] += total_inst_num
                total_inst_num += inst_num
                labels.append(label)
                instance_labels.append(instance_label)
                instance_infos.append(inst_info)
                instance_pointnum.extend(inst_pointnum)

        ### merge all the scenes in the batchd
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)                                # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        overseg = torch.cat(overseg_list, 0).long()               # long(N)
        feats = torch.cat(feats, 0)                              # float (N, C)
        labels = torch.cat(labels, 0).long()                     # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()   # long (N)
        locs_offset = torch.stack(loc_offset_list)               # long (B, 3)

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)       # float (N, 9) (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None) # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

        if "val" in self.task or "train" in self.task:
            return {"locs": locs, "locs_offset": locs_offset, "voxel_locs": voxel_locs,
                    "scene_list": scene_list, "p2v_map": p2v_map, "v2p_map": v2p_map,
                    "locs_float": locs_float, "feats": feats,
                    "instance_labels": instance_labels, "instance_info": instance_infos, "instance_pointnum": instance_pointnum,
                    "offsets": batch_offsets, "spatial_shape": spatial_shape,
                    "overseg": overseg}
        else:
            return {"locs": locs, "locs_offset": locs_offset, "voxel_locs": voxel_locs,
                    "scene_list": scene_list, "p2v_map": p2v_map, "v2p_map": v2p_map,
                    "locs_float": locs_float, "feats": feats,
                    "offsets": batch_offsets, "spatial_shape": spatial_shape,
                    "overseg": overseg}


