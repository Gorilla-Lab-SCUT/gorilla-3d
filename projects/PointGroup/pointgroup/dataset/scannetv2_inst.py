"""
ScanNet v2 Dataloader (Modified from SparseConvNet Dataloader)
Written by Li Jiang
"""

import os
import os.path as osp
import sys
import json
import glob
import math
import pickle

import trimesh
import numpy as np
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
import torch
from torch.utils.data import DataLoader

sys.path.append("../")

from pointgroup import pointgroup_ops
# from lib.pointgroup_ops.functions import pointgroup_ops


class Dataset:
    def __init__(self, cfg=None, logger=None, test=False):
        self.logger = logger
        self.data_root = cfg.data_root
        self.dataset = cfg.dataset
        try:
            self.pose_file = cfg.pose_file
        except:
            self.pose_file = "scene_align.pkl"
        with open(osp.join(self.data_root, self.dataset, self.pose_file), "rb") as f:
            self.all_pose_dict = pickle.load(f)
        self.filename_suffix = cfg.filename_suffix

        self.batch_size = cfg.batch_size
        self.train_workers = cfg.train_workers
        self.val_workers = cfg.train_workers

        self.full_scale = cfg.full_scale
        self.scale = cfg.scale
        self.max_npoint = cfg.max_npoint
        self.mode = cfg.mode
        try:
            self.train_mini = cfg.train_mini
        except:
            self.train_mini = False
        
        self.task = cfg.task

        try:
            self.without_angle = cfg.without_angle
        except:
            self.without_angle = False

        try:
            self.with_elastic = cfg.with_elastic
        except:
            self.with_elastic = False

        try:
            self.num_heading_bin = cfg.num_heading_bin
        except:
            self.num_heading_bin = 12

        if test:
            self.test_split = cfg.split  # val or test
            self.test_workers = cfg.test_workers
            cfg.batch_size = 1

        self.semantic_map = np.ones(40) * -1
        valid_class = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        for i, c in enumerate(valid_class):
            self.semantic_map[c] = i


    def trainLoader(self):
        if self.task == "trainval":
            sub_dir = "trainval"
        elif self.task == "val":
            sub_dir = "val"
        elif self.task == "val_mini":
            sub_dir = "val_mini"
        elif self.train_mini:
            sub_dir = "train_mini"
        else:
            sub_dir = "train"
        train_file_names = sorted(glob.glob(osp.join(self.data_root, self.dataset, sub_dir, "*" + self.filename_suffix)))
        self.train_files = [torch.load(i) for i in train_file_names]

        self.logger.info("Training samples: {}".format(len(self.train_files)))

        train_set = list(range(len(self.train_files)))
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.train_merge, num_workers=self.train_workers,
                                            shuffle=True, sampler=None, drop_last=True, pin_memory=True)
        # self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.val_merge, num_workers=self.train_workers,
        #                                     shuffle=True, sampler=None, drop_last=True, pin_memory=True)


    def valLoader(self):
        if self.task == "trainval":
            split = "val_mini"
        else:
            split = "val"
        self.val_file_names = sorted(glob.glob(osp.join(self.data_root, self.dataset, split, "*" + self.filename_suffix)))
        self.val_files = [torch.load(i) for i in self.val_file_names]

        self.logger.info("Validation samples: {}".format(len(self.val_files)))

        val_set = list(range(len(self.val_files)))
        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=self.val_merge, num_workers=self.val_workers,
                                          shuffle=False, drop_last=False, pin_memory=True)


    def testLoader(self):
        self.test_file_names = sorted(glob.glob(osp.join(self.data_root, self.dataset, self.test_split, "*" + self.filename_suffix)))
        self.test_files = [torch.load(i) for i in self.test_file_names]

        self.logger.info("Testing samples ({}): {}".format(self.test_split, len(self.test_files)))

        test_set = list(np.arange(len(self.test_files)))
        self.test_data_loader = DataLoader(test_set, batch_size=1, collate_fn=self.test_merge, num_workers=self.test_workers, shuffle=False, drop_last=False, pin_memory=True)

    #Elastic distortion
    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype("float32") / 3
        blur1 = np.ones((1, 3, 1)).astype("float32") / 3
        blur2 = np.ones((1, 1, 3)).astype("float32") / 3

        bb = np.abs(x).max(0).astype(np.int32)//gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype("float32") for _ in range(3)]
        noise = [ndimage.filters.convolve(n, blur0, mode="constant", cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode="constant", cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode="constant", cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur0, mode="constant", cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode="constant", cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode="constant", cval=0) for n in noise]
        ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
        interp = [interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
        def g(x_):
            return np.hstack([i(x_)[:,None] for i in interp])
        return x + g(x) * mag


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


    def dataAugmentOld(self, xyz, jitter=False, flip=False, rot=False, boxes=None):
        if jitter:
            m = np.eye(3)
            m += np.random.randn(3, 3) * 0.1
            xyz = xyz @ m
            if boxes is not None:
                boxes[:, :3] = boxes[:, :3] @ m
        if flip:
            # m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
            flag = np.random.randint(0, 2)
            # if flag:
            xyz[:, 0] = -xyz[:, 0]
            if boxes is not None:
                boxes[:, 0] = -boxes[:, 0]
                boxes[:, 6] = np.pi - boxes[:, 6]
        if rot:
            theta = np.random.rand() * 2 * math.pi
            rot_mat = np.eye(3)
            c, s = np.cos(theta), np.sin(theta)
            rot_mat[0, 0] = c
            rot_mat[0, 1] = -s
            rot_mat[1, 1] = c
            rot_mat[1, 0] = s
            xyz = xyz @ rot_mat.T
            if boxes is not None:
                boxes[:, :3] = boxes[:, :3] @ rot_mat.T
                boxes[:, 6] += theta
                boxes[:, 6][boxes[:, 6] > 2 * np.pi] -= 2 * np.pi

        if boxes is not None:
            return xyz, boxes
        else:
            return xyz


    def dataAugment(self, xyz, scale=False, flip=False, rot=False, boxes=None):
        if scale:
            scale = np.random.uniform(0.8, 1.2)
            xyz = xyz * scale
            if boxes is not None:
                boxes[:, :6] = boxes[:, :6] * scale
        if flip:
            # m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
            flag = np.random.randint(0, 2)
            # if flag:
            xyz[:, 0] = -xyz[:, 0]
            if boxes is not None:
                boxes[:, 0] = -boxes[:, 0]
                boxes[:, 6] = np.pi - boxes[:, 6]
        if rot:
            theta = np.random.rand() * 2 * math.pi
            rot_mat = np.eye(3)
            c, s = np.cos(theta), np.sin(theta)
            rot_mat[0, 0] = c
            rot_mat[0, 1] = -s
            rot_mat[1, 1] = c
            rot_mat[1, 0] = s
            xyz = xyz @ rot_mat.T
            if boxes is not None:
                boxes[:, :3] = boxes[:, :3] @ rot_mat.T
                boxes[:, 6] += theta
                boxes[:, 6][boxes[:, 6] > 2 * np.pi] -= 2 * np.pi

        if boxes is not None:
            return xyz, boxes
        else:
            return xyz


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


    def get_cropped_inst_label(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label


    def train_merge(self, id):
        if self.pose_file == "scene_align_old.pkl":
            semantic_label_idxs = [3, 4, 5, 6, 7, 10, 24, 33, 36]
            semantic_label_names = ["cabinet", "bed", "chair", "sofa", "table",
                                    "bookshelf", "refrigerator", "toilet", "bathtub"]
        elif self.pose_file == "scene_align.pkl":
            semantic_label_idxs = [3, 4, 5, 6, 7, 24, 33, 36, 39]
            semantic_label_names = ["cabinet", "bed", "chair", "sofa", "table",
                                    "refrigerator", "toilet", "bathtub", "otherfurniture"]
        else:
            raise ValueError
        label_map = {sem_label: idx for idx, sem_label in enumerate(semantic_label_idxs)}
        label_map[cfg.ignore_label] = cfg.ignore_label
        locs = []
        locs_float = []
        feats = []
        labels = []
        instance_labels = []

        instance_infos = []  # (N, 9)
        instance_pointnum = []  # (total_nInst), int

        batch_offsets = [0]
        bboxes_list = []
        syms_list = []
        angle_classes_list = []
        angle_residuals_list = []
        sample_idx_list = []
        xyz_offset_list = []
        overseg_list = []
        overseg_bias = 0

        total_inst_num = 0
        for i, idx in enumerate(id):
            xyz_origin, rgb, faces, label, instance_label, coords_shift, sample_idx = self.train_files[idx]
            scene = sample_idx.split("/")[1]
            sample_idx_list.append(scene)

            # read overseg
            with open(osp.join(".", "dataset", "scannetv2", "scans", scene, scene+"_vh_clean_2.0.010000.segs.json"), "r") as f:
                overseg = json.load(f)
            overseg = np.array(overseg["segIndices"])

            # read edge

            ### jitter / flip x / rotation
            # xyz_middle, boxes = self.dataAugment(xyz_origin, True, True, True, boxes=boxes)
            xyz_middle = self.dataAugment(xyz_origin, True, True, True, boxes=None)

            ### scale
            xyz = xyz_middle * self.scale

            ### elastic
            if self.with_elastic:
                xyz = self.elastic(xyz, 6 * self.scale // 50, 40 * self.scale / 50)
                xyz = self.elastic(xyz, 20 * self.scale // 50, 160 * self.scale / 50)

            ### offset
            xyz_offset = xyz.min(0)
            xyz -= xyz_offset
            xyz_offset_list.append(torch.from_numpy(xyz_offset))

            ### crop
            xyz, valid_idxs = self.crop(xyz)

            xyz_middle = xyz_middle[valid_idxs]
            xyz = xyz[valid_idxs]
            rgb = rgb[valid_idxs]
            label = label[valid_idxs]
            overseg = overseg[valid_idxs]
            _, overseg = np.unique(overseg, return_inverse=True)
            overseg += overseg_bias
            overseg_bias += (overseg.max() + 1)
            instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)

            ### get instance information
            inst_num, inst_infos = self.get_instance_info(xyz_middle, instance_label.astype(np.int32))
            inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            inst_pointnum = inst_infos["instance_pointnum"]   # (nInst), list

            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb) + torch.randn(3) * 0.1)
            labels.append(torch.from_numpy(label))
            instance_labels.append(torch.from_numpy(instance_label))
            overseg_list.append(torch.from_numpy(overseg))

            instance_infos.append(torch.from_numpy(inst_info))
            instance_pointnum.extend(inst_pointnum)

        ### merge all the scenes in the batchd
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)                                # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        overseg = torch.cat(overseg_list, 0).long()               # long(N)
        feats = torch.cat(feats, 0)                              # float (N, C)
        labels = torch.cat(labels, 0).long()                     # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()   # long (N)
        locs_offset = torch.stack(xyz_offset_list)               # long (B, 3)

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)       # float (N, 9) (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None) # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

        # # visual
        # visual_dir = osp.join(".", "visual")
        # import open3d as o3d
        # for idx, (start, end) in enumerate(zip(batch_offsets[:-1], batch_offsets[1:])):
        #     sample_idx = sample_idx_list[idx]
        #     coords = locs_float[start: end] # (n, 3)
        #     colors = torch.clamp(feats[start: end]+0.5, 0, 1) # (n, 3)
        #     pc = o3d.geometry.PointCloud()
        #     pc.points = o3d.utility.Vector3dVector(coords)
        #     pc.colors = o3d.utility.Vector3dVector(colors)
        #     o3d.io.write_point_cloud(osp.join(visual_dir, "{}_pc.ply".format(sample_idx)), pc)
        #     all_lines = o3d.geometry.LineSet()
            
        #     boxes = bboxes[idx]

        #     mesh = o3d.geometry.TriangleMesh()
        #     for box in boxes:
        #         if box[-1] == -100:
        #             continue
        #         translation = box[:3].numpy()
        #         size = box[3:6].numpy()
        #         corner = o3d.io.read_triangle_mesh("./visual/bbox.ply")
        #         corner.vertices = o3d.utility.Vector3dVector(np.array(corner.vertices) * size)
        #         z_angle = box[6]
        #         rot_mat = np.eye(3)
        #         rot_mat[0, 0] = np.cos(z_angle)
        #         rot_mat[0, 1] = -np.sin(z_angle)
        #         rot_mat[1, 1] = np.cos(z_angle)
        #         rot_mat[1, 0] = np.sin(z_angle)

        #         transform = np.eye(4)
        #         transform[0:3, 0:3] = rot_mat
        #         transform[0:3, 3] = translation
        #         corner.transform(transform)

        #         mesh += corner
        #     o3d.io.write_triangle_mesh(osp.join(visual_dir, "{}_corner.ply".format(sample_idx)), mesh)

        return {"locs": locs, "locs_offset": locs_offset, "voxel_locs": voxel_locs,
                "sample_idx_list": sample_idx_list, "p2v_map": p2v_map, "v2p_map": v2p_map,
                "locs_float": locs_float, "feats": feats, "labels": labels, "instance_labels": instance_labels,
                "instance_info": instance_infos, "instance_pointnum": instance_pointnum,
                "id": id, "offsets": batch_offsets, "spatial_shape": spatial_shape, "overseg": overseg,
                "bboxes": None, "syms": None, "angle_classes": None, "angle_residuals": None}


    def val_merge(self, id):
        if self.pose_file == "scene_align_old.pkl":
            semantic_label_idxs = [3, 4, 5, 6, 7, 10, 24, 33, 36]
            semantic_label_names = ["cabinet", "bed", "chair", "sofa", "table",
                                    "bookshelf", "refrigerator", "toilet", "bathtub"]
        elif self.pose_file == "scene_align.pkl":
            semantic_label_idxs = [3, 4, 5, 6, 7, 24, 33, 36, 39]
            semantic_label_names = ["cabinet", "bed", "chair", "sofa", "table",
                                    "refrigerator", "toilet", "bathtub", "otherfurniture"]
        else:
            raise ValueError
        label_map = {sem_label: idx for idx, sem_label in enumerate(semantic_label_idxs)}
        label_map[16] = cfg.ignore_label
        label_map[cfg.ignore_label] = cfg.ignore_label
        locs = []
        locs_float = []
        feats = []
        labels = []
        instance_labels = []
        semantic_preds_list = []
        overseg_list = []
        overseg_bias = 0

        instance_infos = []  # (N, 9)
        instance_pointnum = []  # (total_nInst), int


        batch_offsets = [0]
        bboxes_list = []
        syms_list = []
        angle_classes_list = []
        angle_residuals_list = []
        sample_idx_list = []
        xyz_offset_list = []

        total_inst_num = 0
        import random
        for i, idx in enumerate(id):
            xyz_origin, rgb, faces, label, instance_label, coords_shift, sample_idx = self.val_files[idx]
            scene = sample_idx.split("/")[1]
            sample_idx_list.append(scene)
            pose_content = self.all_pose_dict[scene]
            semantic_preds = np.load(osp.join("dataset", "scannetv2", "pred_seg_all", scene + ".npy"))
            semantic_preds = self.semantic_map[semantic_preds]

            with open(osp.join(".", "dataset", "scannetv2", "scans", scene, scene+"_vh_clean_2.0.010000.segs.json"), "r") as f:
                overseg = json.load(f)

            overseg = np.array(overseg["segIndices"])

            box_list = []
            sym_list = []
            
            for ins_id, pose_dict in pose_content.items():
                sym = pose_dict["sym"]
                if sym == "__SYM_NONE":
                    sym_list.append(0)
                elif sym == "__SYM_ROTATE_UP_2":
                    sym_list.append(1)
                elif sym == "__SYM_ROTATE_UP_4":
                    sym_list.append(2)
                elif sym == "__SYM_ROTATE_UP_INF":
                    sym_list.append(3)
                else:
                    sym_list.append(0)
                box = np.concatenate([pose_dict["translation"].reshape(-1) + coords_shift,
                                      pose_dict["size"].reshape(-1),
                                      pose_dict["z_angle"].reshape(1),
                                      np.array(label_map[pose_dict["category"]]).reshape(1)])
                box_list.append(box)

            if len(box_list) == 0:
                boxes = np.ones([1, 8], dtype=np.float32) * -100 # (1, 9)
                syms = np.zeros([1], dtype=np.int32) # (1)
            else:
                boxes = np.array(box_list, dtype=np.float32) # (n, 3+3+1+1)
                syms = np.array(sym_list, dtype=np.int32) # (n)

            ### jitter / flip x / rotation
            # xyz_middle, boxes = self.dataAugment(xyz_origin, True, True, True, boxes=boxes)
            xyz_middle, boxes = self.dataAugment(xyz_origin, False, False, False, boxes=boxes)
            
            # max length is 35
            max_length = 35
            # extract angle label
            angle_classes = np.zeros((max_length,), dtype=np.int64)
            angle_residuals = np.zeros((max_length,), dtype=np.float32)
            for idx, box in enumerate(boxes):
                angle_class, angle_residual = self.angle2class(box[6])
                angle_classes[idx] = angle_class
                angle_residuals[idx] = angle_residual
            
            angle_classes_list.append(torch.from_numpy(angle_classes))
            angle_residuals_list.append(torch.from_numpy(angle_residuals))

            # residual = max_length - len(boxes)
            boxes_empty = np.ones([max_length, 8]).astype(boxes.dtype) * -100
            boxes_empty[:len(boxes), :] = boxes
            # boxes = np.pad(boxes, ((0, residual), (0, 0)), "constant", constant_values=-100) # (40, 8)
            sym_empty = np.zeros([max_length]).astype(syms.dtype)
            sym_empty[:len(syms)] = syms

            bboxes_list.append(torch.from_numpy(boxes_empty))
            syms_list.append(torch.from_numpy(sym_empty))

            ### scale
            xyz = xyz_middle * self.scale

            ### offset
            xyz_offset = xyz.min(0)
            xyz -= xyz_offset
            xyz_offset_list.append(torch.from_numpy(xyz_offset))

            ### crop
            xyz, valid_idxs = self.crop(xyz)

            xyz_middle = xyz_middle[valid_idxs]
            xyz = xyz[valid_idxs]
            rgb = rgb[valid_idxs]
            label = label[valid_idxs]
            overseg = overseg[valid_idxs]
            _, overseg = np.unique(overseg, return_inverse=True)
            overseg += overseg_bias
            overseg_bias += (overseg.max() + 1)
            instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)
            semantic_preds = semantic_preds[valid_idxs]

            ### get instance information
            inst_num, inst_infos = self.get_instance_info(xyz_middle, instance_label.astype(np.int32))
            inst_info = inst_infos["instance_info"] # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            inst_pointnum = inst_infos["instance_pointnum"]  # (nInst), list

            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb))
            labels.append(torch.from_numpy(label))
            instance_labels.append(torch.from_numpy(instance_label))
            semantic_preds_list.append(torch.from_numpy(semantic_preds))
            overseg_list.append(torch.from_numpy(overseg))

            instance_infos.append(torch.from_numpy(inst_info))
            instance_pointnum.extend(inst_pointnum)

        ### merge all the scenes in the batch
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)                                  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)    # float (N, 3)
        feats = torch.cat(feats, 0)                                # float (N, C)
        labels = torch.cat(labels, 0).long()                       # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()     # long (N)
        locs_offset = torch.stack(xyz_offset_list)               # long (B, 3)
        semantic_preds = torch.cat(semantic_preds_list, 0).long() # long (N)
        overseg = torch.cat(overseg_list, 0).long()               # long(N)

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)               # float (N, 9) (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)          # int (total_nInst)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

        ### get bboxes(pose)
        bboxes = torch.stack(bboxes_list)
        syms = torch.stack(syms_list)
        angle_classes = torch.stack(angle_classes_list)
        angle_residuals = torch.stack(angle_residuals_list)

        return {"locs": locs, "locs_offset": locs_offset, "voxel_locs": voxel_locs,
                "sample_idx_list": sample_idx_list, "p2v_map": p2v_map, "v2p_map": v2p_map,
                "locs_float": locs_float, "feats": feats, "labels": labels, "instance_labels": instance_labels,
                "instance_info": instance_infos, "instance_pointnum": instance_pointnum,
                "id": id, "offsets": batch_offsets, "spatial_shape": spatial_shape, "overseg": overseg,
                "bboxes": bboxes, "syms": syms, "angle_classes": angle_classes, "angle_residuals": angle_residuals,
                "semantic_preds": semantic_preds}


    def test_merge(self, id):
        locs = []
        locs_float = []
        feats = []

        batch_offsets = [0]
        sample_idx_list = []
        xyz_offset_list = []
        semantic_preds_list = []
        overseg_list = []
        connect_map_list = []

        instance_labels = []
        instance_infos = []  # (N, 9)
        instance_pointnum = []  # (total_nInst), int

        for i, idx in enumerate(id):
            if "val" in self.test_split:
                xyz_origin, rgb, faces, label, instance_label, coords_shift, sample_idx = self.test_files[idx]
                split = "scans"
            elif self.test_split == "train":
                xyz_origin, rgb, faces, label, instance_label, coords_shift, sample_idx = self.test_files[idx]
                split = "scans"
            elif self.test_split == "test":
                xyz_origin, rgb, faces, sample_idx = self.test_files[idx]
                split = "scans_test"
            elif self.test_split == "test_bathtub":
                xyz_origin, rgb, faces, sample_idx = self.test_files[idx]
                split = "scans_test"
            else:
                print("Wrong test split: {}!".format(self.test_split))
                exit(0)
                

            scene = sample_idx.split("/")[1]
            sample_idx_list.append(scene)
            semantic_preds = np.load(osp.join("dataset", "scannetv2", "pred_seg_all", scene + ".npy"))
            semantic_preds = self.semantic_map[semantic_preds]

            # read overseg
            with open(osp.join(".", "dataset", "scannetv2", split, scene, scene+"_vh_clean_2.0.010000.segs.json"), "r") as f:
                overseg = json.load(f)
            overseg = np.array(overseg["segIndices"])
            _, overseg = np.unique(overseg, return_inverse=True)
            # read edge
            edges = np.concatenate([faces[:, :2], faces[:, 1:], faces[:, [0, 2]]]) # (nEdges, 2)
            overseg_edges = overseg[edges]
            overseg_edges = overseg_edges[overseg_edges[:, 0] != overseg_edges[:, 1]]
            num_overseg = overseg.max() + 1
            # save the overseg connection as connect_map
            connect_map = np.eye(num_overseg, dtype=np.bool) # (num_overseg)
            connect_map[overseg_edges[:, 0], overseg_edges[:, 1]] = True
            connect_map[overseg_edges[:, 1], overseg_edges[:, 0]] = True
            connect_map = torch.Tensor(connect_map)
            connect_map_list.append(connect_map)

            ### flip x / rotation
            # xyz_middle = self.dataAugment(xyz_origin, False, True, True)
            xyz_middle = self.dataAugment(xyz_origin, False, False, False)

            ### scale
            xyz = xyz_middle * self.scale

            ### offset
            xyz_offset = xyz.min(0)
            xyz -= xyz_offset
            xyz_offset_list.append(torch.from_numpy(xyz_offset))

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb))
            semantic_preds_list.append(torch.from_numpy(semantic_preds))
            overseg_list.append(torch.from_numpy(overseg))

            if self.test_split == "val":
                ### get instance information
                inst_num, inst_infos = self.get_instance_info(xyz_middle, instance_label.astype(np.int32))
                inst_info = inst_infos["instance_info"] # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
                inst_pointnum = inst_infos["instance_pointnum"]  # (nInst), list
                instance_labels.append(torch.from_numpy(instance_label))
                instance_infos.append(torch.from_numpy(inst_info))
                instance_pointnum.extend(inst_pointnum)

        ### merge all the scenes in the batch
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int) # int (B+1)

        locs = torch.cat(locs, 0)                                 # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)   # float (N, 3)
        semantic_preds = torch.cat(semantic_preds_list, 0).long() # long (N)
        overseg = torch.cat(overseg_list, 0).long()               # long(N)
        feats = torch.cat(feats, 0)                               # float (N, C)
        locs_offset = torch.stack(xyz_offset_list)  # long (B, 3)
        
        
        if self.test_split == "val":
            instance_labels = torch.cat(instance_labels, 0).long()     # long (N)
            instance_infos = torch.cat(instance_infos, 0).to(torch.float32)               # float (N, 9) (meanxyz, minxyz, maxxyz)
            instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)          # int (total_nInst)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)


        if self.test_split == "val":
            return {"locs": locs, "locs_offset": locs_offset, "voxel_locs": voxel_locs,
                    "sample_idx_list": sample_idx_list, "p2v_map": p2v_map, "v2p_map": v2p_map,
                    "locs_float": locs_float, "feats": feats,
                    "instance_labels": instance_labels, "instance_info": instance_infos, "instance_pointnum": instance_pointnum,
                    "id": id, "offsets": batch_offsets, "spatial_shape": spatial_shape, "semantic_preds": semantic_preds,
                    "overseg": overseg, "connect_map": connect_map_list}
        else:
            return {"locs": locs, "locs_offset": locs_offset, "voxel_locs": voxel_locs,
                    "sample_idx_list": sample_idx_list, "p2v_map": p2v_map, "v2p_map": v2p_map,
                    "locs_float": locs_float, "feats": feats,
                    "id": id, "offsets": batch_offsets, "spatial_shape": spatial_shape, "semantic_preds": semantic_preds,
                    "overseg": overseg, "connect_map": connect_map_list}

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from  
            class center angle to current angle.
            
            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle
        '''
        num_class = self.num_heading_bin
        angle = angle % (2 * np.pi)
        assert(angle >= 0 and angle <= 2 * np.pi)
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle


