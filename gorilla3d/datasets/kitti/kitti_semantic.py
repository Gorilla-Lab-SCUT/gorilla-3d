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
from ..utils import PointCloudTransfromer, PolarProcesses, GridProcesses


# modify from https://github.com/xinge008/Cylinder3D/blob/master/dataloader/dataset_semantickitti.py
class KittiSem(Dataset):
    def __init__(self,
                 data_root: str,
                 task: str = "train",
                 label_mapping: str = "semantic-kitti.yaml",
                 return_ref: bool = False,
                 return_test: bool = False,
                 project: bool = False,
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
                 with_instance: bool = False,
                 **kwargs):
        self.logger = gorilla.derive_logger(__name__)
        self.return_ref = return_ref
        self.return_test = return_test
        self.semkittiyaml = gorilla.load(label_mapping)
        self.learning_map = self.semkittiyaml["learning_map"]
        self.label_mapper = np.vectorize(self.learning_map.__getitem__)
        assert task in [
            "train", "val", "test"
        ], f"`task` must be in ['train', 'val', 'test'], but got {task}"
        self.sequences = self.semkittiyaml["split"][task]
        self.task = task
        self.with_instance = with_instance

        self.data_files = []
        for i_folder in self.sequences:
            i_folder = f"{i_folder:0>2}"
            self.data_files += glob.glob(
                os.path.join(data_root, f"{i_folder:0>2}", "velodyne", "*"))

        self.data_files.sort()
        self.data_files = self.data_files

        self.logger.info("Using {} scans from sequences {}".format(
            len(self.data_files), self.sequences))

        # initialize point cloud transformer
        self.pc_transformer = PointCloudTransfromer(**transform_cfg)
        processer_caller = globals()[grid_cfg["type"]]
        # initialize voxelizer
        self.processer = processer_caller(**grid_cfg)

        # projection(range view)
        self.project = project
        if self.project:
            self.max_points = kwargs.get("max_points", 150000)
            self.scan = SemLaserScan(**kwargs["sensor"])
            self.color_map = self.semkittiyaml["color_map"]
            self.sensor_img_means = np.array(kwargs.get(
                "img_means", [12.12, 10.88, 0.23, -1.04, 0.21]),
                                             dtype=np.float32)
            self.sensor_img_stds = np.array(kwargs.get(
                "img_stds", [12.32, 11.47, 6.91, -0.86, 0.16]),
                                            dtype=np.float32)

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.data_files)

    def __getitem__(self, index):
        data_file = self.data_files[index]
        scene_name = (
            f"{data_file.split('/')[-3]}_{data_file.split('/')[-1].split('.')[0]}"
        )  # xx_yyyyyy
        raw_data = np.fromfile(data_file, dtype=np.float32).reshape(
            (-1, 4))  # [N, 4]
        if self.task == "test":
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0],
                                                          dtype=int),
                                            axis=1)
        else:
            annotated_data = np.fromfile(self.data_files[index].replace(
                "velodyne", "labels").replace(".bin", ".label"),
                                         dtype=np.int32).reshape(
                                             (-1, 1))  # [N, 1]
            instance_label = annotated_data.copy()
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = self.label_mapper(
                annotated_data)  # annotated id map

        if self.task == "train":
            raw_data = self.pc_transformer(raw_data)
        voxel_position, processed_label, grid_ind, labels, processed_xyz = self.processer(
            raw_data, annotated_data)
        xyz = raw_data[:, :3]
        if self.return_ref:
            sig = raw_data[:, 3]
            if len(sig.shape) == 2:
                sig = np.squeeze(sig)
            return_fea = np.concatenate((processed_xyz, sig[..., np.newaxis]),
                                        axis=1)

        data_tuple = (voxel_position, processed_label, grid_ind, labels, xyz,
                      return_fea)

        if self.project:
            # reset scan
            self.scan.reset()

            # read points and labels
            self.scan.set_points(raw_data[:, :3], raw_data[:, 3])
            self.scan.set_label(annotated_data)
            # NOTE: the label has been mapped, do not need to map again
            # self.scan.sem_label_map(self.label_mapper) # map labels

            # get un-projected paramters(origin input)
            npoint = self.scan.points.shape[0]
            num_pad = self.max_points - npoint

            # get projected paramters
            proj_range = self.scan.proj_range.copy()  # [H, W]
            proj_xyz = self.scan.proj_xyz.copy()  # [H, W, 3]
            proj_remission = self.scan.proj_remission.copy()  # [H, W]
            proj_mask = self.scan.proj_mask.copy()  # [H, W]
            proj_label = self.scan.proj_sem_label.copy()  # [H, W]
            proj_label = proj_label * proj_mask  # [H, W]
            # model input
            proj = np.concatenate([
                proj_range[None, :, :].copy(),
                proj_xyz.transpose(2, 0, 1).copy(),
                proj_remission[None, :, :].copy()
            ])  # [5, H, W]
            proj = (proj - self.sensor_img_means[:, None, None]
                    ) / self.sensor_img_stds[:, None, None]  # [5, H, W]
            proj = proj * proj_mask.astype(np.float32)

            # get the projection x and y ids(in pixel coordinates)
            proj_x = np.pad(self.scan.proj_x, (0, num_pad),
                            constant_values=(-1.0, -1.0)).astype(
                                np.int64)  # [max_points]
            proj_y = np.pad(self.scan.proj_y, (0, num_pad),
                            constant_values=(-1.0, -1.0)).astype(
                                np.int64)  # [max_points]
            data_tuple += (proj, proj_label, proj_x, proj_y, npoint)

        if self.with_instance:
            data_tuple += (instance_label, )

        data_tuple += (scene_name, )

        return data_tuple

    @property
    def label_name(self):
        SemKITTI_label_name = {}
        for i in sorted(list(self.learning_map.keys()))[::-1]:
            SemKITTI_label_name[
                self.learning_map[i]] = self.semkittiyaml["labels"][i]
        return SemKITTI_label_name

    @staticmethod
    def collate_fn(batch):
        voxel_centers = []
        voxel_labels = []
        grid_inds = []
        point_labels = []
        point_xyzs = []
        point_features = []
        scene_names = []
        # TODO: ugly
        proj_flag = len(batch[0]) >= 12
        if proj_flag:
            projs = []
            proj_labels = []
            proj_xs = []
            proj_ys = []
            npoints = []

        # TODO: ugly
        instance_flag = len(batch[0]) == 8 or len(batch[0]) == 13
        if instance_flag:
            instance_labels = []

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
            if proj_flag:
                projs.append(torch.from_numpy(b[6]).float())
                proj_labels.append(torch.from_numpy(b[7]).float())
                proj_xs.append(torch.from_numpy(b[8]).long())
                proj_ys.append(torch.from_numpy(b[9]).long())
                npoints.append(torch.Tensor(b[10]).long())
            if instance_flag:
                instance_labels.append(torch.from_numpy(
                    b[-2]))  # NOTE: point-wise not voxel-wise
            scene_names.append(b[-1])

        voxel_centers = torch.stack(voxel_centers)  # [B, H, W, D, 3]
        voxel_labels = torch.stack(voxel_labels)  # [B, H, W, D]
        grid_inds = torch.cat(grid_inds, 0)  # [N, 4]
        point_labels = torch.cat(point_labels, 0)  # [N]
        point_xyzs = torch.cat(point_xyzs, 0)  # [N, 3]
        point_features = torch.cat(point_features, 0)  # [N, C]

        data = {
            "voxel_centers": voxel_centers,
            "voxel_labels": voxel_labels,
            "grid_inds": grid_inds,
            "point_labels": point_labels,
            "point_xyzs": point_xyzs,
            "point_features": point_features,
            "scene_names": scene_names
        }

        if proj_flag:
            projs = torch.stack(projs)  # [B, 5, H, W]
            proj_labels = torch.stack(proj_labels)  # [B, H, W]
            proj_xs = torch.stack(proj_xs)  # [B, max_point]
            proj_ys = torch.stack(proj_ys)  # [B, max_point]
            npoints = torch.cat(npoints)  # [B]
            data.update({
                "projs": projs,
                "proj_labels": proj_labels,
                "proj_xs": proj_xs,
                "proj_ys": proj_ys,
                "npoints": npoints,
            })

        if instance_flag:
            data.update({
                "instance_labels": torch.cat(instance_labels, 0)  # [N]
            })

        return data


class KittiSemRV(KittiSem):
    def __init__(
            self,
            data_root: str,  # directory where data is
            task: str = "train",
            label_mapping: str = "semantic-kitti.yaml",
            max_points: int = 150000,  # max number of points present in dataset
            transform: bool = False,
            sensor: Dict = dict(
                project=True,
                height=64,
                width=2048,
                fov_up=3,
                fov_down=-25,
            ),  # sensor to parse scans from
            img_means=[12.12, 10.88, 0.23, -1.04, 0.21],
            img_stds=[12.32, 11.47, 6.91, -0.86, 0.16],
            **kwargs):  # send ground truth?
        # save deats
        super().__init__(data_root, task, label_mapping, **kwargs)

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
        raw_data = np.fromfile(self.data_files[index],
                               dtype=np.float32).reshape((-1, 4))  # [N, 4]
        if gt_flag:
            label_file = scan_file.replace("velodyne",
                                           "labels").replace(".bin", ".label")
            annotated_data = np.fromfile(label_file, dtype=np.int32)  # [N]
        else:
            # construct the fake labels
            annotated_data = np.zeros_like(raw_data[:, 0])  # [N, 1]

        # point cloud data augment
        if self.task == "train":
            raw_data = self.pc_transformer(raw_data)

        # reset scan
        self.scan.reset()

        # read points and labels
        self.scan.set_points(raw_data[:, :3], raw_data[:, 3])
        self.scan.set_label(annotated_data)
        self.scan.sem_label_map(self.label_mapper)  # map labels

        # get un-projected paramters(origin input)
        npoints = self.scan.points.shape[0]
        num_pad = self.max_points - npoints
        unproj_xyz = np.pad(self.scan.points, ((0, num_pad), (0, 0)),
                            constant_values=(-1.0, -1.0))  # [max_points, 3]
        unproj_range = np.pad(self.scan.unproj_range, (0, num_pad),
                              constant_values=(-1.0, -1.0))  # [max_points]
        unproj_remissions = np.pad(self.scan.remissions, (0, num_pad),
                                   constant_values=(-1.0,
                                                    -1.0))  # [max_points]
        unproj_labels = np.pad(self.scan.sem_label, (0, num_pad),
                               constant_values=(-1, -1)).astype(
                                   np.int32)  # [max_points]

        # get projected paramters
        proj_range = self.scan.proj_range.copy()  # [H, W]
        proj_xyz = self.scan.proj_xyz.copy()  # [H, W, 3]
        proj_remission = self.scan.proj_remission.copy()  # [H, W]
        proj_mask = self.scan.proj_mask.copy()  # [H, W]
        proj_labels = self.scan.proj_sem_label.copy()  # [H, W]
        proj_labels = proj_labels * proj_mask  # [H, W]
        # model input
        proj = np.concatenate([
            proj_range[None, :, :].copy(),
            proj_xyz.transpose(2, 0, 1).copy(),
            proj_remission[None, :, :].copy()
        ])  # [5, H, W]
        proj = (proj - self.sensor_img_means[:, None, None]
                ) / self.sensor_img_stds[:, None, None]  # [5, H, W]
        proj = proj * proj_mask.astype(np.float32)

        # get the projection x and y ids(in pixel coordinates)
        proj_x = np.pad(self.scan.proj_x, (0, num_pad),
                        constant_values=(-1.0, -1.0)).astype(
                            np.int64)  # [max_points]
        proj_y = np.pad(self.scan.proj_y, (0, num_pad),
                        constant_values=(-1.0, -1.0)).astype(
                            np.int64)  # [max_points]

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
