# Copyright (c) Gorilla-Lab. All rights reserved.
import enum
import os
import os.path as osp
import glob
import argparse
from scipy.spatial import cKDTree

import numpy as np
import torch

import gorilla

try:
    import pointgroup_ops
except:
    raise ImportError("must install `pointgroup_ops` from lib")


def get_parser():
    parser = argparse.ArgumentParser(description="downsample s3dis by voxelization")
    parser.add_argument("--data-dir",
                        type=str,
                        default="./inputs",
                        help="directory save processed data")
    parser.add_argument("--voxel-size",
                        type=float,
                        default=0.01,
                        help="voxelization size")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="show partition information or not")

    args_cfg = parser.parse_args()

    return args_cfg


if __name__ == "__main__":
    args = get_parser()

    data_dir = args.data_dir
    save_dir = f"{data_dir}_voxelize"
    os.makedirs(save_dir, exist_ok=True)

    # for data_file in [osp.join(data_dir, "Area_6_office_17.pth")]:
    for data_file in gorilla.track(glob.glob(osp.join(data_dir, "*.pth"))):
    # for data_file in glob.glob(osp.join(data_dir, "*.pth")):
        (coords, colors, semantic_labels, instance_labels, room_label, scene) = torch.load(data_file)

        if args.verbose:
            print(f"processing: {scene}")

        save_path = osp.join(save_dir, f"{scene}.pth")
        if os.path.exists(save_path):
            continue

        # move to positive area
        coords -= coords.min(0)
        origin_coords = coords.copy()
        # begin voxelize
        num_points = coords.shape[0]
        voxelize_coords = torch.from_numpy(coords / args.voxel_size).long() # [num_point, 3]
        voxelize_coords = torch.cat([torch.zeros(num_points).view(-1, 1).long(), voxelize_coords], 1)  # [num_point, 1 + 3]
        # mode=4 is mean pooling
        voxelize_coords, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(voxelize_coords, 1, 4)
        v2p_map = v2p_map.cuda()
        coords = torch.from_numpy(coords).float().cuda()
        coords = pointgroup_ops.voxelization(coords, v2p_map, 4).cpu().numpy() # [num_voxel, 3]
        colors = torch.from_numpy(colors).float().cuda()
        colors = pointgroup_ops.voxelization(colors, v2p_map, 4).cpu().numpy() # [num_voxel, 3]

        # processing labels individually (nearest search)
        voxelize_coords = voxelize_coords[:, 1:].cpu().numpy() * args.voxel_size
        tree = cKDTree(origin_coords)

        _, idx = tree.query(voxelize_coords, k=1)
        semantic_labels = semantic_labels[idx]
        instance_labels = instance_labels[idx]

        # # round (maybe not accurate)
        # semantic_labels = torch.from_numpy(semantic_labels).float().view(-1, 1).cuda() # [num_point, 1]
        # instance_labels = torch.from_numpy(instance_labels).float().view(-1, 1).cuda() # [num_point, 1]
        # semantic_labels = pointgroup_ops.voxelization(semantic_labels, v2p_map, 4).cpu().view(-1) # [num_voxel]
        # semantic_labels = torch.round(semantic_labels).long().numpy() # [num_voxel]
        # instance_labels = pointgroup_ops.voxelization(instance_labels, v2p_map, 4).cpu().view(-1) # [num_voxel]
        # instance_labels = torch.round(instance_labels).long().numpy() # [num_voxel]

        torch.save((coords,
                    colors,
                    semantic_labels,
                    instance_labels,
                    room_label,
                    scene), save_path)

