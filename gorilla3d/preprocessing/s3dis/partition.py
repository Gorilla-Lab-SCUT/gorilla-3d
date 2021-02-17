# Copyright (c) Gorilla-Lab. All rights reserved.
import enum
import os
import os.path as osp
import glob
import argparse
from itertools import product
from typing import List, Optional
import gorilla

import numpy as np
import torch


def sample_cloud(num_points, num_sample):
    if num_points >= num_sample:
        indices = np.random.choice(num_points, num_sample, replace=False)
    else:
        indices = np.random.choice(num_points, num_sample - num_points, replace=True)
        indices = list(range(num_points)) + list(indices)
    return indices


def room_to_blocks(coords: np.ndarray,
                   colors: np.ndarray,
                   semantic_labels: np.ndarray,
                   instance_labels: np.ndarray,
                   superpoint: Optional[np.ndarray]=None,
                   size: float=1.0,
                   stride: float=0.5,
                   threshold: int=4096,
                   num_sample: Optional[int]=None,
                   verbose: bool=False) -> List:
    assert coords.shape[0] == colors.shape[0] == semantic_labels.shape[0] == instance_labels.shape[0]
    upper_bound = coords.max(axis=0) # get upper bound of coordinates
    lower_bound = coords.min(axis=0) # get lower bound of coordinates

    if superpoint is not None:
        assert superpoint.shape[0] == coords.shape[0]

    # partition into x-y axis blocks according to size and stride
    width = max(1, int(np.ceil((upper_bound[0] - lower_bound[0]) / stride)))
    depth = max(1, int(np.ceil((upper_bound[1] - lower_bound[1]) / stride)))
    cells = [xy for xy in product(range(width), range(depth))]

    if verbose:
        print(f"number of points: {coords.shape[0]}")

    # generate blocks
    block_list = []
    for (x, y) in cells:
        # get the inner block points' indices
        x_bound = (x * stride + lower_bound[0] + size / 2)
        y_bound = (y * stride + lower_bound[1] + size / 2)
        xcond = (coords[:, 0] >= x_bound) & (coords[:, 0] <= x_bound + size)
        ycond = (coords[:, 1] >= y_bound) & (coords[:, 1] <= y_bound + size)
        cond  = xcond & ycond
        # filter out the meaningless block
        if verbose:
            print(f"({x}, {y}): {cond.sum()}")
        if np.sum(cond) < threshold:
            continue
        num_points = cond.sum()
        block_indices = np.where(cond)[0]
        if num_sample is not None:
            sample_indices = sample_cloud(num_points, num_sample)
            block_indices = block_indices[sample_indices]
        block_coords = coords[block_indices] # [num_block, 3]
        block_colors = colors[block_indices] # [num_block, 3]
        block_semantic_labels = semantic_labels[block_indices, None] # [num_block, 1]
        block_instance_labels = instance_labels[block_indices, None] # [num_block, 1]
        if block_instance_labels.max() < 0:
            continue
        concat_list = [block_coords, block_colors, block_semantic_labels, block_instance_labels]
        if superpoint is not None:
            block_superpoint = superpoint[block_indices, None] # [num_block, 1]
            concat_list.append(block_superpoint)

        block = np.concatenate(concat_list, axis=1) # [num_block, 8/9]
        block_list.append(block)
    return block_list


def get_parser():
    parser = argparse.ArgumentParser(description="s3dis room partition "
                                                 "refer to jsis3d")
    parser.add_argument("--data-root",
                        type=str,
                        default=".",
                        help="root dir save data(different from --data-root in "
                             "prepare_data_inst.py)")
    parser.add_argument("--data-split",
                        type=str,
                        default="train",
                        help="data split (train / val)")
    parser.add_argument("--size",
                        type=float,
                        default=1.0,
                        help="parition block size")
    parser.add_argument("--stride",
                        type=float,
                        default=0.5,
                        help="parition block stride")
    parser.add_argument("--threshold",
                        type=int,
                        default=4096,
                        help="parition number threshold")
    parser.add_argument("--sample",
                        type=int,
                        default=None,
                        help="number of sample points, default is None(do not sample)")
    parser.add_argument("--with-superpoint",
                        action="store_true",
                        help="process superpoint or not")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="show partition information or not")

    args_cfg = parser.parse_args()

    return args_cfg


if __name__ == "__main__":
    args = get_parser()

    data_root = args.data_root
    split = args.data_split
    data_dir = osp.join(data_root, split)
    save_dir = f"{data_dir}_blocks"
    os.makedirs(save_dir, exist_ok=True)

    for data_file in gorilla.track(glob.glob(osp.join(data_dir, "*.pth"))):
    # for data_file in glob.glob(osp.join(data_dir, "*.pth")):
        (coords, colors, semantic_labels, instance_labels, room_label, scene) = torch.load(data_file)
        superpoint = None
        if args.with_superpoint:
            superpoint_file = osp.join(data_root, "superpoint", f"{scene}.npy")
            superpoint = np.load(superpoint_file)

        if args.verbose:
            print(f"processing: {scene}")

        block_list = room_to_blocks(coords,
                                    colors,
                                    semantic_labels,
                                    instance_labels,
                                    superpoint=superpoint,
                                    size=args.size,
                                    stride=args.stride,
                                    threshold=args.threshold,
                                    num_sample=args.sample,
                                    verbose=args.verbose)

        for idx, block in enumerate(block_list):
            block_coords = block[:, 0:3] # [num_block, 3]
            block_colors = block[:, 3:6] # [num_block, 3]
            block_semantic_labels = block[:, 6] # [num_block]
            block_instance_labels = block[:, 7] # [num_block]
            scene_idx = f"{scene}_{idx}"
            torch.save((block_coords,
                        block_colors,
                        block_semantic_labels,
                        block_instance_labels,
                        room_label,
                        scene_idx), osp.join(save_dir, f"{scene_idx}.pth"))
            if superpoint is not None:
                block_superpoint = block[:, 8]
                np.save(osp.join(data_root, "superpoint", f"{scene_idx}.npy"), block_superpoint)

