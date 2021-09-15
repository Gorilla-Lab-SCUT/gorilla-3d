# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import argparse

import gorilla
import numpy as np
from tqdm import tqdm

from .visualize import save_label_scene

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="visualization for semantic kitti(only for label now)")
    parser.add_argument("--data-root",
                        help="path to the input dataset files",
                        default="../../data/kitti/dataset/sequence")
    parser.add_argument("--save-root",
                        help="path to save visual result",
                        default="./vis")
    parser.add_argument("--config",
                        help="path to the input dataset files",
                        default="../../data/kitti/semantic-kitti.yaml")
    parser.add_argument("--fold",
                        help="path to the input dataset files",
                        default="0")
    args = parser.parse_args()

    data_root = args.data_root
    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)

    # build the label mapper according to given config file
    config = gorilla.load(args.config)
    learning_map = config["learning_map"]
    label_mapper = np.vectorize(learning_map.__getitem__)

    # read input fold list
    fold_list = list(map(lambda x: f"{x:0>2}", args.fold.split(",")))

    # process each fold
    for fold in fold_list:
        fold_dir = os.path.join(data_root, fold)
        save_dir = os.path.join(save_root, fold)
        os.makedirs(save_dir, exist_ok=True)
        gorilla.check_dir(fold_dir)
        print(f"process {fold}:")
        pc_dir = os.path.join(fold_dir, "velodyne")
        label_dir = os.path.join(fold_dir, "labels")
        scene_id_list = list(
            map(lambda x: os.path.splitext(x)[0], os.listdir(pc_dir)))

        for scene_id in tqdm(scene_id_list):
            save_path = os.path.join(save_dir, f"{scene_id}.ply")
            save_label_scene(pc_dir, label_dir, scene_id, label_mapper,
                             save_path)
