"""
Generate instance and semantic groundtruth .txt files (for evaluation)
"""
import os
import glob
import argparse

import torch
import plyfile
import numpy as np

import gorilla

if __name__ == "__main__":
    semantic_label_idxs = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
    ]
    semantic_label_names = [
        "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
        "window", "bookshelf", "picture", "counter", "desk", "curtain",
        "refrigerator", "shower curtain", "toilet", "sink", "bathtub",
        "otherfurniture"
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-split",
                        help="data split (train / val)",
                        default="val")
    opt = parser.parse_args()
    split = opt.data_split
    files = sorted(glob.glob(f"{split}/scene*_inst_nostuff.pth"))
    rooms = [torch.load(i) for i in gorilla.track(files)]

    if not os.path.exists(split + "_gt"):
        os.mkdir(split + "_gt")

    print("begin to prepare instance groundtruth")
    # instance segmentaiton
    for i, room in enumerate(rooms):
        xyz, rgb, faces, label, instance_label, coords_shift, sample_idx = \
            room # label 0~19 -100;  instance_label 0~instance_num-1 -100
        scene_name = files[i].split("/")[-1][:12]
        print(f"{i + 1}/{len(rooms)} {scene_name}")

        save_path = os.path.join(split + "_gt", scene_name + ".txt")
        if os.path.exists(save_path):
            continue

        instance_label_new = np.zeros(
            instance_label.shape, dtype=np.int32
        )  # 0 for unannotated, xx00y: x for semantic_label, y for inst_id (1~instance_num)

        instance_num = int(instance_label.max()) + 1
        for inst_id in range(instance_num):
            instance_mask = np.where(instance_label == inst_id)[0]
            sem_id = int(label[instance_mask[0]])
            if (sem_id == -100): sem_id = 0
            semantic_label = semantic_label_idxs[sem_id]
            instance_label_new[
                instance_mask] = semantic_label * 1000 + inst_id + 1

        np.savetxt(save_path, instance_label_new, fmt="%d")
    print("instance groundtruth preparation finish")

    print("begin to prepare semantic groundtruth")
    # semantic segmentaiton
    for i, f in enumerate(files):
        scene_name = f.split("/")[-1][:12]
        save_path = os.path.join(split + "_gt", scene_name + "_sem.txt")
        print(f"{i + 1}/{len(rooms)} {scene_name}")
        if os.path.exists(save_path):
            continue

        ply_file = os.path.join("scans", scene_name, f"{scene_name}_vh_clean_2.labels.ply")
        ply = plyfile.PlyData().read(ply_file)
        label = np.array(ply.elements[0]["label"])

        np.savetxt(save_path, label, fmt="%d")
    print("semantic groundtruth preparation finish")


