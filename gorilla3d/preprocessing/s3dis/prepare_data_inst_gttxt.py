"""
Generate instance groundtruth .txt files (for evaluation)
"""

import argparse
import numpy as np
import glob
import torch
import os

import gorilla

INV_OBJECT_LABEL = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    7: "chair",
    8: "table",
    9: "bookcase",
    10: "sofa",
    11: "board",
    12: "clutter",
}

if __name__ == "__main__":
    semantic_label_idxs = list(range(13))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-split", help="data split (train / val)", default="val")
    opt = parser.parse_args()
    split = opt.data_split
    files = sorted(glob.glob(f"{split}/*.pth"))
    rooms = [torch.load(i) for i in gorilla.track(files)]

    if not os.path.exists(split + "_gt"):
        os.mkdir(split + "_gt")

    for i in range(len(rooms)):
        (xyz, rgb, semantic_labels, instance_labels, room_label, scene) = rooms[i] # semantic label 0-12 instance_labels 0~instance_num-1 -100
        print(f"{i + 1}/{len(rooms)} {scene}")

        instance_labels_new = np.zeros(instance_labels.shape, dtype=np.int32)  # 0 for unannotated, xx00y: x for semantic_label, y for inst_id (1~instance_num)

        instance_num = int(instance_labels.max()) + 1
        inst_ids = np.unique(instance_labels)
        for inst_id in inst_ids:
            if inst_id <= 0:
                continue
            instance_mask = np.where(instance_labels == inst_id)[0]
            sem_id = int(semantic_labels[instance_mask[0]])
            if(sem_id == -100): sem_id = 0
            semantic_label = semantic_label_idxs[sem_id]
            instance_labels_new[instance_mask] = semantic_label * 1000 + inst_id

        np.savetxt(os.path.join(split + "_gt", scene + ".txt"), instance_labels_new, fmt="%d")





