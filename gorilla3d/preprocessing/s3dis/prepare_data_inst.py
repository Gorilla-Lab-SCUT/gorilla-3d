# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import os.path as osp
import glob
import argparse
import multiprocessing as mp
from warnings import warn

import torch
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import gorilla

ROOM_TYPES = {
    "conferenceRoom": 0,
    "copyRoom": 1,
    "hallway": 2,
    "office": 3,
    "pantry": 4,
    "WC": 5,
    "auditorium": 6,
    "storage": 7,
    "lounge": 8,
    "lobby": 9,
    "openspace": 10,
}

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

OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}

def object_name_to_label(object_class):
    r"""convert from object name in S3DIS to an int"""
    object_label = OBJECT_LABEL.get(object_class, OBJECT_LABEL["clutter"])
    return object_label

# modify from https://github.com/nicolas-chaulet/torch-points3d/blob/master/torch_points3d/datasets/segmentation/s3dis.py
def read_s3dis_format(area_id: str,
                      room_name: str,
                      data_root: str="./",
                      label_out: bool=True,
                      verbose: bool=False):
    r"""
    extract data from a room folder
    """
    room_type = room_name.split("_")[0]
    room_label = ROOM_TYPES[room_type]
    raw_path = osp.join(data_root, area_id, room_name, f"{room_name}.txt")
    
    # room_ver = pd.read_csv(raw_path, sep=" ", header=None).values
    # xyz = np.ascontiguousarray(room_ver[:, 0:3], dtype="float32")
    # try:
    #     rgb = np.ascontiguousarray(room_ver[:, 3:6], dtype="uint8")
    # except ValueError:
    #     rgb = np.zeros((room_ver.shape[0], 3), dtype="uint8")
    #     warn("WARN - corrupted rgb data for file %s" % raw_path)
    room_ver = np.genfromtxt(raw_path)
    xyz = np.ascontiguousarray(room_ver[:, 0:3], dtype="float32")
    rgb = np.ascontiguousarray(room_ver[:, 3:6], dtype="uint8")
    if not label_out:
        return xyz, rgb
    n_ver = len(room_ver)
    del room_ver
    # nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(xyz)
    tree = cKDTree(xyz)
    semantic_labels = np.zeros((n_ver,), dtype="int64")
    room_label = np.asarray([room_label])
    # instance_labels = np.zeros((n_ver,), dtype="int64")
    instance_labels = np.ones((n_ver,), dtype="int64") * -100
    objects = glob.glob(osp.join(data_root, area_id, room_name, "Annotations/*.txt"))
    i_object = 1
    for single_object in objects:
        object_name = os.path.splitext(os.path.basename(single_object))[0]
        if verbose:
            print("adding object " + str(i_object) + " : " + object_name)
        object_class = object_name.split("_")[0]
        object_label = object_name_to_label(object_class)
        obj_ver = pd.read_csv(single_object, sep=" ", header=None).values
        # _, obj_ind = nn.kneighbors(obj_ver[:, 0:3])
        _, obj_ind = tree.query(obj_ver[:, 0:3], k=1)
        obj_ind = np.array(obj_ind)
        semantic_labels[obj_ind] = object_label
        if object_label < 3: # background object
            continue
        instance_labels[obj_ind] = i_object
        i_object = i_object + 1

    return (
        xyz,
        rgb,
        semantic_labels,
        instance_labels,
        room_label,
    )



def get_parser():
    parser = argparse.ArgumentParser(description="s3dis data prepare")
    parser.add_argument("--data_split",
                        default="train",
                        help="data split (train / val)")
    parser.add_argument("--data_root",
                        default="./data",
                        help="root dir save data")

    args_cfg = parser.parse_args()

    return args_cfg


if __name__ == "__main__":
    args = get_parser()
    global split
    split = args.data_split
    data_root = args.data_root
    os.makedirs(split, exist_ok=True)

    if split == "train":
        area_list = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"]
    elif split == "val":
        area_list = ["Area_5"]
    else:
        raise ValueError(f"data_split must be 'train' or 'test', but got {split}")

    for area_id in area_list:
        print(f"Processing: {area_id}")
        area_dir = osp.join(data_root, area_id)
        # get the room name list for each area
        room_name_list = os.listdir(area_dir)
        room_name_list.remove(f"{area_id}_alignmentAngle.txt")
        room_name_list.remove(".DS_Store")
        for room_name in gorilla.track(room_name_list):
            scene = f"{area_id}_{room_name}"
            save_path = osp.join(split, scene + ".pth")
            if osp.exists(save_path):
                continue
            (xyz, rgb, semantic_labels, instance_labels, room_label) = read_s3dis_format(area_id, room_name, data_root)
            torch.save((xyz, rgb, semantic_labels, instance_labels, room_label, scene), save_path)
