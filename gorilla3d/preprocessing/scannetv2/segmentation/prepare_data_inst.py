"""
Modified from SparseConvNet data preparation: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py
"""
import os
import os.path as osp
import json
import argparse
import multiprocessing as mp

import torch
import plyfile
import numpy as np

import gorilla

G_LABEL_NAMES = [
    "unannotated", "wall", "floor", "chair", "table", "desk", "bed",
    "bookshelf", "sofa", "sink", "bathtub", "toilet", "curtain", "counter",
    "door", "window", "shower curtain", "refridgerator", "picture", "cabinet",
    "otherfurniture"
]


def f_test(scene):
    fn = f"scans_test/{scene}/{scene}_vh_clean_2.ply"
    print(fn)

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1
    faces = np.array([list(x) for x in f.elements[1]])[:, 0, :]  # [nFaces, 3]
    faces = np.ascontiguousarray(faces)

    torch.save((coords, colors, faces, scene),
               osp.join(split, scene + "_inst_nostuff.pth"))
    print("Saving to " + osp.join(split, scene + "_inst_nostuff.pth"))


def f(scene):
    fn = f"scans/{scene}/{scene}_vh_clean_2.ply"
    fn2 = f"scans/{scene}/{scene}_vh_clean_2.labels.ply"
    fn3 = f"scans/{scene}/{scene}_vh_clean_2.0.010000.segs.json"
    fn4 = f"scans/{scene}/{scene}.aggregation.json"
    print(fn)

    save_path = osp.join(split, scene + "_inst_nostuff.pth")
    if osp.exists(save_path):
        return

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords_shift = -points[:, :3].mean(0)
    coords = np.ascontiguousarray(points[:, :3] + coords_shift)
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1
    faces = np.array([list(x) for x in f.elements[1]])[:, 0, :]  # [nFaces, 3]
    faces = np.ascontiguousarray(faces)

    f2 = plyfile.PlyData().read(fn2)
    semantic_labels = remapper[np.array(f2.elements[0]["label"])]

    with open(fn3) as jsondata:
        d = json.load(jsondata)
        seg = d["segIndices"]
    segid_to_pointid = {}
    for i in range(len(seg)):
        if seg[i] not in segid_to_pointid:
            segid_to_pointid[seg[i]] = []
        segid_to_pointid[seg[i]].append(i)

    instance_segids = []
    labels = []
    with open(fn4) as jsondata:
        d = json.load(jsondata)
        for x in d["segGroups"]:
            if g_raw2scannetv2[x["label"]] != "wall" and g_raw2scannetv2[
                    x["label"]] != "floor":
                instance_segids.append(x["segments"])
                labels.append(x["label"])
                assert (x["label"] in g_raw2scannetv2.keys())
    if (scene == "scene0217_00" and instance_segids[0] == instance_segids[int(
            len(instance_segids) / 2)]):
        instance_segids = instance_segids[:int(len(instance_segids) / 2)]
    check = []
    for i in range(len(instance_segids)):
        check += instance_segids[i]
    assert len(np.unique(check)) == len(check)

    instance_labels = np.ones(semantic_labels.shape[0]) * -100
    for i in range(len(instance_segids)):
        segids = instance_segids[i]
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        instance_labels[pointids] = i
        assert (len(np.unique(semantic_labels[pointids])) == 1)

    save_dir = split
    torch.save((coords, colors, faces, semantic_labels, instance_labels,
                coords_shift, scene), save_path)
    print("Saving to " + osp.join(split, scene + "_inst_nostuff.pth"))


def get_parser():
    parser = argparse.ArgumentParser(description="ScanNet data prepare")
    parser.add_argument("--data-split",
                        default="test",
                        help="data split (train / val / test)")

    args_cfg = parser.parse_args()

    return args_cfg


if __name__ == "__main__":
    args = get_parser()

    meta_data_dir = osp.join(osp.dirname(__file__), "..", "meta_data")

    def get_raw2scannetv2_label_map():
        lines = [
            line.rstrip() for line in open(
                osp.join(meta_data_dir, "scannetv2-labels.combined.tsv"))
        ]
        lines_0 = lines[0].split("\t")
        print(lines_0)
        print(len(lines))
        lines = lines[1:]
        raw2scannet = {}
        for i in range(len(lines)):
            label_classes_set = set(G_LABEL_NAMES)
            elements = lines[i].split("\t")
            raw_name = elements[1]
            if (elements[1] != elements[2]):
                print(f"{i}: {elements[1]} {elements[2]}")
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                raw2scannet[raw_name] = "unannotated"
            else:
                raw2scannet[raw_name] = nyu40_name
        return raw2scannet

    g_raw2scannetv2 = get_raw2scannetv2_label_map()

    # Map relevant classes to {0,1,...,19}, and ignored classes to -100
    remapper = np.ones(150) * (-100)
    for i, x in enumerate([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36,
            39
    ]):
        remapper[x] = i

    global split
    split = args.data_split
    os.makedirs(split, exist_ok=True)
    print(f"data split: {split}")
    scene_list = gorilla.list_from_file(
        osp.join(meta_data_dir, f"scannetv2_{split}.txt"))

    files = sorted(
        list(map(lambda x: f"scans/{x}/{x}_vh_clean_2.ply", scene_list)))

    if split != "test":
        files2 = sorted(
            list(
                map(lambda x: f"scans/{x}/{x}_vh_clean_2.labels.ply",
                    scene_list)))
        files3 = sorted(
            list(
                map(lambda x: f"scans/{x}/{x}_vh_clean_2.0.010000.segs.json",
                    scene_list)))
        files4 = sorted(
            list(map(lambda x: f"scans/{x}/{x}.aggregation.json", scene_list)))
        assert len(files) == len(files2)
        assert len(files) == len(files3)
        assert len(files) == len(files4), f"{len(files)} {len(files4)}"

    p = mp.Pool(processes=mp.cpu_count())
    if split == "test":
        p.map(f_test, scene_list)
    else:
        p.map(f, scene_list)
    p.close()
    p.join()
