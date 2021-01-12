# modify from PointGroup
# Written by Li Jiang
import os
import os.path as osp
import argparse
import logging
from typing import Optional
from operator import itemgetter
from copy import deepcopy

import torch
import numpy as np
import open3d as o3d

COLOR20 = np.array(
        [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
        [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
        [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
        [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128]])

COLOR40 = np.array(
        [[88,170,108], [174,105,226], [78,194,83], [198,62,165], [133,188,52], [97,101,219], [190,177,52], [139,65,168], [75,202,137], [225,66,129],
        [68,135,42], [226,116,210], [146,186,98], [68,105,201], [219,148,53], [85,142,235], [212,85,42], [78,176,223], [221,63,77], [68,195,195],
        [175,58,119], [81,175,144], [184,70,74], [40,116,79], [184,134,219], [130,137,46], [110,89,164], [92,135,74], [220,140,190], [94,103,39],
        [144,154,219], [160,86,40], [67,107,165], [194,170,104], [162,95,150], [143,110,44], [146,72,105], [225,142,106], [162,83,86], [227,124,143]])

SEMANTIC_IDXS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
SEMANTIC_NAMES = np.array([
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter",
    "desk", "curtain", "refridgerator", "shower curtain", "toilet", "sink", "bathtub", "otherfurniture"])
CLASS_COLOR = {
    "unannotated": [0, 0, 0],
    "floor": [143, 223, 142],
    "wall": [171, 198, 230],
    "cabinet": [0, 120, 177],
    "bed": [255, 188, 126],
    "chair": [189, 189, 57],
    "sofa": [144, 86, 76],
    "table": [255, 152, 153],
    "door": [222, 40, 47],
    "window": [197, 176, 212],
    "bookshelf": [150, 103, 185],
    "picture": [200, 156, 149],
    "counter": [0, 190, 206],
    "desk": [252, 183, 210],
    "curtain": [219, 219, 146],
    "refridgerator": [255, 127, 43],
    "bathtub": [234, 119, 192],
    "shower curtain": [150, 218, 228],
    "toilet": [0, 160, 55],
    "sink": [110, 128, 143],
    "otherfurniture": [80, 83, 160]}
SEMANTIC_IDX2NAME = {
    1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair", 6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf", 11: "picture",
    12: "counter", 14: "desk", 16: "curtain", 24: "refridgerator", 28: "shower curtain", 33: "toilet",  34: "sink", 36: "bathtub", 39: "otherfurniture"}


def visualize_instance_mask(clusters: np.ndarray,
                            room_name: str,
                            visual_dir: str,
                            data_root: str,
                            logger: Optional[logging.Logger]=None,
                            cluster_scores: Optional[np.ndarray]=None,
                            semantic_pred: Optional[np.ndarray]=None,
                            color: int=20,):
    assert color in [20, 40]
    colors = globals()["COLOR{}".format(color)]
    mesh_file = osp.join(data_root, room_name, room_name + "_vh_clean_2.ply")
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    pred_mesh = deepcopy(mesh)
    points = np.array(pred_mesh.vertices)
    inst_label_pred_rgb = np.zeros_like(points)  # np.ones(rgb.shape) * 255 #
    for cluster_id, cluster in enumerate(clusters):
        if logger is not None:
            # NOTE: remove the handlers are not FileHandler to avoid 
            #       outputing this message on console(StreamHandler)
            #       and final will recover the handlers of logger
            handler_storage = []
            for handler in logger.handlers:
                if not isinstance(handler, logging.FileHandler):
                    handler_storage.append(handler)
                    logger.removeHandler(handler)
            message = "{:<4}: pointnum: {:<7} ".format(cluster_id, int(cluster.sum()))
            if semantic_pred is not None:
                semantic_label = np.argmax(np.bincount(semantic_pred[np.where(cluster == 1)[0]]))
                semantic_id = int(SEMANTIC_IDXS[semantic_label])
                semantic_name = SEMANTIC_IDX2NAME[semantic_id]
                message += "semantic: {:<3}-{:<15} ".format(semantic_id, semantic_name)
            if cluster_scores is not None:
                score = float(cluster_scores[cluster_id])
                message += "score: {:.4f} ".format(score)
            logger.info(message)
            for handler in handler_storage:
                logger.addHandler(handler)
        inst_label_pred_rgb[cluster == 1] = colors[cluster_id % len(colors)]
    rgb = inst_label_pred_rgb

    pred_mesh.vertex_colors = o3d.utility.Vector3dVector(rgb / 255)
    points[:, 1] += (points[:, 1].max() + 0.5)
    pred_mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh += pred_mesh
    o3d.io.write_triangle_mesh(osp.join(visual_dir, room_name+".ply"), mesh)

# TODO: add the semantic visualization


def visualize_pts_rgb(rgb, room_name):
    mesh_file = osp.join("..", "..", "data", "scannetv2", "scans_test", room_name, room_name + "_vh_clean_2.ply")
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    pred_mesh = deepcopy(mesh)
    pred_mesh.vertex_colors = o3d.utility.Vector3dVector(rgb / 255)
    points = np.array(pred_mesh.vertices)
    # points[:, 2] += 3
    points[:, 1] += (points[:, 1].max() + 0.5)
    pred_mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh += pred_mesh
    o3d.io.write_triangle_mesh(osp.join(".", "vis", room_name+".ply"), mesh)


def get_coords_color(opt):
    input_file = os.path.join(opt.data_root, opt.room_split, opt.room_name + "_inst_nostuff.pth")
    assert os.path.isfile(input_file), "File not exist - {}.".format(input_file)
    if "test" in opt.room_split:
        xyz, rgb, edges, scene_idx = torch.load(input_file)
    else:
        xyz, rgb, label, inst_label = torch.load(input_file)
    rgb = (rgb + 1) * 127.5

    if (opt.task == "semantic_gt"):
        assert "test" not in opt.room_split
        label = label.astype(np.int)
        label_rgb = np.zeros(rgb.shape)
        label_rgb[label >= 0] = np.array(itemgetter(*SEMANTIC_NAMES[label[label >= 0]])(CLASS_COLOR))
        rgb = label_rgb

    elif (opt.task == "instance_gt"):
        assert "test" not in opt.room_split
        inst_label = inst_label.astype(np.int)
        print("Instance number: {}".format(inst_label.max() + 1))
        inst_label_rgb = np.zeros(rgb.shape)
        object_idx = (inst_label >= 0)
        inst_label_rgb[object_idx] = COLOR20[inst_label[object_idx] % len(COLOR20)]
        rgb = inst_label_rgb

    elif (opt.task == "semantic_pred"):
        assert opt.room_split != "train"
        semantic_file = os.path.join(opt.result_root, opt.room_split, "semantic", opt.room_name + ".npy")
        assert os.path.isfile(semantic_file), "No semantic result - {}.".format(semantic_file)
        label_pred = np.load(semantic_file).astype(np.int)  # 0~19
        label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb

    elif (opt.task == "instance_pred"):
        assert opt.room_split != "train"
        instance_file = os.path.join(opt.result_root, opt.room_split, opt.room_name + ".txt")
        assert os.path.isfile(instance_file), "No instance result - {}.".format(instance_file)
        f = open(instance_file, "r")
        masks = f.readlines()
        masks = [mask.rstrip().split() for mask in masks]
        inst_label_pred_rgb = np.zeros(rgb.shape)  # np.ones(rgb.shape) * 255 #
        for i in range(len(masks) - 1, -1, -1):
            mask_path = os.path.join(opt.result_root, opt.room_split, masks[i][0])
            assert os.path.isfile(mask_path), mask_path
            if (float(masks[i][2]) < 0.09):
                continue
            mask = np.loadtxt(mask_path).astype(np.int)
            print("{} {}: {} pointnum: {}".format(i, masks[i], SEMANTIC_IDX2NAME[int(masks[i][1])], mask.sum()))
            inst_label_pred_rgb[mask == 1] = COLOR20[i % len(COLOR20)]
        rgb = inst_label_pred_rgb

    if "test" not in opt.room_split:
        sem_valid = (label != -100)
        xyz = xyz[sem_valid]
        rgb = rgb[sem_valid]

    return xyz, rgb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", help="path to the input dataset files", default="../../data/scannetv2")
    parser.add_argument("--result_root", help="path to the predicted results", default="../../log/bn_overseg/result/epoch512_nmst0.3_scoret0.009_npointt100")
    parser.add_argument("--room_name", help="room_name", default="scene0707_00")
    parser.add_argument("--room_split", help="train / val / test", default="test")
    parser.add_argument("--task", help="input / semantic_gt / semantic_pred / instance_gt / instance_pred", default="instance_pred")
    opt = parser.parse_args()

    print(opt.room_name)

    xyz, rgb = get_coords_color(opt)

    visualize_pts_rgb(rgb, opt.room_name)

