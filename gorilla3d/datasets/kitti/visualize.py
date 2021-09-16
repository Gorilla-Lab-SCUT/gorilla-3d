# Copyright (c) Gorilla-Lab. All rights reserved.
import os
from typing import Dict

import numpy as np
from ...utils.pc import save_pc

COLORSEMANTIC = np.array([
    [0, 0, 0],  # "unlabeled"     rgb(0,   0,   0)
    [245, 150, 100],  # "car"           rgb(245, 150, 100)
    [245, 230, 100],  # "bicycle"       rgb(245, 230, 100)
    [150, 60, 30],  # "motorcycle"    rgb(150,  60,  30)
    [180, 30, 80],  # "truck"         rgb(180,  30,  80)
    [250, 80, 100],  # "other-vihicle" rgb(250,  80, 100)
    [30, 30, 255],  # "person"        rgb(30,  30, 255)
    [200, 40, 255],  # "bicyclist"     rgb(200,  40, 255)
    [90, 30, 150],  # "motorcycle"    rgb(90,  30, 150)
    [255, 0, 255],  # "road"          rgb(255,   0, 255)
    [255, 150, 255],  # "parking"       rgb(255, 150, 255)
    [75, 0, 75],  # "sidewalk"      rgb(75,   0,  75)
    [75, 0, 175],  # "other-ground"  rgb(75,   0, 175)
    [0, 200, 255],  # "building"      rgb(0, 200, 255)
    [50, 120, 255],  # "fence"         rgb(50, 120, 255)
    [0, 175, 0],  # "vegetatiojn"   rgb(0, 175,   0)
    [0, 60, 135],  # "trunk"         rgb(0,  60, 135)
    [80, 240, 150],  # "terrain"       rgb(80, 240, 150)
    [150, 240, 255],  # "pole"          rgb(150, 240, 255)
    [0, 0, 255]  # "traffic-sign"  rgb(0,   0, 255)
])


def save_label_scene(pc_dir: str, label_dir: str, scene_id: str,
                     label_mapper: np.vectorize, save_path: str):
    pc_file = os.path.join(pc_dir, f"{scene_id}.bin")
    label_file = os.path.join(label_dir, f"{scene_id}.label")
    pc = np.fromfile(pc_file, dtype=np.float32).reshape(
        (-1, 4))[:, :3]  # [N, 3]
    label = np.fromfile(label_file, dtype=np.int32).reshape(-1)  # [N]
    label = label & 0xFFFF  # delete high 16 digits binary
    label = label_mapper(label)  # annotated id map
    colors = COLORSEMANTIC[label]  # [N, 3]
    save_pc(pc, colors, save_path)
