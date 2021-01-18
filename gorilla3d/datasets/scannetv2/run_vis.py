# Copyright (c) Gorilla-Lab. All rights reserved.
import argparse
from .visualize import get_coords_color, visualize_pts_rgb

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


