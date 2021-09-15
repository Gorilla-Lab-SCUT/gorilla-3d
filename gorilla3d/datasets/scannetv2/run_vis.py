# Copyright (c) Gorilla-Lab. All rights reserved.
import argparse
from .visualize import get_coords_color, visualize_pts_rgb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",
                        help="path to the input dataset files",
                        default="../../data/scannetv2")
    parser.add_argument("--result-root",
                        help="path to the predicted results",
                        default=None)
    parser.add_argument("--save-dir",
                        help="path to save visual result",
                        default="./vis")
    parser.add_argument("--room-name",
                        help="room_name",
                        default="scene0707_00")
    parser.add_argument("--room-split",
                        help="train / val / test",
                        default="test")
    parser.add_argument(
        "--task",
        help=
        "input / semantic_gt / semantic_pred / instance_gt / instance_pred",
        default="instance_pred")
    args = parser.parse_args()

    print(args.room_name)

    xyz, rgb = get_coords_color(args.data_root, args.result_root,
                                args.room_split, args.room_name, args.task)

    visualize_pts_rgb(rgb,
                      args.room_name,
                      args.data_root,
                      args.save_dir,
                      mode=args.room_split)
