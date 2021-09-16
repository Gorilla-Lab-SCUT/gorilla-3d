# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import argparse
import shutil

CLASS_NAME = None
SPLIT = None
SRC_DATASET_DIR = None
MOVED_DATASET_DIR = None
LOAD_SPLIT_DIR = None
SAVE_SPLIT_DIR = None
MESH_NAME = None
MINMB = None


def try_make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def is_okey(file_path, minMB):
    fsize = os.path.getsize(file_path)
    fsize = fsize / float(1024 * 1024)  # MB
    return (fsize >= minMB)


def auto_process():
    dst_class_dir = os.path.join(MOVED_DATASET_DIR, CLASS_NAME)
    try_make_dirs(dst_class_dir)
    class_dir = os.path.join(SRC_DATASET_DIR, CLASS_NAME)

    with open(os.path.join(LOAD_SPLIT_DIR, f"{CLASS_NAME}_{SPLIT}.lst"),
              "r") as f:
        obj_list = f.readlines()
    obj_list = [s.strip() for s in obj_list if s.strip() != ""]
    final_obj_list = []

    for i, obj_name in enumerate(obj_list):
        file_path = os.path.join(class_dir, obj_name, MESH_NAME)
        if is_okey(file_path, MINMB):
            print(f"file[{i}/{len(obj_list)}]: {file_path} is okey!")
            final_obj_list.append(obj_name)
        else:
            print(f"file[{i}/{len(obj_list)}]: {file_path} is not okey!")
            src_dir = os.path.join(class_dir, obj_name)
            shutil.move(src_dir, dst_class_dir)
            print(f"dir: {src_dir} is moved to folder: {dst_class_dir}")
        print("======================================")

    try_make_dirs(SAVE_SPLIT_DIR)
    save_split_file_path = os.path.join(SAVE_SPLIT_DIR,
                                        f"{CLASS_NAME}_{SPLIT}.lst")
    with open(save_split_file_path, "w") as f:
        f.writelines([f"{s}\n" for s in final_obj_list])
    print(f"save split file to: {save_split_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name",
                        type=str,
                        nargs="+",
                        default=["03001627"],
                        help="Categories to process")
    parser.add_argument("--split",
                        type=str,
                        default="train",
                        help="Which split")
    parser.add_argument("--src_dataset_dir",
                        type=str,
                        help="The folder containing simplified meshes")
    parser.add_argument(
        "--moved_dataset_dir",
        type=str,
        help="The folder to save those unqualified meshes (move to this folder)"
    )
    parser.add_argument("--load_split_dir",
                        type=str,
                        help="The original split folder")
    parser.add_argument(
        "--save_split_dir",
        type=str,
        help="The target split folder (new split file is save to here)")
    parser.add_argument("--mesh_name",
                        type=str,
                        default="mesh_gt.ply",
                        help="Which mesh file is the criterion")
    parser.add_argument(
        "--minMB",
        type=float,
        default=2.0,
        help=
        "The size of qualified mesh file should be larger than this value (MB)"
    )  # should be >= 2.0MB
    args = parser.parse_args()

    if len(args.class_name) == 1:

        CLASS_NAME = args.class_name[0]
        SPLIT = args.split
        SRC_DATASET_DIR = args.src_dataset_dir
        MOVED_DATASET_DIR = args.moved_dataset_dir
        LOAD_SPLIT_DIR = args.load_split_dir
        SAVE_SPLIT_DIR = args.save_split_dir
        MESH_NAME = args.mesh_name
        MINMB = args.minMB

        auto_process()

    else:
        for class_name in args.class_name:
            os.system(
                f"python {__file__} --class_name {class_name} --split {args.split} --src_dataset_dir {args.src_dataset_dir} "
                f"--moved_dataset_dir {args.moved_dataset_dir} --load_split_dir {args.load_split_dir} "
                f"--save_split_dir {args.save_split_dir} --mesh_name {args.mesh_name} --minMB {args.minMB}"
            )

    print("All done.")
