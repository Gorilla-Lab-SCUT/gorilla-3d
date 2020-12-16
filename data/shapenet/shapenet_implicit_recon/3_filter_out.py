""" move those unqualified meshes (too small) into another folder
"""

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


def is_okey(filePath, minMB):
    fsize = os.path.getsize(filePath)
    fsize = fsize / float(1024 * 1024)  # MB
    return (fsize >= minMB)


def auto_process():
    dst_class_dir = os.path.join(MOVED_DATASET_DIR, CLASS_NAME)
    try_make_dirs(dst_class_dir)
    class_dir = os.path.join(SRC_DATASET_DIR, CLASS_NAME)

    with open(os.path.join(LOAD_SPLIT_DIR, f'{CLASS_NAME}_{SPLIT}.lst'), 'r') as f:
        obj_list = f.readlines()
    obj_list = [s.strip() for s in obj_list if s.strip() != '']
    final_obj_list = []

    for i, obj_name in enumerate(obj_list):
        filePath = os.path.join(class_dir, obj_name, MESH_NAME)
        if is_okey(filePath, MINMB):
            print(f"file[{i}/{len(obj_list)}]: {filePath} is okey!")
            final_obj_list.append(obj_name)
        else:
            print(f"file[{i}/{len(obj_list)}]: {filePath} is not okey!")
            src_dir = os.path.join(class_dir, obj_name)
            shutil.move(src_dir, dst_class_dir)
            print(f"dir: {src_dir} is moved to folder: {dst_class_dir}")
        print("======================================")

    try_make_dirs(SAVE_SPLIT_DIR)
    save_split_file_path = os.path.join(SAVE_SPLIT_DIR, f'{CLASS_NAME}_{SPLIT}.lst')
    with open(save_split_file_path, 'w') as f:
        f.writelines([f"{s}\n" for s in final_obj_list])
    print(f"save split file to: {save_split_file_path}")
    print(f'Exits.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_name', type=str, default='03001627')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--src_dataset_dir', type=str, default='/data/lab-lei.jiabao/ShapeNet_GT/ShapenetV1_tpami')
    parser.add_argument('--moved_dataset_dir',
                        type=str,
                        default='/data/lab-lei.jiabao/ShapeNet_GT/ShapenetV1_tpami_unqualified')
    parser.add_argument('--load_split_dir', type=str, default='/data/lab-lei.jiabao/ShapeNet_Split/')
    parser.add_argument('--save_split_dir', type=str, default='/data/lab-lei.jiabao/ShapeNet_GT/ShapenetV1_tpami/split')
    parser.add_argument('--mesh_name', type=str, default='mesh_gt.ply')
    parser.add_argument('--minMB', type=float, default=2.0)  # should be >= 2.0MB
    args = parser.parse_args()

    CLASS_NAME = args.class_name
    SPLIT = args.split
    SRC_DATASET_DIR = args.src_dataset_dir
    MOVED_DATASET_DIR = args.moved_dataset_dir
    LOAD_SPLIT_DIR = args.load_split_dir
    SAVE_SPLIT_DIR = args.save_split_dir
    MESH_NAME = args.mesh_name
    MINMB = args.minMB

    auto_process()
