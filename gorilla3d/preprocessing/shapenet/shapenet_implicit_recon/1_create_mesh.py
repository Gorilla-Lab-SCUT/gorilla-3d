# Copyright (c) Gorilla-Lab. All rights reserved.
import argparse
import os
import time
import numpy as np
import trimesh
import mcubes

SRC_DATASET_DIR = None
DST_DATASET_DIR = None
CLASS_NAME = None
SPLIT = None
SPLIT_DIR = None
SDF_COMMAND = None
RES = None
OVERRIDE = None


def get_src_path(class_name, obj_name, file_name):
    return os.path.join(SRC_DATASET_DIR, class_name, obj_name, file_name)


def get_save_path(class_name, obj_name, file_name):
    save_dir = os.path.join(DST_DATASET_DIR, class_name, obj_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return os.path.join(save_dir, file_name)


def create_bsdf_file(obj_file, bsdf_file, g=0.0):
    """ create sdf file (binary)
    """
    command_str = f"{SDF_COMMAND} {obj_file} {RES} {RES} {RES} -s -e 1.2 -o {bsdf_file} -m 1"
    if g > 0.0:
        command_str = f"{command_str} -g {g}"
    os.system(command_str)


def get_sdf(bsdf_file, res=RES):
    """ load sdf file to get sdf
    """
    intsize = 4
    floatsize = 8
    sdf = {"param": [], "value": []}
    with open(bsdf_file, "rb") as f:
        try:
            bytes = f.read()
            ress = np.frombuffer(bytes[:intsize * 3], dtype=np.int32)
            if -1 * ress[0] != res or ress[1] != res or ress[2] != res:
                raise Exception(f"resolution error: {bsdf_file}")
            positions = np.frombuffer(bytes[intsize * 3:intsize * 3 +
                                            floatsize * 6],
                                      dtype=np.float64)
            # bottom left corner, x,y,z and top right corner, x, y, z
            sdf["param"] = [
                positions[0], positions[1], positions[2], positions[3],
                positions[4], positions[5]
            ]  # min, min, min, max, max, max
            sdf["param"] = np.float32(sdf["param"])
            sdf["value"] = np.frombuffer(bytes[intsize * 3 + floatsize * 6:],
                                         dtype=np.float32)
            sdf["value"] = np.reshape(sdf["value"],
                                      (res + 1, res + 1, res + 1))
        finally:
            f.close()
    return sdf


##############################################################################################
def process(class_name, obj_name, override=False):

    st = time.time()

    mesh_load_path = get_src_path(class_name, obj_name, "model.obj")
    bsdf_save_path = f"dense_sdf_{class_name}_{obj_name}.sdf"
    mesh_gt_path = get_save_path(class_name, obj_name,
                                 "mesh_gt.ply")  # it is not normalized

    # marching cubes gt mesh
    if (not os.path.exists(mesh_gt_path)) or override:
        create_bsdf_file(mesh_load_path, bsdf_save_path)
        sdf_info = get_sdf(bsdf_save_path, RES)

        vertices, triangles = mcubes.marching_cubes(sdf_info["value"], 0.0)
        vertices[:, 0] = (vertices[:, 0] / RES) * (
            sdf_info["param"][3] - sdf_info["param"][0]) + sdf_info["param"][0]
        vertices[:, 1] = (vertices[:, 1] / RES) * (
            sdf_info["param"][4] - sdf_info["param"][1]) + sdf_info["param"][1]
        vertices[:, 2] = (vertices[:, 2] / RES) * (
            sdf_info["param"][5] - sdf_info["param"][2]) + sdf_info["param"][2]
        vertices[:, [0, 2]] = vertices[:, [2, 0]]
        trimesh.Trimesh(vertices=vertices,
                        faces=triangles).export(mesh_gt_path,
                                                encoding="binary")
        print(f"save gt mesh to: {mesh_gt_path}")
        del sdf_info
    else:
        print(f"Skip: {mesh_gt_path}")

    if os.path.exists(bsdf_save_path):
        os.system(f"rm -rf {bsdf_save_path}")
        print(f"delete: {bsdf_save_path}")

    print(f"total time = {time.time() - st}")
    print(
        "===================================================================")
    print()


def auto_process():
    with open(os.path.join(SPLIT_DIR, f"{CLASS_NAME}_{SPLIT}.lst")) as f:
        object_list = f.readlines()
    object_list = [s.strip() for s in object_list if s.strip() != ""]

    for obj_name in object_list:
        process(CLASS_NAME, obj_name, OVERRIDE)


########################################################################################################

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
    parser.add_argument("--res", type=int, default=512, help="GT Resolution")
    parser.add_argument("--src_dataset_dir",
                        type=str,
                        help="Path to the unzipped `ShapeNetCore.v1` folder")
    parser.add_argument("--dst_dataset_dir",
                        type=str,
                        help="Path to a folder to save result")
    parser.add_argument("--split_dir",
                        type=str,
                        help="Path to the folder to save split files")
    parser.add_argument("--sdf_executable",
                        type=str,
                        help="Path to the computeDistanceField executable")
    parser.add_argument("--override",
                        action="store_true",
                        help="Overriding existing files")
    args = parser.parse_args()

    SPLIT = args.split
    RES = args.res
    SRC_DATASET_DIR = args.src_dataset_dir
    DST_DATASET_DIR = args.dst_dataset_dir
    SPLIT_DIR = args.split_dir
    SDF_COMMAND = args.sdf_executable
    OVERRIDE = args.override

    if len(args.class_name) == 1:
        CLASS_NAME = args.class_name[0]
        auto_process()

    else:
        for class_name in args.class_name:
            os.system(
                f"python {__file__} --class_name {class_name} --split {SPLIT} --res {RES} "
                f"--src_dataset_dir {SRC_DATASET_DIR} --dst_dataset_dir {DST_DATASET_DIR} "
                f"--split_dir {SPLIT_DIR} --sdf_executable {SDF_COMMAND}" +
                (" --override" if OVERRIDE else ""))

    print("All done.")
