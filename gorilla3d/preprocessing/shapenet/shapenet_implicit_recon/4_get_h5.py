# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import time
import trimesh
import argparse
import numpy as np
from scipy.spatial import cKDTree
import point_cloud_utils as pcu
import h5py
import cv2

CLASS_NAME = None
SPLIT = None
SRC_DATASET_DIR = None
DST_DATASET_DIR = None
IMG_DATASET_DIR = None
SPLIT_DIR = None
LOAD_MESH_NAME = None
IMAGES_NUM = None
RAND_SAMPLING_NUM = None
POISSON_SAMPLING_NUM = None
SDF_METHOD = None
SDF_APPROX_SAMPLING_NUM = None
NEAR_STD = None
NEAR_NUM = None
FAR_STD = None
FAR_NUM = None
UNI_SCALE = None
UNI_NUM = None
N_JOBS = None
OVERRIDE = None

##############################################################################################


def process(class_name, obj_name):

    st = time.time()

    load_mesh_path = os.path.join(SRC_DATASET_DIR, class_name, obj_name,
                                  LOAD_MESH_NAME)
    mesh = trimesh.load_mesh(load_mesh_path)
    samples, face_index = trimesh.sample.sample_surface(
        mesh, count=SDF_APPROX_SAMPLING_NUM)
    normals = mesh.face_normals[face_index]
    tree = cKDTree(samples, compact_nodes=False, balanced_tree=False)

    def normals_method(num_, std_, in_pts=None):
        if in_pts is None:
            index = np.random.choice(len(samples), size=num_)
            pts = samples[index] + np.random.normal(
                loc=0.0, scale=std_, size=[num_, 3])
        else:
            pts = in_pts
        _, idx = tree.query(pts, k=1, n_jobs=N_JOBS)
        sdf = ((pts - samples[idx]) * normals[idx]).sum(axis=1).reshape(
            [-1, 1])
        pts_sdf = np.concatenate([pts, sdf], axis=1)
        return pts_sdf

    def nearest_method(num_, std_, in_pts=None):
        if in_pts is None:
            index = np.random.choice(len(samples), size=num_)
            pts = samples[index] + np.random.normal(
                loc=0.0, scale=std_, size=[num_, 3])
        else:
            pts = in_pts
        _, idx = tree.query(pts, k=1, n_jobs=N_JOBS)
        sdf = np.linalg.norm(
            pts - samples[idx], axis=1, keepdims=True) * np.sign(
                ((pts - samples[idx]) * normals[idx]).sum(axis=1).reshape(
                    [-1, 1]))
        pts_sdf = np.concatenate([pts, sdf], axis=1)
        return pts_sdf

    method = normals_method if SDF_METHOD == "normals" else nearest_method

    near_pts_sdf = method(NEAR_NUM, NEAR_STD)
    far_pts_sdf = method(FAR_NUM, FAR_STD)

    ctr = np.mean(samples, axis=0)  # center
    mag = np.max(np.linalg.norm(samples - ctr, axis=1))  # magnitude
    ctrmag = np.array([*ctr, mag])

    uni_bb_min = ctr - np.full([3], fill_value=mag) * UNI_SCALE
    uni_bb_max = ctr + np.full([3], fill_value=mag) * UNI_SCALE
    uni_pts = np.random.rand(UNI_NUM,
                             3) * (uni_bb_max - uni_bb_min) + uni_bb_min
    uni_pts_sdf = method(None, None, uni_pts)

    pts_sdf = np.concatenate([near_pts_sdf, far_pts_sdf, uni_pts_sdf], axis=0)
    np.random.shuffle(pts_sdf)

    ################################################################
    v, f, n, _ = pcu.read_ply(load_mesh_path)
    v_poisson, n_poisson = pcu.sample_mesh_poisson_disk(
        v, f, n, POISSON_SAMPLING_NUM, use_geodesic_distance=True)
    if v_poisson.shape[0] >= POISSON_SAMPLING_NUM:
        v_poisson = v_poisson[:POISSON_SAMPLING_NUM]
    else:
        v_poisson = np.concatenate([
            v_poisson, v_poisson[:(POISSON_SAMPLING_NUM - v_poisson.shape[0])]
        ],
                                   axis=0)
    np.random.shuffle(v_poisson)
    ################################################################
    rand_pts = samples[np.random.choice(samples.shape[0],
                                        size=RAND_SAMPLING_NUM,
                                        replace=False)]
    ################################################################
    if IMAGES_NUM > 0:
        imgs = np.empty([IMAGES_NUM, 137, 137, 3], dtype=np.uint8)
        for index in range(IMAGES_NUM):
            image_path = os.path.join(IMG_DATASET_DIR, class_name, obj_name,
                                      "rendering",
                                      f"{str(index).zfill(2)}.png")
            img = cv2.imread(image_path)  # 0~255 HWC uint8
            imgs[index] = img
        imgs = np.transpose(imgs, [0, 3, 1, 2])  # BHWC -> BCHW
    else:
        imgs = None
    ################################################################

    results_dict = dict(
        pts_sdf=pts_sdf,
        ctrmag=ctrmag,
        poisson_pts=v_poisson,
        rand_pts=rand_pts,
        imgs=imgs,
    )

    print(f"time = {(time.time() - st):.3f} ({class_name} / {obj_name})")
    print(
        "===================================================================")
    print(flush=True)

    return results_dict


##############################################################################################


def auto_process():
    with open(os.path.join(SPLIT_DIR, f"{CLASS_NAME}_{SPLIT}.lst")) as f:
        object_list = f.readlines()
    object_list = [s.strip() for s in object_list if s.strip() != ""]

    save_h5_path = os.path.join(DST_DATASET_DIR, f"{CLASS_NAME}_{SPLIT}.h5")
    print(f"we are going to save to: {save_h5_path}")
    if not os.path.exists(save_h5_path):
        f = h5py.File(save_h5_path, "w")
        f.create_dataset("obj_list", data=np.array(object_list, dtype="S"))
        dset_pts_sdf = f.create_dataset(
            "pts_sdf",
            shape=(0, NEAR_NUM + FAR_NUM + UNI_NUM, 4),
            maxshape=(None, NEAR_NUM + FAR_NUM + UNI_NUM, 4),
            chunks=(1, NEAR_NUM + FAR_NUM + UNI_NUM, 4),
            dtype="f2",
            compression="gzip",
            compression_opts=4,
        )
        dset_poisson_pts = f.create_dataset(
            "poisson_pts",
            shape=(0, POISSON_SAMPLING_NUM, 3),
            maxshape=(None, POISSON_SAMPLING_NUM, 3),
            chunks=(1, POISSON_SAMPLING_NUM, 3),
            dtype="f2",
            compression="gzip",
            compression_opts=4,
        )
        dset_rand_pts = f.create_dataset(
            "rand_pts",
            shape=(0, RAND_SAMPLING_NUM, 3),
            maxshape=(None, RAND_SAMPLING_NUM, 3),
            chunks=(1, RAND_SAMPLING_NUM, 3),
            dtype="f2",
            compression="gzip",
            compression_opts=4,
        )
        dset_ctrmag = f.create_dataset(
            "ctrmag",
            shape=(0, 4),
            maxshape=(None, 4),
            chunks=(1, 4),
            dtype="f4",
            compression="gzip",
            compression_opts=4,
        )
        if IMAGES_NUM > 0:
            dset_imgs = f.create_dataset(
                "imgs",
                shape=(0, IMAGES_NUM, 3, 137, 137),
                maxshape=(None, IMAGES_NUM, 3, 137, 137),
                chunks=(1, 1, 3, 137, 137),
                dtype="u1",
                compression="gzip",
                compression_opts=4,
            )
        f.close()

    for i, obj_name in enumerate(object_list):
        f = h5py.File(save_h5_path, "a")
        dset_pts_sdf = f["pts_sdf"]
        dset_poisson_pts = f["poisson_pts"]
        dset_rand_pts = f["rand_pts"]
        dset_ctrmag = f["ctrmag"]
        if IMAGES_NUM > 0:
            dset_imgs = f["imgs"]
        if i >= dset_pts_sdf.shape[0]:
            results_dict = process(CLASS_NAME, obj_name)

            dset_pts_sdf.resize(
                (dset_pts_sdf.shape[0] + 1, NEAR_NUM + FAR_NUM + UNI_NUM, 4))
            dset_pts_sdf[-1, :, :] = results_dict["pts_sdf"].astype(np.float16)

            dset_poisson_pts.resize(
                (dset_poisson_pts.shape[0] + 1, POISSON_SAMPLING_NUM, 3))
            dset_poisson_pts[-1, :, :] = results_dict["poisson_pts"].astype(
                np.float16)

            dset_rand_pts.resize(
                (dset_rand_pts.shape[0] + 1, RAND_SAMPLING_NUM, 3))
            dset_rand_pts[-1, :, :] = results_dict["rand_pts"].astype(
                np.float16)

            dset_ctrmag.resize((dset_ctrmag.shape[0] + 1, 4))
            dset_ctrmag[-1, :] = results_dict["ctrmag"].astype(np.float32)

            if IMAGES_NUM > 0:
                dset_imgs.resize(
                    (dset_imgs.shape[0] + 1, IMAGES_NUM, 3, 137, 137))
                dset_imgs[-1, :, :, :, :] = results_dict["imgs"].astype(
                    np.uint8)
        else:
            print(f"Skip: {CLASS_NAME} / {obj_name}", flush=True)
        f.close()


##############################################################################################

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
                        help="Path to load mesh")
    parser.add_argument("--dst_dataset_dir",
                        type=str,
                        help="Path to save h5 file")
    parser.add_argument("--img_dataset_dir",
                        type=str,
                        help="Path to the ShapeNetRendering folder")
    parser.add_argument("--split_dir",
                        type=str,
                        help="Path to the split folder")
    parser.add_argument("--load_mesh_name",
                        type=str,
                        default="mesh_gt_simplified.ply",
                        help="Mesh name to load")
    ###################################################
    parser.add_argument("--images_num",
                        type=int,
                        default=24,
                        help="Images num to store to h5 file"
                        )  # set to <=0 (e.g. -1) to disable storing images
    ###################################################
    parser.add_argument("--rand_sampling_num",
                        type=int,
                        default=100000,
                        help="Num of randomly sampled points on the surface")
    ###################################################
    parser.add_argument("--poisson_sampling_num",
                        type=int,
                        default=10000,
                        help="Num of poisson disk sampling surface points")
    ###################################################
    # SDF
    parser.add_argument(
        "--sdf_method",
        type=str,
        default="nearest",
        help="Method to calculate sdf value")  # normals / nearest
    parser.add_argument(
        "--sdf_approx_sampling_num",
        type=int,
        default=3000000,
        help=
        "Num of surface points to help calculate sdf value (will not store to h5 file)"
    )
    parser.add_argument("--near_std",
                        type=float,
                        default=0.01,
                        help="Gaussian std for near locations")
    parser.add_argument("--near_num",
                        type=int,
                        default=300000,
                        help="Num of near points")
    parser.add_argument("--far_std",
                        type=float,
                        default=0.1,
                        help="Gaussian std for far locations")
    parser.add_argument("--far_num",
                        type=int,
                        default=100000,
                        help="Num of far points")
    parser.add_argument("--uni_scale",
                        type=float,
                        default=1.5,
                        help="Scale of the uniform sampling bounding box")
    parser.add_argument("--uni_num",
                        type=int,
                        default=100000,
                        help="Num of uniform points")

    parser.add_argument("--n_jobs",
                        type=int,
                        default=8,
                        help="Workers num to query")
    parser.add_argument("--override",
                        action="store_true",
                        help="Overriding existing files")
    args = parser.parse_args()

    if len(args.class_name) == 1:

        CLASS_NAME = args.class_name[0]
        SPLIT = args.split
        SRC_DATASET_DIR = args.src_dataset_dir
        DST_DATASET_DIR = args.dst_dataset_dir
        IMG_DATASET_DIR = args.img_dataset_dir
        SPLIT_DIR = args.split_dir
        LOAD_MESH_NAME = args.load_mesh_name
        ###################################################
        IMAGES_NUM = args.images_num
        ###################################################
        RAND_SAMPLING_NUM = args.rand_sampling_num
        ###################################################
        POISSON_SAMPLING_NUM = args.poisson_sampling_num
        ###################################################
        # SDF
        SDF_METHOD = args.sdf_method
        SDF_APPROX_SAMPLING_NUM = args.sdf_approx_sampling_num
        NEAR_STD = args.near_std
        NEAR_NUM = args.near_num
        FAR_STD = args.far_std
        FAR_NUM = args.far_num
        UNI_SCALE = args.uni_scale
        UNI_NUM = args.uni_num
        ###################################################
        N_JOBS = args.n_jobs
        OVERRIDE = args.override

        assert SDF_METHOD in ["normals", "nearest"]

        auto_process()

    else:
        for class_name in args.class_name:
            os.system(
                f"python {__file__} --class_name {class_name} --split {args.split} --src_dataset_dir {args.src_dataset_dir} "
                f"--dst_dataset_dir {args.dst_dataset_dir} --img_dataset_dir {args.img_dataset_dir} --split_dir {args.split_dir} "
                f"--load_mesh_name {args.load_mesh_name} --images_num {args.images_num} --rand_sampling_num {args.rand_sampling_num} "
                f"--poisson_sampling_num {args.poisson_sampling_num} --sdf_method {args.sdf_method} "
                f"--sdf_approx_sampling_num {args.sdf_approx_sampling_num} --near_std {args.near_std} --near_num {args.near_num} "
                f"--far_std {args.far_std} --far_num {args.far_num} --uni_scale {args.uni_scale} "
                f"--uni_num {args.uni_num} --n_jobs {args.n_jobs}")

    print("All done.")
