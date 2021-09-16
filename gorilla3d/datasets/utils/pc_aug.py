# Copyright (c) Gorilla-Lab. All rights reserved.
import numpy as np
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
import transforms3d.euler as euler


def elastic(xyz, gran, mag):
    """Elastic distortion (from point group)

    Args:
        xyz (np.ndarray): input point cloud
        gran (float): distortion param
        mag (float): distortion scalar

    Returns:
        xyz: point cloud with elastic distortion
    """
    blur0 = np.ones((3, 1, 1)).astype("float32") / 3
    blur1 = np.ones((1, 3, 1)).astype("float32") / 3
    blur2 = np.ones((1, 1, 3)).astype("float32") / 3

    bb = np.abs(xyz).max(0).astype(np.int32) // gran + 3
    noise = [
        np.random.randn(bb[0], bb[1], bb[2]).astype("float32")
        for _ in range(3)
    ]
    noise = [
        ndimage.filters.convolve(n, blur0, mode="constant", cval=0)
        for n in noise
    ]
    noise = [
        ndimage.filters.convolve(n, blur1, mode="constant", cval=0)
        for n in noise
    ]
    noise = [
        ndimage.filters.convolve(n, blur2, mode="constant", cval=0)
        for n in noise
    ]
    noise = [
        ndimage.filters.convolve(n, blur0, mode="constant", cval=0)
        for n in noise
    ]
    noise = [
        ndimage.filters.convolve(n, blur1, mode="constant", cval=0)
        for n in noise
    ]
    noise = [
        ndimage.filters.convolve(n, blur2, mode="constant", cval=0)
        for n in noise
    ]
    ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
    interp = [
        interpolate.RegularGridInterpolator(ax,
                                            n,
                                            bounds_error=0,
                                            fill_value=0) for n in noise
    ]

    def g(xyz_):
        return np.hstack([i(xyz_)[:, None] for i in interp])

    return xyz + g(xyz) * mag


def pc_jitter(xyz, std=0.1):
    jitter_mat = np.eye(3)
    jitter_mat += np.random.randn(3, 3) * std
    xyz = xyz @ jitter_mat
    return xyz


def pc_flipper(xyz, dim="x"):
    dims = ["x", "y", "z"]
    assert dim in dims
    flip_dim = dims.index(dim)
    xyz[:, flip_dim] = -xyz[:, flip_dim]
    return xyz


def pc_rotator(xyz):
    theta = np.random.rand() * 2 * np.pi
    rot_mat = euler.euler2mat(0, 0, theta, "syxz")
    xyz = xyz @ rot_mat.T
    return xyz


def pc_aug(xyz, jitter=False, flip=False, rot=False):
    """point cloud augmentation(from point group)

    Args:
        x (np.ndarray): input point cloud
        jitter (bool, optional): [description]. Defaults to False.
        flip (bool, optional): [description]. Defaults to False.
        rot (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    if jitter:
        xyz = pc_jitter(xyz)
    if flip:
        flag = np.random.randint(0, 2)
        if flag:
            xyz = pc_flipper(xyz)
    if rot:
        xyz = pc_rotator(xyz)

    return xyz
