import numpy as np


def project_depth_to_points(
        intrinsics: np.array,
        depth: np.array,
        instance_mask: np.array = None) -> [np.array, tuple]:
    r"""Projection of depth map to points.

        Input:
            intrinsics: camera intrinsics, [3, 3]
            depth: depth map, [H, W]
            instance_mask: mask, [H, W]. Defaults to None.
        Output:
            [pts, ids]
            pts: points, [N, 3]
            ids: x,y indexes of the depth map corresponding to the points, [np.array(N), np.array(N)]
    """
    intrinsics_inv = np.linalg.inv(intrinsics)

    non_zero_mask = (depth > 0)
    if instance_mask is not None:
        non_zero_mask = np.logical_and(instance_mask, non_zero_mask)

    ids = np.where(non_zero_mask)
    grid = np.array([ids[1], ids[0]])

    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0)  # [3, num_pixel]

    xyz = np.dot(intrinsics_inv, uv_grid)  # [3, num_pixel]
    xyz = np.transpose(xyz)  # [num_pixel, 3]

    z = depth[ids[0], ids[1]]

    pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
    return pts, ids


def project_points_to_2d(intrinsics: np.array, points: np.array) -> np.array:
    r"""Projection of points to 2d.

        Input:
            intrinsics: camera intrinsics, [3, 3]
            points: coordinates of points, [N, 3]
        Output:
            2d coordinate: [N, 2]
        """
    points = points / points[:, -1:]
    points = np.dot(points, intrinsics.T)
    return points[:, :2].astype(np.int16)
