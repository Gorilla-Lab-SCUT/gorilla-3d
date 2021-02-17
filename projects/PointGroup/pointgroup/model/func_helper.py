# Copyright (c) Gorilla-Lab. All rights reserved.
from copy import deepcopy

import torch
import numpy as np
from scipy.sparse import coo_matrix
from torch_scatter import scatter_max, scatter_mean
from numba import jit


def superpoint_pooling(feats, superpoint, mode="max"):
    r"""pooling according to superpoint

    Args:
        feats (torch.Tensor): (N, C)
        superpoint (torch.Tensor): (N)
        mode (str, optional): pooling type. Defaults to "max".

    Returns:
        superpoint_feats (torch.Tensor): (num_superpoint)
    """
    if mode == "max": # max-pooling
        superpoint_feats = scatter_max(feats, superpoint, dim=0)[0] # [num_superpoint, C]
    elif mode == "mean": # mean-pooling
        superpoint_feats = scatter_mean(feats, superpoint, dim=0) # [num_superpoint, C]
    else:
        raise ValueError(f"mode must be 'max' or 'mean', but got {mode}")
    
    return superpoint_feats


def superpoint_graph_fusion(superpoint, batch_idxs, superpoint_semantic_preds, superpoint_batch_idxs, adajency_matrix_list):
    clusters = []
    bias = 0
    offset_bias = 0
    cluster_bias = 0
    proposals_idxs = []
    proposals_offsets = [0]
    for batch_idx in range(batch_idxs.max() + 1):
        adajency_matrix = np.array(adajency_matrix_list[batch_idx].cpu())
        ids = (batch_idxs == batch_idx)
        batch_superpoint = np.array(superpoint[ids].cpu())
        batch_superpoint -= batch_superpoint.min()
        superpoint_ids = (superpoint_batch_idxs == batch_idx)
        batch_superpoint_semantic_preds = np.array(superpoint_semantic_preds[superpoint_ids].cpu())
        batch_clusters = bfs_cluster(adajency_matrix, batch_superpoint_semantic_preds)
        clusters.extend(batch_clusters)
        for cluster in batch_clusters:
            proposals_idx = torch.Tensor(np.where(np.isin(batch_superpoint, np.array(cluster)))[0]) # [n']
            if len(proposals_idx) < 50:
                continue
            cluster_ids = torch.ones_like(proposals_idx) * cluster_bias # [n']
            cluster_bias += 1
            temp_proposals_idx = torch.stack([cluster_ids, torch.Tensor(proposals_idx + bias)], dim=1) # [n', 2]
            proposals_idxs.append(temp_proposals_idx)
            offset_bias += len(proposals_idx)
            proposals_offsets.append(deepcopy(offset_bias))
        bias += ids.sum()
    proposals_idxs = torch.cat(proposals_idxs).int() # [N, 2]
    proposals_offsets = torch.Tensor(proposals_offsets).int() # [nCluster + 1]
    return proposals_idxs, proposals_offsets
        
@jit(nopython=True)
def bfs_cluster(adajency_matrix, semantic_preds):
    """bfs cluster

    Args:
        adajency_matrix (np.ndarray): (num_superpoint, num_superpoint)
    Return:
        clusters (list): list of cluster lists
    """
    visited = np.zeros(len(adajency_matrix)) # [num_superpoint]
    clusters = []
    for superpoint_id in range(len(adajency_matrix)):
        superpoint_id = int(superpoint_id)
        if visited[superpoint_id] == 1 or semantic_preds[superpoint_id] < 2:
            continue
        queue = []
        cluster = []
        queue.append(superpoint_id)
        while (len(queue) > 0):
            # get aim point id from queue
            superpoint_id = queue.pop(0)
            if visited[superpoint_id] == 1:
                continue
            visited[superpoint_id] = 1
            cluster.append(superpoint_id)
            group = np.where(adajency_matrix[superpoint_id])[0]
            for neighbor in group:
                neighbor = int(neighbor)
                if visited[neighbor] == 1:
                    continue
                queue.append(neighbor)

        if len(cluster) > 1:
            clusters.append(cluster)

    return clusters

def superpoint_fusion(proposals_idx, proposals_offset, superpoint, thr_filter=0.4, thr_fusion=0.4):
    r"""use superpoint to refine box_idxs_of_pts

    Args:
        proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
        proposals_offset: (nProposal + 1), int
    """
    proposals_idx_fusion = []
    proposals_offset_fusion = [0]
    cluster_id = 0
    bias = 0
    _, superpoint = torch.unique(superpoint, return_inverse=True)
    superpoint = superpoint.cpu()
    superpoint_count_map = torch.bincount(superpoint)
    for start, end in zip(proposals_offset[:-1], proposals_offset[1:]):
        temp_proposals_idx = proposals_idx[start:end, :].long()
        mask_ids = temp_proposals_idx[:, 1]
        mask = superpoint.new_zeros(superpoint.shape).bool()
        mask[mask_ids] = True
        temp_cluster_count = torch.bincount(superpoint[mask])
        cluster_count_map = superpoint_count_map.new_zeros(superpoint_count_map.shape)
        cluster_count_map[:temp_cluster_count.shape[0]] = temp_cluster_count
        # fusion the superpoint which has been covered over thr_fusion
        ratio_map = (cluster_count_map.float() / superpoint_count_map.float())
        valid_mask = (ratio_map > thr_fusion)
        cluster_ids = torch.unique(superpoint)[valid_mask].cpu().numpy()
        temp_idxs = torch.Tensor(np.isin(superpoint.cpu().numpy(), cluster_ids)).bool()
        # if the covered ratio of the superpoint is lower than thr_fusion but higher than thr_filter
        # keep constant
        ignore_mask = (ratio_map > thr_filter) & (ratio_map < thr_fusion)
        ignore_cluster_ids = torch.unique(superpoint)[ignore_mask].cpu().numpy()
        ignore_temp_idxs = torch.Tensor(np.isin(superpoint.cpu().numpy(), ignore_cluster_ids)).bool() & mask

        # fusion the above mask
        temp_idxs = temp_idxs + ignore_temp_idxs

        # temp_idxs = (temp_idxs + mask).int()
        proposal = torch.where(temp_idxs)[0]
        cluster_id_proposal = proposal.new_ones(proposal.shape) * cluster_id
        proposal_idx = torch.stack([cluster_id_proposal, proposal], dim=1).int()  # [length_proposal, 2]
        if len(proposal_idx) < 50:
            continue
        bias += len(proposal)
        cluster_id += 1
        proposals_idx_fusion.append(proposal_idx)
        proposals_offset_fusion.append(bias)

    try:
        proposals_idx_fusion = torch.cat(proposals_idx_fusion)
        proposals_offset_fusion = torch.Tensor(proposals_offset_fusion).int()
    except:
        print("warning! empty fusion")
        proposals_idx_fusion = proposals_idx
        proposals_offset_fusion = proposals_offset

    return proposals_idx_fusion, proposals_offset_fusion


def align_superpoint_semantic_label(semantic_labels, superpoint, num_semantic=20):
    """refine semantic segmentation by superpoint

    Args:
        semantic_labels (torch.Tensor): N, semantic label of points
        superpoint: (torch.Tensor): N, superpoint cluster id of points

    Returns:
        semantic_map: (torch.Tensor): num_superpoint, superpoint's semantic label
    """
    row = superpoint.cpu().numpy() # superpoint has been compression
    # _, row = np.unique(superpoint.cpu().numpy(), return_inverse=True)
    col = semantic_labels.cpu().numpy()
    col[col < 0] = num_semantic
    data = np.ones(len(superpoint))
    shape = (len(np.unique(row)), num_semantic + 1)
    semantic_map = coo_matrix((data, (row, col)), shape=shape).toarray()  # [num_superpoint, num_semantic + 1]
    semantic_map = torch.Tensor(np.argmax(semantic_map, axis=1)).long().to(semantic_labels.device)  # [num_superpoint]
    semantic_map[semantic_map == num_semantic] = -100 # ignore_label

    return semantic_map


def refine_semantic_segmentation(semantic_preds, superpoint, num_semantic=20):
    """refine semantic segmentation by superpoint

    Args:
        semantic_preds (torch.Tensor): N, semantic label of points
        superpoint: (torch.Tensor): N, superpoint cluster id of points

    Returns:
        replace_semantic: (torch.Tensor): N, refine semantic label of points
    """
    _, row = np.unique(superpoint.cpu().numpy(), return_inverse=True)
    col = semantic_preds.cpu().numpy()
    data = np.ones(len(superpoint))
    shape = (len(np.unique(row)), num_semantic)
    semantic_map = coo_matrix((data, (row, col)), shape=shape).toarray()  # [num_superpoint, num_semantic]
    semantic_map = torch.Tensor(np.argmax(semantic_map, axis=1)).to(semantic_preds.device) # [num_superpoint]
    replace_semantic = semantic_map[torch.Tensor(row).to(semantic_preds.device).long()]

    return replace_semantic


def visual_tree(coords, superpoint, batch_offsets, superpoint_centers, superpoint_batch_idxs, adajency_matrix_list, scene_list, save_dir="visual", suffix="lines"):
    r"""visualize the tree build from superpoint

    Args:
        coords (torch.Tensor): (N, 3)
        superpoint (torch.Tensor): (N)
        batch_offsets (torch.Tensor): (B + 1)
        superpoint_centers (torch.Tensor): (num_superpoint, 3)
        superpoint_batch_idxs (torch.Tensor): (num_superpoint)
        scene_list (list): list of sample_scene
        adajency_matrix_list (torch.Tensor): (num_superpoint, num_superpoint)
    """
    import os.path as osp
    import open3d as o3d
    for batch_idx, (start, end, scene) in enumerate(zip(batch_offsets[:-1], batch_offsets[1:], scene_list)):
        # visual superpoint point cloud
        batch_coords = coords[start:end].cpu().numpy() # [N, 3]
        batch_superpoint = superpoint[start:end].cpu().numpy() # [N]
        all_pc = o3d.geometry.PointCloud()
        for superpoint_id in np.unique(batch_superpoint):
            ids = (batch_superpoint == superpoint_id)
            temp_coords = batch_coords[ids] # [n, 3]
            temp_colors = np.zeros_like(temp_coords) # [n, 3]
            temp_colors[:] = np.random.random(3)
            temp_pc = o3d.geometry.PointCloud()
            temp_pc.points = o3d.utility.Vector3dVector(temp_coords)
            temp_pc.colors = o3d.utility.Vector3dVector(temp_colors)
            all_pc += temp_pc
        o3d.io.write_point_cloud(osp.join(save_dir, scene + "_superpoint.ply"), all_pc)

        # visual graph
        line_set = o3d.geometry.LineSet()
        superpoint_ids = (superpoint_batch_idxs == batch_idx)
        batch_superpoint_center = superpoint_centers[superpoint_ids].cpu().numpy() # [num_superpoint, 3]
        adajency_matrix = adajency_matrix_list[batch_idx].bool().cpu().numpy() # [num_superpoint, num_superpoint]
        edges_x, edges_y = np.where(adajency_matrix) # [num_edges], [num_edges]
        lines = np.stack([edges_x, edges_y], axis=1) # [num_edges, 2]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(batch_superpoint_center)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        o3d.io.write_line_set(osp.join(save_dir, scene + f"_{suffix}.ply"), line_set)

