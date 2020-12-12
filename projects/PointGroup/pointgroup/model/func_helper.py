from random import sample
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import coo_matrix
from torch_scatter import scatter_max, scatter_mean
from numba import jit


def overseg_pooling(feats, overseg, mode="max"):
    r"""pooling according to overseg

    Args:
        feats (torch.Tensor): (N, C)
        overseg (torch.Tensor): (N)
        mode (str, optional): pooling type. Defaults to "max".

    Returns:
        overseg_feats (torch.Tensor): (num_overseg)
    """
    if mode == "max": # max-pooling
        overseg_feats = scatter_max(feats, overseg, dim=0)[0] # [num_overseg, C]
    elif mode == "mean": # mean-pooling
        overseg_feats = scatter_mean(feats, overseg, dim=0) # [num_overseg, C]
    else:
        raise ValueError("mode must be 'max' or 'mean', but got {}".format(mode))
    
    return overseg_feats


def overseg_graph_fusion(overseg, batch_idxs, overseg_semantic_preds, overseg_batch_idxs, adajency_matrix_list):
    clusters = []
    bias = 0
    offset_bias = 0
    cluster_bias = 0
    proposals_idxs = []
    proposals_offsets = [0]
    for batch_idx in range(batch_idxs.max() + 1):
        adajency_matrix = np.array(adajency_matrix_list[batch_idx].cpu())
        ids = (batch_idxs == batch_idx)
        batch_overseg = np.array(overseg[ids].cpu())
        batch_overseg -= batch_overseg.min()
        overseg_ids = (overseg_batch_idxs == batch_idx)
        batch_overseg_semantic_preds = np.array(overseg_semantic_preds[overseg_ids].cpu())
        batch_clusters = bfs_cluster(adajency_matrix, batch_overseg_semantic_preds)
        clusters.extend(batch_clusters)
        for cluster in batch_clusters:
            proposals_idx = torch.Tensor(np.where(np.isin(batch_overseg, np.array(cluster)))[0]) # [n']
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
        adajency_matrix (np.ndarray): (num_overseg, num_overseg)
    Return:
        clusters (list): list of cluster lists
    """
    visited = np.zeros(len(adajency_matrix)) # [num_overseg]
    clusters = []
    for overseg_id in range(len(adajency_matrix)):
        overseg_id = int(overseg_id)
        if visited[overseg_id] == 1 or semantic_preds[overseg_id] < 2:
            continue
        queue = []
        cluster = []
        queue.append(overseg_id)
        while (len(queue) > 0):
            # get aim point id from queue
            overseg_id = queue.pop(0)
            if visited[overseg_id] == 1:
                continue
            visited[overseg_id] = 1
            cluster.append(overseg_id)
            group = np.where(adajency_matrix[overseg_id])[0]
            for neighbor in group:
                neighbor = int(neighbor)
                if visited[neighbor] == 1:
                    continue
                queue.append(neighbor)

        if len(cluster) > 1:
            clusters.append(cluster)

    return clusters

def overseg_fusion(proposals_idx, proposals_offset, overseg, thr_filter=0.4, thr_fusion=0.4):
    r"""use overseg to refine box_idxs_of_pts

    Args:
        proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
        proposals_offset: (nProposal + 1), int
    """
    proposals_idx_fusion = []
    proposals_offset_fusion = [0]
    cluster_id = 0
    bias = 0
    _, overseg = torch.unique(overseg, return_inverse=True)
    overseg = overseg.cpu()
    overseg_count_map = torch.bincount(overseg)
    for start, end in zip(proposals_offset[:-1], proposals_offset[1:]):
        temp_proposals_idx = proposals_idx[start:end, :].long()
        mask_ids = temp_proposals_idx[:, 1]
        mask = overseg.new_zeros(overseg.shape).bool()
        mask[mask_ids] = True
        temp_cluster_count = torch.bincount(overseg[mask])
        cluster_count_map = overseg_count_map.new_zeros(overseg_count_map.shape)
        cluster_count_map[:temp_cluster_count.shape[0]] = temp_cluster_count
        # fusion the overseg which has been covered over thr_fusion
        ratio_map = (cluster_count_map.float() / overseg_count_map.float())
        valid_mask = (ratio_map > thr_fusion)
        cluster_ids = torch.unique(overseg)[valid_mask].cpu().numpy()
        temp_idxs = torch.Tensor(np.isin(overseg.cpu().numpy(), cluster_ids)).bool()
        # if the covered ratio of the overseg is lower than thr_fusion but higher than thr_filter
        # keep constant
        ignore_mask = (ratio_map > thr_filter) & (ratio_map < thr_fusion)
        ignore_cluster_ids = torch.unique(overseg)[ignore_mask].cpu().numpy()
        ignore_temp_idxs = torch.Tensor(np.isin(overseg.cpu().numpy(), ignore_cluster_ids)).bool() & mask

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


def align_overseg_semantic_label(semantic_labels, overseg, num_semantic=20):
    """refine semantic segmentation by overseg

    Args:
        semantic_labels (torch.Tensor): N, semantic label of points
        overseg: (torch.Tensor): N, overseg cluster id of points

    Returns:
        semantic_map: (torch.Tensor): num_overseg, overseg's semantic label
    """
    row = overseg.cpu().numpy() # overseg has been compression
    # _, row = np.unique(overseg.cpu().numpy(), return_inverse=True)
    col = semantic_labels.cpu().numpy()
    col[col < 0] = num_semantic
    data = np.ones(len(overseg))
    shape = (len(np.unique(row)), num_semantic + 1)
    semantic_map = coo_matrix((data, (row, col)), shape=shape).toarray()  # [num_overseg, num_semantic + 1]
    semantic_map = torch.Tensor(np.argmax(semantic_map, axis=1)).long().to(semantic_labels.device)  # [num_overseg]
    semantic_map[semantic_map == num_semantic] = -100 # ignore_label

    return semantic_map


def refine_semantic_segmentation(semantic_preds, overseg, num_semantic=20):
    """refine semantic segmentation by overseg

    Args:
        semantic_preds (torch.Tensor): N, semantic label of points
        overseg: (torch.Tensor): N, overseg cluster id of points

    Returns:
        replace_semantic: (torch.Tensor): N, refine semantic label of points
    """
    _, row = np.unique(overseg.cpu().numpy(), return_inverse=True)
    col = semantic_preds.cpu().numpy()
    data = np.ones(len(overseg))
    shape = (len(np.unique(row)), num_semantic)
    semantic_map = coo_matrix((data, (row, col)), shape=shape).toarray()  # [num_overseg, num_semantic]
    semantic_map = torch.Tensor(np.argmax(semantic_map, axis=1)).to(semantic_preds.device) # [num_overseg]
    replace_semantic = semantic_map[torch.Tensor(row).to(semantic_preds.device).long()]

    return replace_semantic


def visual_tree(coords, overseg, batch_offsets, overseg_centers, overseg_batch_idxs, adajency_matrix_list, scene_list, save_dir="visual", suffix="lines"):
    r"""visualize the tree build from overseg

    Args:
        coords (torch.Tensor): (N, 3)
        overseg (torch.Tensor): (N)
        batch_offsets (torch.Tensor): (B + 1)
        overseg_centers (torch.Tensor): (num_overseg, 3)
        overseg_batch_idxs (torch.Tensor): (num_overseg)
        scene_list (list): list of sample_scene
        adajency_matrix_list (torch.Tensor): (num_overseg, num_overseg)
    """
    import os.path as osp
    import open3d as o3d
    for batch_idx, (start, end, scene) in enumerate(zip(batch_offsets[:-1], batch_offsets[1:], scene_list)):
        # visual overseg point cloud
        batch_coords = coords[start:end].cpu().numpy() # [N, 3]
        batch_overseg = overseg[start:end].cpu().numpy() # [N]
        all_pc = o3d.geometry.PointCloud()
        for overseg_id in np.unique(batch_overseg):
            ids = (batch_overseg == overseg_id)
            temp_coords = batch_coords[ids] # [n, 3]
            temp_colors = np.zeros_like(temp_coords) # [n, 3]
            temp_colors[:] = np.random.random(3)
            temp_pc = o3d.geometry.PointCloud()
            temp_pc.points = o3d.utility.Vector3dVector(temp_coords)
            temp_pc.colors = o3d.utility.Vector3dVector(temp_colors)
            all_pc += temp_pc
        o3d.io.write_point_cloud(osp.join(save_dir, scene + "_overseg.ply"), all_pc)

        # visual graph
        line_set = o3d.geometry.LineSet()
        overseg_ids = (overseg_batch_idxs == batch_idx)
        batch_overseg_center = overseg_centers[overseg_ids].cpu().numpy() # [num_overseg, 3]
        adajency_matrix = adajency_matrix_list[batch_idx].bool().cpu().numpy() # [num_overseg, num_overseg]
        edges_x, edges_y = np.where(adajency_matrix) # [num_edges], [num_edges]
        lines = np.stack([edges_x, edges_y], axis=1) # [num_edges, 2]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(batch_overseg_center)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        o3d.io.write_line_set(osp.join(save_dir, scene + "_{}.ply".format(suffix)), line_set)

