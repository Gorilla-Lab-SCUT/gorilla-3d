# Copyright (c) Gorilla-Lab. All rights reserved.

import torch
import numpy as np
from scipy.sparse import coo_matrix


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


