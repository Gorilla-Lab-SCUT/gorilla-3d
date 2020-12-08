# Copyright (c) Gorilla-Lab. All rights reserved.
import torch
import torch.nn as nn
import gorilla
from torch_scatter import scatter_mean

from ..lib.pointgroup_ops.functions import pointgroup_ops

class PointGroupLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ignore_label = cfg.data.ignore_label
        self.prepare_epochs = cfg.model.prepare_epochs
        self.fg_thresh = cfg.model.fg_thresh
        self.bg_thresh = cfg.model.bg_thresh
        self.loss_weight = cfg.model.loss_weight

        #### criterion
        self.semantic_criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_label)
        self.score_criterion = nn.BCELoss(reduction="none")

    def forward(self, loss_inp, epoch):
        loss_out = {}
        prepare_flag = (epoch > self.prepare_epochs)

        """semantic loss"""
        semantic_scores, semantic_labels = loss_inp["semantic_scores"]
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda

        semantic_loss = self.semantic_criterion(semantic_scores, semantic_labels)
        loss_out["semantic_loss"] = (semantic_loss, semantic_scores.shape[0])

        """offset loss"""
        pt_offsets, coords, instance_info, instance_labels = loss_inp["pt_offsets"]
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 9), float32 tensor (meanxyz, minxyz, maxxyz)
        # instance_labels: (N), long

        gt_offsets = instance_info[:, 0:3] - coords   # (N, 3)
        pt_diff = pt_offsets - gt_offsets   # (N, 3)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)
        valid = (instance_labels != self.ignore_label).float()

        offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)   # (N), float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
        pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)   # (N)
        offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

        loss_out["offset_norm_loss"] = (offset_norm_loss, valid.sum())
        loss_out["offset_dir_loss"] = (offset_dir_loss, valid.sum())


        """overseg semantic loss"""
        overseg_semantic_scores, overseg_labels = loss_inp["overseg_semantic_scores"]
        # overseg_semantic_scores: (num_overseg, nClass), float32, cuda
        # overseg_labels: (num_overseg), long, cuda
        overseg_semantic_loss = self.semantic_criterion(overseg_semantic_scores, overseg_labels)
        loss_out["overseg_semantic_loss"] = (overseg_semantic_loss, overseg_semantic_scores.shape[0])

        """"overseg offset loss"""
        overseg_centers, overseg_pt_offsets, overseg_instance_labels = loss_inp["overseg_pt_offsets"]
        # overseg_centers: (num_overseg, 3), float32, cuda
        # overseg_pt_offsets: (num_overseg, 3), float32, cuda
        # overseg_instance_labels: (num_overseg), long, cuda
        valid = (overseg_instance_labels != self.ignore_label).float()  # (num_overseg)
        
        # build instance_label to instance center map
        convert_instance_labels = instance_labels
        convert_instance_labels[instance_labels == self.ignore_label] = instance_labels.max() + 1
        instance_center_map = scatter_mean(instance_info[:, 0:3], convert_instance_labels, dim=0)  # (num_instance + 1, 3)
        overseg_instance_labels[overseg_instance_labels == self.ignore_label] = instance_labels.max()
        gt_overseg_centers = instance_center_map[overseg_instance_labels]  # (num_overseg, 3)
        overseg_gt_offsets = gt_overseg_centers - overseg_centers  # (num_overseg, 3)

        # calculate overseg offset loss
        overseg_pt_diff = overseg_pt_offsets - overseg_gt_offsets # (num_overseg, 3)
        overseg_pt_dist = torch.sum(torch.abs(overseg_pt_diff), dim=-1) # (num_overseg)
        overseg_offset_norm_loss = torch.sum(overseg_pt_dist * valid) / (torch.sum(valid) + 1e-6)

        overseg_gt_offsets_norm = torch.norm(overseg_gt_offsets, p=2, dim=1) # (num_overseg), float
        overseg_gt_offsets_ = overseg_gt_offsets / (overseg_gt_offsets_norm.unsqueeze(-1) + 1e-8) # (num_overseg, 3)
        overseg_pt_offsets_norm = torch.norm(overseg_pt_offsets, p=2, dim=1) # (num_overseg)
        overseg_pt_offsets_ = overseg_pt_offsets / (overseg_pt_offsets_norm.unsqueeze(-1) + 1e-8) # (num_overseg)
        overseg_direction_diff = - (overseg_gt_offsets_ * overseg_pt_offsets_).sum(-1)   # (N)
        overseg_offset_dir_loss = torch.sum(overseg_direction_diff * valid) / (torch.sum(valid) + 1e-6)

        loss_out["overseg_offset_norm_loss"] = (overseg_offset_norm_loss, valid.sum())
        loss_out["overseg_offset_dir_loss"] = (overseg_offset_dir_loss, valid.sum())


        if prepare_flag:
            """score loss"""
            scores, proposals_idx, proposals_offset, instance_pointnum = loss_inp["proposal_scores"]
            # scores: (nProposal, 1), float32
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            # instance_pointnum: (total_nInst), int

            ious = pointgroup_ops.get_iou(proposals_idx[:, 1].cuda(),
                                          proposals_offset.cuda(),
                                          instance_labels,
                                          instance_pointnum) # (nProposal, nInstance), float

            gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long
            score_loss = gorilla.iou_guided_loss(scores.view(-1), gt_ious, self.fg_thresh, self.bg_thresh)
            score_loss = score_loss.mean()

            loss_out["score_loss"] = (score_loss, gt_ious.shape[0])


        """total loss"""
        loss = self.loss_weight[0] * semantic_loss + self.loss_weight[1] * offset_norm_loss + self.loss_weight[2] * offset_dir_loss
        loss += self.loss_weight[4] * overseg_semantic_loss + self.loss_weight[5] * overseg_offset_norm_loss + self.loss_weight[6] * overseg_offset_dir_loss
        if prepare_flag:
            loss += (self.loss_weight[3] * score_loss)

        return loss, loss_out

