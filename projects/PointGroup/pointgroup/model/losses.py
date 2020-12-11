# Copyright (c) Gorilla-Lab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.dynamic = cfg.model.dynamic

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


        if prepare_flag:
            """score loss"""
            scores, proposals_idx, proposals_offset, instance_pointnum = loss_inp["proposal_scores"]
            # scores: (num_prop, 1), float32
            # proposals_idx: (sum_points, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (num_prop + 1), int, cpu
            # instance_pointnum: (total_num_inst), int

            ious = pointgroup_ops.get_iou(proposals_idx[:, 1].cuda(),
                                          proposals_offset.cuda(),
                                          instance_labels,
                                          instance_pointnum) # (num_prop, num_inst), float

            gt_ious, gt_inst_idxs = ious.max(1)  # (num_prop) float, long
            score_loss = gorilla.iou_guided_loss(scores.view(-1), gt_ious, self.fg_thresh, self.bg_thresh)
            score_loss = score_loss.mean()

            loss_out["score_loss"] = (score_loss, gt_ious.shape[0])

            if self.dynamic:
                mask_pred, batch_mask, proposals_idx_dynamic, proposals_offset_dynamic = loss_inp["proposal_dynamic"]
                # mask_pred: (num_prop, N), float, gpu, sigmoid value of each proposal mask
                # batch_mask: (num_prop, N), bool, gpu, filter out the batch outside
                # proposals_idx: (sum_points, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (num_prop + 1), int, cpu
                
                # get the gt_mask using gt_instance_idxs (TODO: not elegant and slow)
                mask_gt = mask_pred.new_zeros(mask_pred.shape) # (num_prop, N)
                for mask_i, inst_idx in enumerate(gt_inst_idxs):
                    mask_gt[mask_i, :] = (instance_labels == inst_idx).float()

                mask_loss = F.binary_cross_entropy(mask_pred, mask_gt, reduction="none") # (num_prop, N)
                mask_loss = torch.sum(mask_loss[batch_mask]) / (torch.sum(batch_mask) + 1e-6)

                loss_out["mask_loss"] = (mask_loss, torch.sum(batch_mask))
                print("mask_loss: ", mask_loss)

        """total loss"""
        # loss = mask_loss
        loss = self.loss_weight[0] * semantic_loss + self.loss_weight[1] * offset_norm_loss + self.loss_weight[2] * offset_dir_loss
        if prepare_flag:
            loss += (self.loss_weight[3] * score_loss)
            if self.dynamic:
                loss += (self.loss_weight[4] * mask_loss)

        return loss, loss_out

