# Copyright (c) Gorilla-Lab. All rights reserved.
import time

import torch
import torch.nn.functional as F
import numpy as np
from numpy import pi

from gorilla import multi_apply
from .box3d_nms import aligned_3d_nms
from ...losses import chamfer_distance


class PartialBinBasedBBoxCoder(object):
    r"""Partial bin based bbox coder.
    
    Args:
        num_dir_bins (int): Number of bins to encode direction angle.
        num_sizes (int): Number of size clusters.
        mean_sizes (list[list[int]]): Mean size of bboxes in each class.
        with_rot (bool): Whether the bbox is with rotation.
    """
    def __init__(self,
                 num_dir_bins,
                 num_sizes,
                 num_classes,
                 mean_sizes,
                 with_rot=True,
                 gt_per_seed=None,
                 train_cfg=None,
                 test_cfg=None):
        super(PartialBinBasedBBoxCoder, self).__init__()
        assert len(mean_sizes) == num_sizes
        self.num_dir_bins = num_dir_bins
        self.num_sizes = num_sizes
        self.mean_sizes = mean_sizes
        self.num_classes = num_classes
        self.with_rot = with_rot
        self.gt_per_seed = gt_per_seed
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def __call__(self, processed, be_train=True):
        r"""Generate bboxes from vote head predictions.

        Args:
            points (torch.Tensor): Input points.
            bbox_preds (dict): Predictions from vote head.
        Returns:
            list[tuple[torch.Tensor]]: Bounding boxes, scores and labels.
        """
        # NOTE: this is for test, and need to fix train
        if be_train:
            points = processed["points"]
            bbox_preds = processed["bbox_preds"]
            gt_bboxes_3d = processed["gt_bboxes_3d"]
            gt_labels_3d = processed["gt_labels_3d"]
            pts_semantic_mask = processed["pts_semantic_mask"]
            pts_instance_mask = processed["pts_instance_mask"]
            sample_metas = processed["sample_metas"]

            predictions = bbox_preds["predictions"]
            aggregated_points = bbox_preds["aggregated_points"]

            decode_res = self.split_pred(predictions, aggregated_points)
            bbox_preds.update(decode_res)

            vote_targets, vote_target_masks, size_class_targets, size_res_targets, dir_class_targets, \
            dir_res_targets, center_targets, mask_targets, valid_gt_masks, objectness_targets, \
            objectness_weights, box_loss_weights, valid_gt_weights = self.get_targets(points,
                                                                                      gt_bboxes_3d,
                                                                                      gt_labels_3d,
                                                                                      pts_semantic_mask,
                                                                                      pts_instance_mask,
                                                                                      bbox_preds)

            bbox_result = dict(bbox_preds=bbox_preds,
                               vote_targets=vote_targets,
                               vote_target_masks=vote_target_masks,
                               size_class_targets=size_class_targets,
                               size_res_targets=size_res_targets,
                               dir_class_targets=dir_class_targets,
                               dir_res_targets=dir_res_targets,
                               center_targets=center_targets,
                               mask_targets=mask_targets,
                               valid_gt_masks=valid_gt_masks,
                               objectness_targets=objectness_targets,
                               objectness_weights=objectness_weights,
                               box_loss_weights=box_loss_weights,
                               valid_gt_weights=valid_gt_weights,
                               sample_metas=sample_metas)

        else:
            points = processed["points"]
            bbox_preds = processed["bbox_preds"]
            sample_metas = processed["sample_metas"]

            predictions = bbox_preds["predictions"]
            aggregated_points = bbox_preds["aggregated_points"]

            decode_res = self.split_pred(predictions, aggregated_points)
            bbox_preds.update(decode_res)

            # decode boxes
            obj_scores = F.softmax(bbox_preds["obj_scores"], dim=-1)[..., -1]
            sem_scores = F.softmax(bbox_preds["sem_scores"], dim=-1)
            bbox3d = self.decode(bbox_preds, aggregated_points)

            # the batch_size for test is 1, just use `0` to index data
            bbox_selected, score_selected, labels = self.multiclass_nms_single(
                obj_scores[0], sem_scores[0], bbox3d[0], points[0, ..., :3],
                sample_metas[0])
            bbox = sample_metas[0]["box_type_3d"](
                bbox_selected,
                box_dim=bbox_selected.shape[-1],
                with_yaw=self.with_rot)

            # import os.path as osp
            # sample_idx = processed["sample_metas"][0]["sample_idx"]
            # save_dir = osp.join(".", "votenet_bbox")
            # bbox_array = np.array(bbox.tensor.to("cpu"))
            # score_array = np.array(score_selected.to("cpu"))
            # label_array = np.array(labels.to("cpu"))
            # ids = (score_array > 0.0)
            # np.save(osp.join(save_dir, sample_idx + "_bbox.npy"), bbox_array[ids])
            # np.save(osp.join(save_dir, sample_idx + "_score.npy"), score_array[ids])
            # np.save(osp.join(save_dir, sample_idx + "_label.npy"), label_array[ids])
            bbox_result = dict(boxes_3d=bbox.to("cpu"),
                               scores_3d=score_selected.cpu(),
                               labels_3d=labels.cpu())

        return bbox_result

    def encode(self, gt_bboxes_3d, gt_labels_3d):
        r"""Encode ground truth to prediction targets.

        Args:
            gt_bboxes_3d (BaseInstance3DBoxes): Ground truth bboxes \
                with shape (n, 7).
            gt_labels_3d (torch.Tensor): Ground truth classes.
        Returns:
            tuple: Targets of center, size and direction.
        """
        # generate center target
        center_target = gt_bboxes_3d.gravity_center

        # generate bbox size target
        size_class_target = gt_labels_3d
        size_res_target = gt_bboxes_3d.dims - gt_bboxes_3d.tensor.new_tensor(
            self.mean_sizes)[size_class_target]

        # generate dir target
        box_num = gt_labels_3d.shape[0]
        if self.with_rot:
            dir_class_target, dir_res_target = self.angle2class(
                gt_bboxes_3d.yaw)
        else:
            dir_class_target = gt_labels_3d.new_zeros(box_num)
            dir_res_target = gt_bboxes_3d.tensor.new_zeros(box_num)

        return (center_target, size_class_target, size_res_target,
                dir_class_target, dir_res_target)

    def decode(self, bbox_out, center):
        r"""Decode predicted parts to bbox3d.

        Args:
            bbox_out (dict): Predictions from model, should contain keys below.
                - center: predicted bottom center of bboxes.
                - dir_class: predicted bbox direction class.
                - dir_res: predicted bbox direction residual.
                - size_class: predicted bbox size class.
                - size_res: predicted bbox size residual.
        Returns:
            torch.Tensor: Decoded bbox3d with shape (batch, n, 7).
        """
        # center = bbox_out["center"]
        batch_size, num_proposal = center.shape[:2]

        # decode heading angle
        if self.with_rot:
            dir_class = torch.argmax(bbox_out["dir_class"], -1)
            dir_res = torch.gather(bbox_out["dir_res"], 2,
                                   dir_class.unsqueeze(-1))
            dir_res.squeeze_(2)
            dir_angle = self.class2angle(dir_class, dir_res).reshape(
                batch_size, num_proposal, 1)
        else:
            dir_angle = center.new_zeros(batch_size, num_proposal, 1)

        # decode bbox size
        size_class = torch.argmax(bbox_out["size_class"], -1, keepdim=True)
        size_res = torch.gather(bbox_out["size_res"], 2,
                                size_class.unsqueeze(-1).repeat(1, 1, 1, 3))
        mean_sizes = center.new_tensor(self.mean_sizes)
        size_base = torch.index_select(mean_sizes, 0, size_class.reshape(-1))
        bbox_size = size_base.reshape(batch_size, num_proposal,
                                      -1) + size_res.squeeze(2)

        bbox3d = torch.cat([center, bbox_size, dir_angle], dim=-1)
        return bbox3d

    def get_targets(self,
                    points,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    pts_semantic_mask=None,
                    pts_instance_mask=None,
                    bbox_preds=None):
        """Generate targets of vote head.
        Args:
            points (list[torch.Tensor]): Points of each batch.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): Labels of each batch.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise instance
                label of each batch.
            bbox_preds (torch.Tensor): Bounding box predictions of vote head.
        Returns:
            tuple[torch.Tensor]: Targets of vote head.
        """
        # find empty example
        valid_gt_masks = list()
        gt_num = list()
        for index in range(len(gt_labels_3d)):
            if len(gt_labels_3d[index]) == 0:
                fake_box = gt_bboxes_3d[index].tensor.new_zeros(
                    1, gt_bboxes_3d[index].tensor.shape[-1])
                gt_bboxes_3d[index] = gt_bboxes_3d[index].new_box(fake_box)
                gt_labels_3d[index] = gt_labels_3d[index].new_zeros(1)
                valid_gt_masks.append(gt_labels_3d[index].new_zeros(1))
                gt_num.append(1)
            else:
                valid_gt_masks.append(gt_labels_3d[index].new_ones(
                    gt_labels_3d[index].shape))
                gt_num.append(gt_labels_3d[index].shape[0])
        max_gt_num = max(gt_num)

        if pts_semantic_mask is None:
            pts_semantic_mask = [None for i in range(len(gt_labels_3d))]
            pts_instance_mask = [None for i in range(len(gt_labels_3d))]

        aggregated_points = [
            bbox_preds["aggregated_points"][i]
            for i in range(len(gt_labels_3d))
        ]

        (vote_targets, vote_target_masks, size_class_targets, size_res_targets,
         dir_class_targets, dir_res_targets, center_targets, mask_targets,
         objectness_targets, objectness_masks) = multi_apply(
             self.get_targets_single, points, gt_bboxes_3d, gt_labels_3d,
             pts_semantic_mask, pts_instance_mask, aggregated_points)

        # pad targets as original code of votenet.
        for index in range(len(gt_labels_3d)):
            pad_num = max_gt_num - gt_labels_3d[index].shape[0]
            center_targets[index] = F.pad(center_targets[index],
                                          (0, 0, 0, pad_num))
            valid_gt_masks[index] = F.pad(valid_gt_masks[index], (0, pad_num))

        vote_targets = torch.stack(vote_targets)
        vote_target_masks = torch.stack(vote_target_masks)
        center_targets = torch.stack(center_targets)
        valid_gt_masks = torch.stack(valid_gt_masks)

        objectness_targets = torch.stack(objectness_targets)
        objectness_weights = torch.stack(objectness_masks)
        objectness_weights /= (torch.sum(objectness_weights) + 1e-6)
        box_loss_weights = objectness_targets.float() / (
            torch.sum(objectness_targets).float() + 1e-6)
        valid_gt_weights = valid_gt_masks.float() / (
            torch.sum(valid_gt_masks.float()) + 1e-6)
        dir_class_targets = torch.stack(dir_class_targets)
        dir_res_targets = torch.stack(dir_res_targets)
        size_class_targets = torch.stack(size_class_targets)
        size_res_targets = torch.stack(size_res_targets)
        mask_targets = torch.stack(mask_targets)

        return (vote_targets, vote_target_masks, size_class_targets,
                size_res_targets, dir_class_targets, dir_res_targets,
                center_targets, mask_targets, valid_gt_masks,
                objectness_targets, objectness_weights, box_loss_weights,
                valid_gt_weights)

    def get_targets_single(self,
                           points,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           pts_semantic_mask=None,
                           pts_instance_mask=None,
                           aggregated_points=None):
        """Generate targets of vote head for single batch.
        Args:
            points (torch.Tensor): Points of each batch.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth \
                boxes of each batch.
            gt_labels_3d (torch.Tensor): Labels of each batch.
            pts_semantic_mask (None | torch.Tensor): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | torch.Tensor): Point-wise instance
                label of each batch.
            aggregated_points (torch.Tensor): Aggregated points from
                vote aggregation layer.
        Returns:
            tuple[torch.Tensor]: Targets of vote head.
        """
        assert self.with_rot or pts_semantic_mask is not None

        gt_bboxes_3d = gt_bboxes_3d.to(points.device)

        # generate votes target
        num_points = points.shape[0]
        if self.with_rot:
            vote_targets = points.new_zeros([num_points, 3 * self.gt_per_seed])
            vote_target_masks = points.new_zeros([num_points],
                                                 dtype=torch.long)
            vote_target_idx = points.new_zeros([num_points], dtype=torch.long)
            box_indices_all = gt_bboxes_3d.points_in_boxes(points)
            for i in range(gt_labels_3d.shape[0]):
                box_indices = box_indices_all[:, i]
                indices = torch.nonzero(box_indices,
                                        as_tuple=False).squeeze(-1)
                selected_points = points[indices]
                vote_target_masks[indices] = 1
                vote_targets_tmp = vote_targets[indices]
                votes = gt_bboxes_3d.gravity_center[i].unsqueeze(
                    0) - selected_points[:, :3]

                for j in range(self.gt_per_seed):
                    column_indices = torch.nonzero(
                        vote_target_idx[indices] == j,
                        as_tuple=False).squeeze(-1)
                    vote_targets_tmp[column_indices,
                                     int(j * 3):int(j * 3 +
                                                    3)] = votes[column_indices]
                    if j == 0:
                        vote_targets_tmp[column_indices] = votes[
                            column_indices].repeat(1, self.gt_per_seed)

                vote_targets[indices] = vote_targets_tmp
                vote_target_idx[indices] = torch.clamp(
                    vote_target_idx[indices] + 1, max=2)
        elif pts_semantic_mask is not None:
            vote_targets = points.new_zeros([num_points, 3])
            vote_target_masks = points.new_zeros([num_points],
                                                 dtype=torch.long)

            for i in torch.unique(pts_instance_mask):
                indices = torch.nonzero(pts_instance_mask == i,
                                        as_tuple=False).squeeze(-1)
                if pts_semantic_mask[indices[0]] < self.num_classes:
                    selected_points = points[indices, :3]
                    center = 0.5 * (selected_points.min(0)[0] +
                                    selected_points.max(0)[0])
                    vote_targets[indices, :] = center - selected_points
                    vote_target_masks[indices] = 1
            vote_targets = vote_targets.repeat((1, self.gt_per_seed))
        else:
            raise NotImplementedError

        (center_targets, size_class_targets, size_res_targets,
         dir_class_targets,
         dir_res_targets) = self.encode(gt_bboxes_3d, gt_labels_3d)

        proposal_num = aggregated_points.shape[0]
        distance1, _, assignment, _ = chamfer_distance(
            aggregated_points.unsqueeze(0),
            center_targets.unsqueeze(0),
            reduction="none")
        assignment = assignment.squeeze(0)
        euclidean_distance1 = torch.sqrt(distance1.squeeze(0) + 1e-6)

        objectness_targets = points.new_zeros((proposal_num), dtype=torch.long)
        objectness_targets[
            euclidean_distance1 < self.train_cfg["pos_distance_thr"]] = 1

        objectness_masks = points.new_zeros((proposal_num))
        objectness_masks[
            euclidean_distance1 < self.train_cfg["pos_distance_thr"]] = 1.0
        objectness_masks[
            euclidean_distance1 > self.train_cfg["neg_distance_thr"]] = 1.0

        dir_class_targets = dir_class_targets[assignment]
        dir_res_targets = dir_res_targets[assignment]
        dir_res_targets /= (pi / self.num_dir_bins)
        size_class_targets = size_class_targets[assignment]
        size_res_targets = size_res_targets[assignment]

        one_hot_size_targets = gt_bboxes_3d.tensor.new_zeros(
            (proposal_num, self.num_sizes))
        one_hot_size_targets.scatter_(1, size_class_targets.unsqueeze(-1), 1)
        one_hot_size_targets = one_hot_size_targets.unsqueeze(-1).repeat(
            1, 1, 3)
        mean_sizes = size_res_targets.new_tensor(self.mean_sizes).unsqueeze(0)
        pos_mean_sizes = torch.sum(one_hot_size_targets * mean_sizes, 1)
        size_res_targets /= pos_mean_sizes

        mask_targets = gt_labels_3d[assignment]

        return (vote_targets, vote_target_masks, size_class_targets,
                size_res_targets,
                dir_class_targets, dir_res_targets, center_targets,
                mask_targets.long(), objectness_targets, objectness_masks)

    def split_pred(self, preds, base_xyz):
        r"""Split predicted features to specific parts.

        Args:
            preds (torch.Tensor): Predicted features to split.
            base_xyz (torch.Tensor): Coordinates of points.
        Returns:
            dict[str, torch.Tensor]: Split results.
        """
        results = {}
        start, end = 0, 0
        preds_trans = preds.transpose(2, 1)

        # decode objectness score
        end += 2
        results["obj_scores"] = preds_trans[..., start:end]
        start = end

        # decode center
        end += 3
        # [batch_size, num_proposal, 3]
        results["center"] = base_xyz + preds_trans[..., start:end]
        start = end

        # decode direction
        end += self.num_dir_bins
        results["dir_class"] = preds_trans[..., start:end]
        start = end

        end += self.num_dir_bins
        dir_res_norm = preds_trans[..., start:end]
        start = end

        results["dir_res_norm"] = dir_res_norm
        results["dir_res"] = dir_res_norm * (pi / self.num_dir_bins)

        # decode size
        end += self.num_sizes
        results["size_class"] = preds_trans[..., start:end]
        start = end

        end += self.num_sizes * 3
        size_res_norm = preds_trans[..., start:end]
        batch_size, num_proposal = preds_trans.shape[:2]
        size_res_norm = size_res_norm.view(
            [batch_size, num_proposal, self.num_sizes, 3])
        start = end

        results["size_res_norm"] = size_res_norm
        mean_sizes = preds.new_tensor(self.mean_sizes)
        results["size_res"] = (size_res_norm *
                               mean_sizes.unsqueeze(0).unsqueeze(0))

        # decode semantic score
        results["sem_scores"] = preds_trans[..., start:]

        return results

    def angle2class(self, angle):
        r"""Convert continuous angle to a discrete class and a residual.
        Convert continuous angle to a discrete class and a small
        regression number from class center angle to current angle.
        Args:
            angle (torch.Tensor): Angle is from 0-2pi (or -pi~pi),
                class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N).
        Returns:
            tuple: Encoded discrete class and residual.
        """
        angle = angle % (2 * pi)
        angle_per_class = 2 * pi / float(self.num_dir_bins)
        shifted_angle = (angle + angle_per_class / 2) % (2 * pi)
        angle_cls = shifted_angle // angle_per_class
        angle_res = shifted_angle - (angle_cls * angle_per_class +
                                     angle_per_class / 2)
        return angle_cls.long(), angle_res

    def class2angle(self, angle_cls, angle_res, limit_period=True):
        r"""Inverse function to angle2class.
        Args:
            angle_cls (torch.Tensor): Angle class to decode.
            angle_res (torch.Tensor): Angle residual to decode.
            limit_period (bool): Whether to limit angle to [-pi, pi].
        Returns:
            torch.Tensor: Angle decoded from angle_cls and angle_res.
        """
        angle_per_class = 2 * pi / float(self.num_dir_bins)
        angle_center = angle_cls.float() * angle_per_class
        angle = angle_center + angle_res
        if limit_period:
            angle[angle > pi] -= 2 * pi
        return angle

    def multiclass_nms_single(self, obj_scores, sem_scores, bbox, points,
                              input_meta):
        """Multi-class nms in single batch.
        Args:
            obj_scores (torch.Tensor): Objectness score of bounding boxes.
            sem_scores (torch.Tensor): semantic class score of bounding boxes.
            bbox (torch.Tensor): Predicted bounding boxes.
            points (torch.Tensor): Input points.
            input_meta (dict): Point cloud and image"s meta info.
        Returns:
            tuple[torch.Tensor]: Bounding boxes, scores and labels.
        """
        bbox = input_meta["box_type_3d"](bbox,
                                         box_dim=bbox.shape[-1],
                                         with_yaw=self.with_rot,
                                         origin=(0.5, 0.5, 0.5))
        box_indices = bbox.points_in_boxes(points)

        corner3d = bbox.corners
        minmax_box3d = corner3d.new(torch.Size((corner3d.shape[0], 6)))
        minmax_box3d[:, :3] = torch.min(corner3d, dim=1)[0]
        minmax_box3d[:, 3:] = torch.max(corner3d, dim=1)[0]

        nonempty_box_mask = box_indices.T.sum(1) > 5

        bbox_classes = torch.argmax(sem_scores, -1)
        t1 = time.time()
        nms_selected = aligned_3d_nms(minmax_box3d[nonempty_box_mask],
                                      obj_scores[nonempty_box_mask],
                                      bbox_classes[nonempty_box_mask],
                                      self.test_cfg.nms_thr)

        # filter empty boxes and boxes with low score
        scores_mask = (obj_scores > self.test_cfg.score_thr)
        nonempty_box_inds = torch.nonzero(nonempty_box_mask,
                                          as_tuple=False).flatten()
        nonempty_mask = torch.zeros_like(bbox_classes).scatter(
            0, nonempty_box_inds[nms_selected], 1)
        selected = (nonempty_mask.bool() & scores_mask.bool())

        if self.test_cfg.per_class_proposal:
            bbox_selected, score_selected, labels = [], [], []
            for k in range(sem_scores.shape[-1]):
                bbox_selected.append(bbox[selected].tensor)
                score_selected.append(obj_scores[selected] *
                                      sem_scores[selected][:, k])
                labels.append(
                    torch.zeros_like(bbox_classes[selected]).fill_(k))
            bbox_selected = torch.cat(bbox_selected, 0)
            score_selected = torch.cat(score_selected, 0)
            labels = torch.cat(labels, 0)
        else:
            bbox_selected = bbox[selected].tensor
            score_selected = obj_scores[selected]
            labels = bbox_classes[selected]

        return bbox_selected, score_selected, labels
