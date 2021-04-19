# Copyright (c) Gorilla-Lab. All rights reserved.
import numpy as np
import torch
from gorilla.nn import GorillaConv
from torch import nn as nn
from torch.nn import functional as F

from gorilla3d.ops import furthest_point_sample
from ..pointnet2 import PointnetSAModule
from .vote_module import VoteModule


class VoteHead(nn.Module):
    r"""Bbox head of `Votenet <https://arxiv.org/abs/1904.09664>`_.
    Args:
        num_classes (int): The number of class.
        vote_moudule_cfg (dict): Config of VoteModule for point-wise votes.
        vote_aggregation_cfg (dict): Config of vote aggregation layer.
        feat_channels (tuple[int]): Convolution channels of
            prediction layer.
        D (int): Dimension of convolution. Defualt: 1.
        norm_cfg (dict): Config of BN in prediction layer.
    """
    def __init__(self,
                 num_classes,
                 num_sizes,
                 num_dir_bins,
                 vote_moudule_cfg=None,
                 vote_aggregation_cfg=None,
                 feat_channels=(128, 128),
                 D=1,
                 norm_cfg=dict(type="BN1d")):
        super(VoteHead, self).__init__()
        self.num_classes = num_classes
        self.num_proposal = vote_aggregation_cfg["num_point"]

        assert vote_aggregation_cfg["mlp_channels"][0] == \
            vote_moudule_cfg["in_channels"]

        self.vote_module = VoteModule(**vote_moudule_cfg)
        self.vote_aggregation = PointnetSAModule(**vote_aggregation_cfg)

        prev_channel = vote_aggregation_cfg["mlp_channels"][-1]
        conv_pred_list = list()
        for k in range(len(feat_channels)):
            conv_pred_list.append(
                GorillaConv(prev_channel,
                            feat_channels[k],
                            D=D,
                            padding=0,
                            norm_cfg=norm_cfg))
            prev_channel = feat_channels[k]
        self.conv_pred = nn.Sequential(*conv_pred_list)

        # Objectness scores (2), center residual (3),
        # heading class+residual (num_dir_bins*2),
        # size class+residual(num_sizes*4)
        conv_out_channel = (2 + 3 + num_dir_bins * 2 + num_sizes * 4 +
                            num_classes)
        self.conv_pred.add_module("conv_out",
                                  nn.Conv1d(prev_channel, conv_out_channel, 1))

    def init_weights(self):
        """Initialize weights of VoteHead."""
        pass

    def forward(self, feat_dict, sample_mod):
        """Forward pass.
        Note:
            The forward of VoteHead is devided into 4 steps:
                1. Generate vote_points from seed_points.
                2. Aggregate vote_points.
                3. Predict bbox and score.
                4. Decode predictions.(in VoteLoss)
        Args:
            feat_dict (dict): Feature dict from backbone.
            sample_mod (str): Sample mode for vote aggregation layer.
                valid modes are "vote", "seed" and "random".
        Returns:
            dict: Predictions of vote head.
        """
        assert sample_mod in ["vote", "seed", "random"]

        seed_points = feat_dict["fp_xyz"][-1]  # [B, 1024, 3]
        seed_features = feat_dict["fp_features"][-1]  # [B, 256, 1024]
        seed_indices = feat_dict["fp_indices"][-1]  # [B, 1024]

        # 1. generate vote_points from seed_points
        # vote_points: (B, num_vote, 3)
        # vote_features: (B, 256, num_vote)
        vote_points, vote_features = self.vote_module(seed_points,
                                                      seed_features)

        results = dict(seed_points=seed_points,
                       seed_indices=seed_indices,
                       vote_points=vote_points,
                       vote_features=vote_features)

        # 2. aggregate vote_points
        if sample_mod == "vote":
            # use fps in vote_aggregation
            sample_indices = None
        elif sample_mod == "seed":
            # FPS on seed and choose the votes corresponding to the seeds
            sample_indices = furthest_point_sample(
                seed_points,  # [B, 256]
                self.num_proposal)
        elif sample_mod == "random":
            # Random sampling from the votes
            batch_size, num_seed = seed_points.shape[:2]
            sample_indices = seed_points.new_tensor(torch.randint(
                0, num_seed, (batch_size, self.num_proposal)),
                                                    dtype=torch.int32)
        else:
            raise NotImplementedError

        vote_aggregation_ret = self.vote_aggregation(vote_points,
                                                     vote_features,
                                                     sample_indices)

        # aggregated_indices: (B, 256)
        # aggregated_points: (B, 256, 3)
        # features: (B, 128, 256)
        aggregated_points, features, aggregated_indices = vote_aggregation_ret
        results["aggregated_points"] = aggregated_points
        results["aggregated_indices"] = aggregated_indices

        # 3. predict bbox and score
        predictions = self.conv_pred(features)  # [B, 2+3+2+18*4+18, 256]
        results["predictions"] = predictions

        # 4. decode predictions (this will be processed in poster)
        return results
