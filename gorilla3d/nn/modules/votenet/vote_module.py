# Copyright (c) Gorilla-Lab. All rights reserved.
import torch
from torch import nn as nn

from gorilla.nn.conv import GorillaConv


class VoteModule(nn.Module):
    """Vote module.
    Generate votes from seed point features.
    Args:
        in_channels (int): Number of channels of seed point features.
        vote_per_seed (int): Number of votes generated from each seed point.
        conv_channels (tuple[int]): Out channels of vote
            generating convolution.
        norm_cfg (dict): Config of normalization.
            Default: dict(type="BN1d").
        norm_feats (bool): Whether to normalize features.
            Default: True.
    """
    def __init__(self,
                 in_channels,
                 vote_per_seed=1,
                 conv_channels=(16, 16),
                 norm_cfg=dict(type="BN1d"),
                 norm_feats=True):
        super().__init__()
        self.in_channels = in_channels
        self.vote_per_seed = vote_per_seed
        self.norm_feats = norm_feats

        prev_channels = in_channels
        vote_conv_list = list()
        for k in range(len(conv_channels)):
            vote_conv_list.append(
                GorillaConv(prev_channels,
                            conv_channels[k],
                            1,
                            padding=0,
                            D=1,
                            norm_cfg=norm_cfg))
            prev_channels = conv_channels[k]
        self.vote_conv = nn.Sequential(*vote_conv_list)

        # conv_out predicts coordinate and residual features
        out_channel = (3 + in_channels) * self.vote_per_seed
        self.conv_out = nn.Conv1d(prev_channels, out_channel, 1)

    def forward(self, seed_points, seed_feats):
        """forward.
        Args:
            seed_points (torch.Tensor): Coordinate of the seed
                points in shape (B, N, 3).
            seed_feats (torch.Tensor): Features of the seed points in shape
                (B, C, N).
        Returns:
            tuple[torch.Tensor]:
                - vote_points: Voted xyz based on the seed points \
                    with shape (B, M, 3), ``M=num_seed*vote_per_seed``.
                - vote_features: Voted features based on the seed points with \
                    shape (B, C, M) where ``M=num_seed*vote_per_seed``, \
                    ``C=vote_feature_dim``.
        """
        batch_size, feat_channels, num_seed = seed_feats.shape
        num_vote = num_seed * self.vote_per_seed
        x = self.vote_conv(seed_feats)  # [B, 256, num_seed]

        # [B, (3+out_dim)*vote_per_seed, num_seed]
        votes = self.conv_out(x)

        # [B, num_seed, vote_per_seed, (3+out_dim)]
        votes = votes.transpose(2, 1).view(batch_size, num_seed,
                                           self.vote_per_seed, -1)
        offset = votes[:, :, :, 0:3]  # [B, num_seed, vote_per_seed, 3]
        res_feats = votes[:, :, :, 3:]  # [B, num_seed, vote_per_seed, in_dim]

        vote_points = (seed_points.unsqueeze(2) +
                       offset).contiguous()  # [B, num_seed, vote_per_seed, 3]
        vote_points = vote_points.view(batch_size, num_vote,
                                       3)  # [B, num_vote, 3]
        vote_feats = (
            seed_feats.transpose(2, 1).unsqueeze(2) +
            res_feats).contiguous()  # [B, num_seed, vote_per_seed, in_dim]
        # [B, in_dim, num_vote]
        vote_feats = vote_feats.view(batch_size, num_vote,
                                     feat_channels).transpose(2,
                                                              1).contiguous()

        if self.norm_feats:
            features_norm = torch.norm(vote_feats, p=2, dim=1)
            vote_feats = vote_feats.div(features_norm.unsqueeze(1))

        return vote_points, vote_feats
