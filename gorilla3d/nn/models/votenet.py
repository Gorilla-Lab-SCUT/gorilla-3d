# Copyright (c) Gorilla-Lab. All rights reserved.
import torch
import torch.nn as nn

from ..modules import PointNet2SASSG, VoteHead


class VoteNet(nn.Module):
    r"""Define VoteNet
        https://arxiv.org/pdf/1904.09664.pdf

    Args:
        backbone (dict): config dict of VoteNet's backbone(PointNet2SASSG)
        bbox_head (dict): config dict of VoteNet's head
    """
    def __init__(self,
                 backbone=dict(in_channels=4,
                               num_points=(2048, 1024, 512, 256),
                               radius=(0.2, 0.4, 0.8, 1.2),
                               num_samples=(64, 32, 16, 16),
                               sa_channels=((64, 64, 128), (128, 128, 256),
                                            (128, 128, 256), (128, 128, 256)),
                               fp_channels=((256, 256), (256, 256)),
                               norm_cfg=dict(type="BN2d"),
                               pool_mod="max"),
                 bbox_head=dict(vote_moudule_cfg=dict(
                     in_channels=256,
                     vote_per_seed=1,
                     conv_channels=(256, 256),
                     D=1,
                     norm_cfg=dict(type="BN1d"),
                     norm_feats=True),
                                vote_aggregation_cfg=dict(
                                    num_point=256,
                                    radius=0.3,
                                    num_sample=16,
                                    mlp_channels=[256, 128, 128, 128],
                                    use_xyz=True,
                                    normalize_xyz=True),
                                feat_channels=(128, 128),
                                D=1,
                                norm_cfg=dict(type="BN1d")),
                 train_sample_mod="vote",
                 test_sample_mod="random"):
        super(VoteNet, self).__init__()
        self.backbone = PointNet2SASSG(**backbone)
        self.bbox_head = VoteHead(**bbox_head)
        self.train_sample_mod = train_sample_mod
        self.test_sample_mod = test_sample_mod

    def forward(self, data, be_train=True):
        r"""Calls either forward_train or forward_test depending on be_train.
            Note this setting will change the expected inputs.
        """
        if be_train:
            return self.forward_train(data)
        else:
            with torch.no_grad():
                return self.forward_test(data)

    def forward_train(self, data):
        r"""Forward of training.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            sample_metas (list): Image metas.

        Returns:
            dict: Pridict results with gt.
        """
        points = data["points"]
        sample_metas = data["sample_metas"]

        x = self.backbone(points)
        bbox_preds = self.bbox_head(x, self.train_sample_mod)
        predicts = dict(bbox_preds=bbox_preds, points=points)
        return predicts

    def forward_test(self, data):
        """Forward of testing.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            sample_metas (list): Image metas.

        Returns:
            list: Predicted 3d boxes.
        """
        points = data["points"]
        sample_metas = data["sample_metas"]

        x = self.backbone(points)
        bbox_preds = self.bbox_head(x, self.test_sample_mod)
        predicts = dict(bbox_preds=bbox_preds,
                        points=points,
                        sample_metas=sample_metas)
        return predicts

    def init_weights(self, pretrained=None):
        r"""Initialize weights of detector."""
        super(VoteNet, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.bbox_head.init_weights()
