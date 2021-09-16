# modified from https://github.com/AnTao97/dgcnn.pytorch
from typing import List, Union, Optional

import torch
import torch.nn as nn

from ...modules.dgcnn.util import *
from ...modules.dgcnn.transformer import *


class DGCNNPartSeg(nn.Module):
    def __init__(self,
                 cfg: dict,
                 seg_num_all: int,
                 aggregation_nodes: Optional[List[List[int]]] = [[3, 64, 64],
                                                                 [64, 64, 64],
                                                                 [64, 64]],
                 mask_channel: Optional[int] = 64,
                 mlp_nodes: Optional[List[int]] = [256, 256, 128],
                 dropout: Optional[Union[float,
                                         List[float]]] = [0.5, 0.5, 0.0]):
        """Author: shi.xian, liang.zhihao
        dgcnn part segmentation network

        Args:
            cfg (dict): the config dict
                k (int): Num of nearest neighbors
                emb_dims (int): Dimension of embeddings
            seg_num_all (int): Num of segmentation classes
            aggregation_nodes (Optional[List[List[int]]]):
                The num of nodes of dynamic layer (including input layer and output layer).
            mask_channel (Optional[int]): Features channels of mask vector to extract, Defaults to 64.
            mlp_nodes (Optional[[List[int]]):
                The num of nodes of mlp layer (including input layer and output layer).
            dropout (Optional[Union[List[float], float]], optional):
                Dropout ratio of each mlp, if given single float, then propogate to a list. Defaults to [0.5, 0.5, 0.0].
        """
        super().__init__()
        self.cfg = cfg
        self.k = cfg.get("k", 20)
        num_class = cfg.get("num_class", 16)
        emb_dims = cfg.get("emb_dims", 1024)
        seg_num_all = seg_num_all

        self.with_trans = cfg.get("with_trans", True)

        if self.with_trans:
            self.transform_net = TransformNet(cfg)

        # using knn and Conv2d to aggregate nearest neighbor features
        self.aggregation = DGCNNAggregation(aggregation_nodes, self.k)
        aggregation_dim = sum(
            [node_list[-1] for node_list in aggregation_nodes])
        self.media = nn.Sequential(
            nn.Conv1d(aggregation_dim, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims), nn.LeakyReLU(negative_slope=0.2))

        self.mask_conv = nn.Sequential(
            nn.Conv1d(num_class, mask_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(mask_channel), nn.LeakyReLU(negative_slope=0.2))

        # init mlp
        input_channel = aggregation_dim + emb_dims + mask_channel
        mlp_nodes.insert(0, input_channel)
        assert len(mlp_nodes) >= 2
        self.num_mlp = len(mlp_nodes) - 1

        # list wrapper dropout
        if isinstance(dropout, List):
            assert len(dropout) == self.num_mlp
        else:
            dropout[dropout] * self.num_mlp

        for idx, (in_channels, out_channels, drop) in enumerate(
                zip(mlp_nodes[:-1], mlp_nodes[1:], dropout)):
            setattr(
                self, f"conv{idx + 1}",
                nn.Sequential(
                    nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=1,
                              bias=False), nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(negative_slope=0.2), nn.Dropout(p=drop)))

        # output
        self.output = nn.Conv1d(mlp_nodes[-1],
                                seg_num_all,
                                kernel_size=1,
                                bias=False)

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> dict:
        """[summary]

        Args:
            points (torch.Tensor, [batch_size, 3, num_points]): input points
            mask (torch.Tensor, [batch_size, num_cat]): input shape category mask

        Returns:
            dict('input_pc', 'prediction')
            input_pc: [batch_size, input_channels, num_points]
            prediction: [batch_size, output_channels, num_points]: [the part segmentation probability of each category]
        """
        results = dict(input_pc=points)

        batch_size = points.size(0)
        num_points = points.size(2)

        x = points
        # with transformation
        if self.with_trans:
            x0 = get_graph_feature(x, k=self.k)
            t = self.transform_net(x0)
            x = x.transpose(2, 1)
            x = torch.bmm(x, t)
            x = x.transpose(2, 1)

        # knn aggregation
        _, aggregation_results = self.aggregation(x)
        x = torch.cat(aggregation_results, dim=1)

        # Conv1d to extract features
        x = self.media(x)  # [B, emb_dim, N]
        x = x.max(dim=-1, keepdim=True)[0]  # [B, emb_dim, 1]

        # extract mask features
        mask = mask.view(batch_size, -1, 1)  # [batch_size, num_cat, 1]
        mask = self.mask_conv(mask)  # [batch_size, mask_channel, 1]

        # propogate global feature to point-wise features
        x = torch.cat((x, mask),
                      dim=1)  # [batch_size, emb_dim + mask_channel, 1]
        x = x.repeat(
            1, 1,
            num_points)  # [batch_size, emb_dim + mask_channel, num_points]

        aggregation_results.insert(0, x)
        x = torch.cat(aggregation_results,
                      dim=1)  # [batch_size, channels, num_points]

        for conv_idx in range(self.num_mlp):
            x = getattr(self, f"conv{conv_idx + 1}")(x)

        # output partseg result
        x = self.output(x)

        results["prediction"] = x
        return results
