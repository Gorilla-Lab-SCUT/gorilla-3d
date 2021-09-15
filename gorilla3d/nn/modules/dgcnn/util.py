# Copyright (c) Gorilla-Lab. All rights reserved.
import enum
from typing import List

import torch
import torch.nn as nn


def knn(x, k):
    r"""
    It is a function to collect k neighbors. 
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # [batch_size, num_points, k]
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    r"""
    This is the function of dgcnn to calculate the neighborhood feature.
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # [batch_size, num_points, k)]
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1,
                                                               1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # [batch_size, num_points, num_dims]
    feature = x.view(batch_size * num_points,
                     -1)[idx, :]  # [batch_size*num_points, num_dims]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1,
                                                         2).contiguous()

    return feature  # [batch_size, 2*num_dims, num_points, k]


class DGCNNAggregation(nn.Module):
    def __init__(self, nodes: List[List[int]], k: int = 20) -> None:
        """Author: liang.zhihao, shi.xian
        DGCNN knn aggregation forward

        Args:
            nodes (List[List[int]): layers of DGCNN aggregation.
            k (int, optional): number of knn. Defaults to 20.
        """
        super().__init__()
        self.k = k

        self.nodes = nodes

        conv_bias = 1
        for node_list in nodes:
            assert len(node_list) >= 2
            for idx, (in_channels, out_channels) in enumerate(
                    zip(node_list[:-1], node_list[1:])):
                # NOTE: the first layer of DGCNN aggregation will concat the point and its
                #       neighbors' features so the in_channels will be twice of itself
                if idx == 0:
                    in_channels *= 2
                setattr(
                    self, f"conv{conv_bias}",
                    nn.Sequential(
                        nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=1,
                                  bias=False), nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU(negative_slope=0.2)))
                conv_bias += 1

    def forward(self, x):
        # aggregation results container
        results = []
        conv_bias = 1
        for aggre_idx, node_list in enumerate(self.nodes):
            x = get_graph_feature(x, k=self.k)  # knn search and cat features
            # conv2d forward
            num_conv = len(node_list) - 1
            for conv_idx in range(num_conv):
                x = getattr(self, f"conv{conv_bias}")(x)
                conv_bias += 1
            # maxpooling
            x = x.max(dim=-1, keepdim=False)[0]
            results.append(x)

        return x, results

    def __repr__(self) -> str:
        content = "DGCNNAggregation(\n"
        bias = 1
        num_aggre = len(self.nodes) - 1
        for idx, node_list in enumerate(self.nodes):
            # show get_graph_feature
            content += f"  get_graph_feature(k={self.k})\n"
            num_conv = len(node_list) - 1
            for _ in range(num_conv):
                content += "  "
                content += str(self._modules[f"conv{bias}"]).replace(
                    "\n", "\n    ")
                bias += 1
            # show maxpooling
            content += "\n  MaxPooling()\n"
            if idx == num_aggre:
                content += ")"
        return content
