# Copyright (c) Gorilla-Lab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_min

class DynamicConv(nn.Module):

    def __init__(self,
                 in_channel: int,
                 out_channel: int=1,
                 hid_channel: int=8,
                 n_kernels: int=3,
                 with_bias: bool=True,
                 with_norm: bool=False,
                 use_coords: bool=True):
        """Author: liang.zhihao
        Dynamic convolution

        Args:
            in_channel (int): input channel
            out_channel (int, optional): output channel. Defaults to 1.
            hid_channel (int, optional): hidden channel. Defaults to 8.
            n_kernels (int, optional): the number of conv kernel(layers). Defaults to 3.
            with_bias (bool, optional): with bias or not. Defaults to True.
            with_norm (bool, optional): with norm(layernorm) or not. Defaults to False.
            use_coords (bool, optional): features with coordinates or not. Defaults to True.
        """
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hid_channel = hid_channel
        self.n_kernels = n_kernels
        self.with_bias = with_bias
        self.with_norm = with_norm
        self.use_coords = use_coords

        # calculate parameters_count
        n_hid = n_kernels - 1
        if self.use_coords:
            in_channel += 3
        self.parameters_count = (
            np.array([in_channel] + [hid_channel] * n_hid) * \
            np.array([hid_channel] * n_hid + [out_channel]) \
        ).sum()

        if with_bias:
            self.parameters_count += n_hid * hid_channel + out_channel

        # get the dynamic conv parameters output layer
        self.dynamic_layer = nn.Linear(self.in_channel, self.parameters_count)

        # get the channels of each layer
        self.layers_channel = [in_channel] + [hid_channel] * n_hid + [out_channel]

        # activation function
        self.activation = nn.ReLU(inplace=True)

        if self.with_norm:
            for i in range(n_hid - 1):
                setattr(self, "norm{}".format(i), nn.LayerNorm(hid_channel))

        self.out_layer = nn.Linear(hid_channel, out_channel)
        

    def forward(self,
                prop_feats: torch.Tensor,
                features: torch.Tensor,
                coords: torch.Tensor,
                cluster_centers: torch.Tensor,
                proposals_idx: torch.Tensor,
                batch_offsets: torch.Tensor) -> torch.Tensor:
        r"""Author: liang.zhihao
        dynamic convolution forward

        Args:
            prop_feats (torch.Tensor): (num_prop, in_channel) proposals features
            features (torch.Tensor): (num_all, in_channel) features of point cloud
            coords (torch.Tensor): (num_all, 3) coordinates of point cloud
            cluster_centers (torch.Tensor): (num_prop, 3) centers of proposals
            proposals_idx (torch.Tensor): (sum_points, 2) dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            batch_offsets (torch.Tensor): (B + 1), start and end offset of batch ids

        Returns:
            batch_features_list [torch.Tensor]: [(num_batch_prop, num_batch, out_c)], list contained batches' output features
            batch_proposals_ids [torch.Tensor]: [(num_batch_prop)], list contained batches' number of proposals
        """

        point_ids_min = scatter_min(proposals_idx[:, 1].cuda().long(), proposals_idx[:, 0].cuda().long(), dim=0)[0] # (num_prop)

        # get the dynamic convolution weight
        parameters = self.dynamic_layer(prop_feats) # (num_prop, parameters_count)

        channel = features.shape[-1]

        batch_features_list = []
        batch_proposals_ids = []

        for b_idx, (b_start, b_end) in enumerate(zip(batch_offsets[:-1], batch_offsets[1:])):
            b_ids = ((b_start <= point_ids_min) & (point_ids_min < b_end)) # (num_prop)
            num_batch = b_end - b_start
            num_batch_prop = b_ids.sum()
            batch_proposals_ids.append(torch.where(b_ids)[0]) # (num_batch_prop)

            batch_features = features[b_start:b_end, :].expand(num_batch_prop, num_batch, channel)
            batch_coords = coords[b_start:b_end, :].expand(num_batch_prop, num_batch, 3)
            batch_cluster_centers = cluster_centers[b_ids, None, :] # (num_batch_prop, 1, 3)
            batch_coords = batch_coords - batch_cluster_centers # (num_batch_prop, num_batch, 3)
            batch_features = torch.cat([batch_features, batch_coords], dim=2) # (num_batch_prop, num_batch, channel + 3)

            # the parameters index
            start, end = 0, 0
            for i in range(self.n_kernels):
                # get the conv parameters (the bmm matrix)
                in_c = self.layers_channel[i]
                out_c = self.layers_channel[i + 1]
                # conv(bmm)
                end += in_c * out_c
                conv_mat = parameters[b_ids, start: end].view(num_batch_prop, in_c, out_c)
                batch_features = torch.bmm(batch_features, conv_mat) # (num_batch_prop, num_batch, out_c)
                start = end
                # add bias
                if self.with_bias:
                    end += out_c
                    bias = parameters[b_ids, start: end].view(-1, 1, out_c) # (num_batch_prop, 1, out_c)
                    batch_features = batch_features + bias # (num_batch_prop, num_batch, out_c)
                    start = end
                if i != (self.n_kernels - 1): # last layer without norm
                    # activation
                    batch_features = self.activation(batch_features) # (num_batch_prop, num_batch, out_c)
                    # norm
                    if self.with_norm:
                        batch_features = getattr(self, "norm{}".format(i + 1))(batch_features) # (num_batch_prop, num_batch, out_c)
            batch_features_list.append(batch_features)

        return batch_features_list, batch_proposals_ids

