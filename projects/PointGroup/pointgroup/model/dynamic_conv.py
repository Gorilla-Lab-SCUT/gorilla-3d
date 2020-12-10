import torch
import torch.nn as nn
import numpy as np

class DynamicConv(nn.Module):

    def __init__(self,
                 in_channel: int,
                 out_channel: int=1,
                 hid_channel: int=8,
                 n_kernels: int=3,
                 with_bias: bool=True,
                 with_norm: bool=False,
                 use_coords: bool=True):
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
        

    def forward(self, prop_feats: torch.Tensor, all_features: torch.Tensor) -> torch.Tensor:
        """dynamic convolution forward

        Args:
            pro_features (torch.Tensor): proposals features (num_prop, in_channel)
            roi_features (torch.Tensor): all region features (num_all, in_channel)

        Returns:
            torch.Tensor: (num_prop, num_all, out_channel), the match score
        """
        # the parameters index
        start, end = 0, 0

        parameters = self.dynamic_layer(prop_feats) # (num_prop, parameters_count)

        num_prop = prop_feats.shape[0]
        num_all = all_features.shape[0]
        channel = self.in_channel
        if self.use_coords:
            channel += 3
        features = all_features.expand(num_prop, num_all, channel) # (num_prop, num_all, in_channel)

        for i in range(self.n_kernels):
            # get the conv parameters (the bmm matrix)
            in_c = self.layers_channel[i]
            out_c = self.layers_channel[i + 1]
            # conv(bmm)
            end += in_c * out_c
            conv_mat = parameters[:, start: end].view(num_prop, in_c, out_c) # (num_prop, in_c, out_c)
            features = torch.bmm(features, conv_mat) # (num_prob, num_all, out_c)
            start = end
            # add bias
            if self.with_bias:
                end += out_c
                bias = parameters[:, start: end].view(-1, 1, out_c) # (num_prop, 1, out_c)
                features = features + bias # (num_prop, num_all, out_c)
                start = end
            if i != (self.n_kernels - 1): # last layer without norm
                # activation
                features = self.activation(features) # (num_prop, num_all, out_c)
                # norm
                if self.with_norm:
                    features = getattr(self, "norm{}".format(i + 1))(features) # (num_prop, num_all, out_c)

        return features

