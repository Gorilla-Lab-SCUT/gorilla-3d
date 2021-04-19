# Copyright (c) Gorilla-Lab. All rights reserved.
import functools

import torch
import torch.nn as nn

try:
    import spconv
    from spconv.modules import SparseModule
    MODULE = SparseModule
except:
    MODULE = nn.Module
    pass


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)

class ResContextBlock(MODULE):
    def __init__(self,
                 in_filters,
                 out_filters,
                 norm_fn=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
                 indice_key=None):
        super().__init__()

        self.branch1 = spconv.SparseSequential(
            norm_fn(out_filters),
            nn.LeakyReLU(),
            conv1x3(in_filters, out_filters, indice_key=indice_key),
            norm_fn(out_filters),
            nn.LeakyReLU(),
            conv3x1(out_filters, out_filters, indice_key=indice_key),
        )

        self.branch2 = spconv.SparseSequential(
            norm_fn(out_filters),
            nn.LeakyReLU(),
            conv3x1(in_filters, out_filters, indice_key=indice_key),
            norm_fn(out_filters),
            nn.LeakyReLU(),
            conv1x3(out_filters, out_filters, indice_key=indice_key),
        )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features,
                                           input.indices,
                                           input.spatial_shape,
                                           input.batch_size)
        
        result = self.branch1(identity)
        shortcut = self.branch2(identity)

        result.features += shortcut.features

        return result


class ResidualBlock(MODULE):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_fn=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
                 indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels,
                                  out_channels,
                                  kernel_size=1,
                                  bias=False))

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels,
                              out_channels,
                              kernel_size=3,
                              padding=1,
                              bias=False,
                              indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels,
                              out_channels,
                              kernel_size=3,
                              padding=1,
                              bias=False,
                              indice_key=indice_key))

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features,
                                           input.indices,
                                           input.spatial_shape,
                                           input.batch_size)

        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class VGGBlock(MODULE):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_fn=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
                 indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels,
                              out_channels,
                              kernel_size=3,
                              padding=1,
                              bias=False,
                              indice_key=indice_key))

    def forward(self, input):
        return self.conv_layers(input)


