# Copyright (c) Gorilla-Lab. All rights reserved.
import functools
from typing import Callable, Dict, Optional, Union

import gorilla
import torch
import torch.nn as nn

try:
    import spconv
    from spconv.modules import SparseModule
    MODULE = SparseModule
except:
    MODULE = nn.Module
    pass


def conv1x3(in_planes: int,
            out_planes: int,
            stride: int = 1,
            indice_key: Optional[str] = None):
    return spconv.SubMConv3d(in_planes,
                             out_planes,
                             kernel_size=(1, 3, 3),
                             stride=stride,
                             padding=(0, 1, 1),
                             bias=False,
                             indice_key=indice_key)


def conv3x1(in_planes: int,
            out_planes: int,
            stride: int = 1,
            indice_key: Optional[str] = None):
    return spconv.SubMConv3d(in_planes,
                             out_planes,
                             kernel_size=(3, 1, 3),
                             stride=stride,
                             padding=(1, 0, 1),
                             bias=False,
                             indice_key=indice_key)


class AsymResidualBlock(MODULE):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_fn: Union[Callable,
                                Dict] = functools.partial(nn.BatchNorm1d,
                                                          eps=1e-4,
                                                          momentum=0.1),
                 indice_key: Optional[str] = None,
                 normalize_before: bool = False):
        super().__init__()

        if isinstance(norm_fn, Dict):
            norm_caller = gorilla.nn.get_torch_layer_caller(
                norm_fn.pop("type"))
            norm_fn = functools.partial(norm_caller, **norm_fn)

        if normalize_before:
            self.conv_1 = spconv.SparseSequential(
                norm_fn(in_channels),
                nn.LeakyReLU(),
                conv3x1(in_channels, out_channels, indice_key=indice_key),
                norm_fn(out_channels),
                nn.LeakyReLU(),
                conv1x3(out_channels, out_channels, indice_key=indice_key),
            )
            self.conv_2 = spconv.SparseSequential(
                norm_fn(in_channels),
                nn.LeakyReLU(),
                conv1x3(in_channels, out_channels, indice_key=indice_key),
                norm_fn(out_channels),
                nn.LeakyReLU(),
                conv3x1(out_channels, out_channels, indice_key=indice_key),
            )
        else:
            self.conv_1 = spconv.SparseSequential(
                conv3x1(in_channels, out_channels, indice_key=indice_key),
                norm_fn(out_channels),
                nn.LeakyReLU(),
                conv1x3(out_channels, out_channels, indice_key=indice_key),
                norm_fn(out_channels),
                nn.LeakyReLU(),
            )
            self.conv_2 = spconv.SparseSequential(
                conv1x3(in_channels, out_channels, indice_key=indice_key),
                norm_fn(out_channels),
                nn.LeakyReLU(),
                conv3x1(out_channels, out_channels, indice_key=indice_key),
                norm_fn(out_channels),
                nn.LeakyReLU(),
            )
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        output = self.conv_1(input)
        output.features += self.conv_2(input).features

        return output


class ResidualBlock(MODULE):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_fn: Union[Callable,
                                Dict] = functools.partial(nn.BatchNorm1d,
                                                          eps=1e-4,
                                                          momentum=0.1),
                 indice_key: Optional[str] = None,
                 normalize_before: bool = True):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels,
                                  out_channels,
                                  kernel_size=1,
                                  bias=False))

        if isinstance(norm_fn, Dict):
            norm_caller = gorilla.nn.get_torch_layer_caller(
                norm_fn.pop("type"))
            norm_fn = functools.partial(norm_caller, **norm_fn)

        if normalize_before:
            self.conv_branch = spconv.SparseSequential(
                norm_fn(in_channels), nn.ReLU(),
                spconv.SubMConv3d(in_channels,
                                  out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  bias=False,
                                  indice_key=indice_key),
                norm_fn(out_channels), nn.ReLU(),
                spconv.SubMConv3d(out_channels,
                                  out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  bias=False,
                                  indice_key=indice_key))
        else:
            self.conv_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels,
                                  out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  bias=False,
                                  indice_key=indice_key),
                norm_fn(out_channels), nn.ReLU(),
                spconv.SubMConv3d(out_channels,
                                  out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  bias=False,
                                  indice_key=indice_key),
                norm_fn(out_channels), nn.ReLU())

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices,
                                           input.spatial_shape,
                                           input.batch_size)

        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class VGGBlock(MODULE):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_fn: Union[Callable,
                                Dict] = functools.partial(nn.BatchNorm1d,
                                                          eps=1e-4,
                                                          momentum=0.1),
                 indice_key: Optional[str] = None,
                 normalize_before: bool = True):
        super().__init__()

        if isinstance(norm_fn, Dict):
            norm_caller = gorilla.nn.get_torch_layer_caller(
                norm_fn.pop("type"))
            norm_fn = functools.partial(norm_caller, **norm_fn)

        if normalize_before:
            self.conv_layers = spconv.SparseSequential(
                norm_fn(in_channels), nn.ReLU(),
                spconv.SubMConv3d(in_channels,
                                  out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  bias=False,
                                  indice_key=indice_key))
        else:
            self.conv_layers = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels,
                                  out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  bias=False,
                                  indice_key=indice_key), norm_fn(in_channels),
                nn.ReLU())

    def forward(self, input):
        return self.conv_layers(input)
