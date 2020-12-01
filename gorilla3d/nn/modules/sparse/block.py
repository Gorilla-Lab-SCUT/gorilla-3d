# Copyright (c) Gorilla-Lab. All rights reserved.
import torch
import torch.nn as nn

try:
    import spconv
    from spconv.modules import SparseModule
    MODULE = SparseModule
except:
    MODULE = nn.Module
    pass


def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 1, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
    )

def double_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
    )

def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
    )

def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SparseConv3d(in_channels, out_channels, 3, (2, 2, 2), padding=1, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
    )


class ResidualBlock(MODULE):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class VGGBlock(MODULE):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        return self.conv_layers(input)



