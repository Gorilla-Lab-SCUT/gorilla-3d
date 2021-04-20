from collections import OrderedDict

import torch
import torch.nn as nn

try:
    import spconv
    from spconv.modules import SparseModule
except:
    pass

from .block import ResidualBlock, VGGBlock


class UBlock(nn.Module):
    def __init__(self,
                 nPlanes,
                 norm_fn,
                 block_reps,
                 block,
                 indice_key_id=1,
                 return_blocks=False,):

        super().__init__()

        self.return_blocks = return_blocks
        self.nPlanes = nPlanes

        blocks = {
            f"block{i}":
            block(nPlanes[0],
                  nPlanes[0],
                  norm_fn,
                  indice_key=f"subm{indice_key_id}")
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]), nn.ReLU(),
                spconv.SparseConv3d(
                    nPlanes[0],
                    nPlanes[1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key=f"spconv{indice_key_id}"))

            self.u = UBlock(nPlanes[1:],
                            norm_fn,
                            block_reps,
                            block,
                            indice_key_id=indice_key_id + 1,
                            return_blocks=return_blocks)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]), nn.ReLU(),
                spconv.SparseInverseConv3d(
                    nPlanes[1],
                    nPlanes[0],
                    kernel_size=2,
                    bias=False,
                    indice_key=f"spconv{indice_key_id}"))

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail[f"block{i}"] = block(
                    nPlanes[0] * (2 - i),
                    nPlanes[0],
                    norm_fn,
                    indice_key=f"subm{indice_key_id}")
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self,
                input,
                previous_outputs=[]):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features,
                                           output.indices,
                                           output.spatial_shape,
                                           output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            if self.return_blocks:
                output_decoder, previous_outputs = self.u(output_decoder, previous_outputs)
            else:
                output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output.features = torch.cat(
                (identity.features, output_decoder.features), dim=1)

            output = self.blocks_tail(output)

        if self.return_blocks:
            previous_outputs.append(output)
            return output, previous_outputs
        else:
            return output

