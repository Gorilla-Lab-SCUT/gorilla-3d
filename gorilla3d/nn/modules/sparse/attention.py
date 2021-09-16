# Copyright (c) Gorilla-Lab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim,
                                               num_heads,
                                               dropout=dropout,
                                               **kwargs)

    def forward(self, batch_ids: torch.Tensor,
                features: torch.Tensor) -> torch.Tensor:
        r"""
        The concat length features attention forward

        Args:
            batch_ids (torch.Tensor, [N]): batch idx of each features
            features (torch.Tensor, [N, C]): input features

        Returns:
            attn_outputs (torch.Tensor, [N, C']): output attention features
        """
        assert batch_ids.dim() == 1
        assert features.dim() == 2
        # get the longest embedding according to batch_ids
        batch_lengthes = torch.bincount(batch_ids)  # [bs]
        bs = len(batch_lengthes)
        length = batch_lengthes.max()

        padding_features = []
        key_padding_mask = torch.zeros([bs, length]).bool().to(
            features.device)  # [bs, length]
        for batch_idx in range(bs):
            # get padding faeture
            ids = (batch_ids == batch_idx)
            batch_feature = features[ids]  # [num_batch, C]
            padding_length = length - ids.sum()
            padding_feature = F.pad(batch_feature, (0, 0, 0, padding_length),
                                    "constant",
                                    value=0)  # [length, C]
            padding_features.append(padding_feature)
            # get the padding mask
            if padding_length != 0:
                key_padding_mask[batch_idx, -padding_length:] = True
        padding_features = torch.stack(padding_features)  # [bs, length, C]
        padding_features = padding_features.permute(1, 0, 2)  # [length, bs, C]

        # self attention forward
        attn_output = self.self_attn(padding_features, padding_features,
                                     padding_features,
                                     key_padding_mask)[0]  # [length, bs, C_o]

        # concat the output features
        attn_outputs = []
        for batch_idx in range(bs):
            # get padding faeture
            ids = (batch_ids == batch_idx)
            num_batch = int(ids.sum())
            batch_output_feature = attn_output[:num_batch,
                                               batch_idx, :]  # [num_batch, C_o]
            attn_outputs.append(batch_output_feature)

        attn_outputs = torch.cat(attn_outputs)  # [N, C_o]
        assert attn_outputs.shape[0] == features.shape[0]
        return attn_outputs
