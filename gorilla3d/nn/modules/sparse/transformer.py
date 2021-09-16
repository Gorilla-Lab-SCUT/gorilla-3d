# Copyright (c) Gorilla-Lab. All rights reserved.
import math
import torch
import torch.nn as nn

import gorilla
from gorilla.nn import Transformer
# ! incomplete


def get_real_dense_from_sparse_tensor(sparse_tensor):
    bs = sparse_tensor.batch_size
    h, w, d = sparse_tensor.spatial_shape
    feat = sparse_tensor.features
    inds = sparse_tensor.indices.long()
    c = sparse_tensor.features.shape[-1]
    dense = feat.new_zeros([bs, c, h, w, d])  # [Bs, C, H, W, D]
    dense[inds[:, 0], :, inds[:, 1], inds[:, 2], inds[:, 3]] = c
    return dense


class TransformerSparse3D(Transformer):
    def forward(self, src, mask, query_embed, pos_embed):
        bs = src.shape[1]
        query_embed = query_embed.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query, Bs, dims]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # [H*W*D, Bs, C]
        mask = mask.flatten(1)  # [Bs, H*W*D]

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt,
                          memory,
                          memory_key_padding_mask=mask,
                          pos=pos_embed,
                          query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0)


class PositionEmbeddingSine3d(nn.Module):
    r"""
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, sparse_tensor):
        bs = sparse_tensor.batch_size
        inds = sparse_tensor.indices.long()
        features = sparse_tensor.features  # [N, C]

        max_length = torch.bincount(inds[:, 0]).max()
        feats = features.new_zeros(
            bs, max_length, features.shape[-1])  # [B, length_sequence, C]
        for b_idx in range(bs):
            ids = (inds[:, 0] == b_idx)
            feats[b_idx, :ids.sum(), :] = features[ids]

        return feats
