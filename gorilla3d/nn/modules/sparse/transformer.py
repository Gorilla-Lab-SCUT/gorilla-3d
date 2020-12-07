# Copyright (c) Gorilla-Lab. All rights reserved.
import math
import torch
import torch.nn as nn

import gorilla
from gorilla.nn import Transformer


def get_real_dense_from_sparse_tensor(sparse_tensor):
    bs = sparse_tensor.batch_size
    h, w, d = sparse_tensor.spatial_shape
    feat = sparse_tensor.features
    inds = sparse_tensor.indices.long()
    c = sparse_tensor.features.shape[-1]
    dense = feat.new_zeros([bs, c, h, w, d]) # (Bs, C, H, W, D)
    dense[inds[:, 0], :, inds[:, 1], inds[:, 2], inds[:, 3]] = c
    return dense


class TransformerSparse3D(Transformer):
    def forward(self, src, mask, query_embed, pos_embed):
        bs = src.shape[1]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1) # (num_query, Bs, dims)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1) # (H*W*D, Bs, C)
        mask = mask.flatten(1) # (Bs, H*W*D)

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
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
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
        h, w, d = sparse_tensor.spatial_shape
        inds = sparse_tensor.indices.long()
        features = sparse_tensor.features # (N, C)
        mask = features.new_ones([bs, h, w, d], dtype=torch.bool) # (Bs, H, W, D)
        mask[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]] = False

        not_mask = ~mask
        z_embed = not_mask.cumsum(1, dtype=torch.float32) # (B, H, W, D)
        y_embed = not_mask.cumsum(2, dtype=torch.float32) # (B, H, W, D)
        x_embed = not_mask.cumsum(3, dtype=torch.float32) # (B, H, W, D)
        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale # (B, H, W, D)
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale # (B, H, W, D)
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale # (B, H, W, D)

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=features.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, :, None] / dim_t # (B, H, W, D, num_pos_feats)
        pos_y = y_embed[:, :, :, :, None] / dim_t # (B, H, W, D, num_pos_feats)
        pos_z = z_embed[:, :, :, :, None] / dim_t # (B, H, W, D, num_pos_feats)
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4) # (B, H, W, D, num_pos_feats)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4) # (B, H, W, D, num_pos_feats)
        pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4) # (B, H, W, D, num_pos_feats)
        pos = torch.cat((pos_z, pos_y, pos_x), dim=4).permute(0, 4, 1, 2, 3) # (B, 3*num_pos_feats, H, W, D)

        max_length = torch.bincount(inds[:, 0]).max()
        feats = features.new_zeros(bs, max_length, features.shape[-1]) # (B, length_sequence, C)
        pos_embed = features.new_zeros(bs, pos.shape[1], max_length) # (B, 3*num_pos_feats, length_sequence)
        for b_idx in range(bs):
            ids = (inds[:, 0] == b_idx)
            x_ids = inds[ids, 1]
            y_ids = inds[ids, 2]
            z_ids = inds[ids, 3]
            feats[b_idx, :ids.sum(), :] = features[ids]
            pos_embed[b_idx, :, :ids.sum()] = pos[b_idx, :, x_ids, y_ids, z_ids]

        return feats, pos_embed

    # def forward(self, mask, device):
    #     assert mask is not None
    #     not_mask = ~mask
    #     z_embed = not_mask.cumsum(1, dtype=torch.float32) # (B, H, W, D)
    #     y_embed = not_mask.cumsum(2, dtype=torch.float32) # (B, H, W, D)
    #     x_embed = not_mask.cumsum(3, dtype=torch.float32) # (B, H, W, D)
    #     if self.normalize:
    #         eps = 1e-6
    #         z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale # (B, H, W, D)
    #         y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale # (B, H, W, D)
    #         x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale # (B, H, W, D)

    #     dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
    #     dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

    #     pos_x = x_embed[:, :, :, :, None] / dim_t # (B, H, W, D, num_pos_feats)
    #     pos_y = y_embed[:, :, :, :, None] / dim_t # (B, H, W, D, num_pos_feats)
    #     pos_z = z_embed[:, :, :, :, None] / dim_t # (B, H, W, D, num_pos_feats)
    #     pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4) # (B, H, W, D, num_pos_feats)
    #     pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4) # (B, H, W, D, num_pos_feats)
    #     pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4) # (B, H, W, D, num_pos_feats)
    #     pos = torch.cat((pos_z, pos_y, pos_x), dim=4).permute(0, 4, 1, 2, 3) # (B, num_pos_feats, H, W, D)
    #     return pos
