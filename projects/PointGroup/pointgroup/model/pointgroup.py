"""
PointGroup
Written by Li Jiang
"""
import sys
import functools
from typing import Dict, List

import spconv
import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_min, scatter_mean, scatter_max

import pointgroup_ops
import gorilla
import gorilla.nn as gn
import gorilla3d.nn as g3n

@gorilla.MODELS.register_module()
class PointGroup(nn.Module):
    def __init__(self,
                 input_channel: int=3,
                 use_coords: bool=True,
                 blocks: int=5,
                 media: int=3,
                 classes: int=20,
                 block_reps: int=2,
                 score_scale: int=50,
                 score_fullscale: int=14,
                 score_mode: int=4,
                 prepare_epochs: int=128,
                 cluster_cfg: Dict = dict(
                    radius=0.04,
                    radius_shift=0.03,
                    mean_active=50,
                    shift_mean_active=300,
                    npoint_thresh=50
                 ),
                 fix_module: List[str]=[]):
        super().__init__()
        self.score_scale = score_scale
        self.score_fullscale = score_fullscale
        self.score_mode = score_mode

        self.prepare_epochs = prepare_epochs

        self.fix_module = fix_module

        self.cluster_radius = cluster_cfg["radius"]
        self.cluster_radius_shift = cluster_cfg["radius_shift"]
        self.cluster_mean_active = cluster_cfg["mean_active"]
        self.cluster_shift_mean_active = cluster_cfg["shift_mean_active"]
        self.cluster_npoint_thresh = cluster_cfg["npoint_thresh"]

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        block = g3n.ResidualBlock

        if use_coords:
            input_channel += 3

        #### backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, media, kernel_size=3, padding=1, bias=False, indice_key="subm1")
        )

        block_list = [media * (i + 1) for i in range(blocks)]
        self.unet = g3n.UBlock(block_list, norm_fn, block_reps, block, indice_key_id=1)

        # #### self attention module
        # self.self_attn = nn.MultiheadAttention(m * blocks, 8)

        self.output_layer = spconv.SparseSequential(
            norm_fn(media),
            nn.ReLU()
        )

        #### semantic segmentation
        self.linear = nn.Linear(media, classes) # bias(default): True

        #### offset
        self.offset = nn.Sequential(
            nn.Linear(media, media, bias=True),
            norm_fn(media),
            nn.ReLU()
        )
        self.offset_linear = nn.Linear(media, 3, bias=True)

        #### score branch
        self.score_unet = g3n.UBlock([media, 2*media], norm_fn, 2, block, indice_key_id=1)
        self.score_outputlayer = spconv.SparseSequential(
            norm_fn(media),
            nn.ReLU()
        )
        self.score_linear = nn.Linear(media, 1)

        self.apply(self.set_bn_init)

        #### fix parameter
        self.module_map = {"input_conv": self.input_conv, "unet": self.unet, "output_layer": self.output_layer,
                           "linear": self.linear, "offset": self.offset, "offset_linear": self.offset_linear,
                           "score_unet": self.score_unet, "score_outputlayer": self.score_outputlayer, "score_linear": self.score_linear}
        
        for m in self.fix_module:
            # self.module_map[m].eval()
            mod = self.module_map[m]
            mod.eval()
            for param in mod.parameters():
                param.requires_grad = False

    @staticmethod
    def freeze_bn(module):
        for name, child in module._modules.items():
            if child is not None:
                PointGroup.freeze_bn(child)
            if isinstance(child, nn.BatchNorm1d):
                if hasattr(child, "weight"):
                    child.weight.requires_grad_(False)
                if hasattr(child, "bias"):
                    child.bias.requires_grad_(False)

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            try:
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0.0)
            except:
                pass


    def clusters_voxelization(self, clusters_idx, feats, coords, fullscale, scale, mode):
        """
        :param clusters_idx: [SumNPoint, 2], int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
        :param feats: [N, C], float, cuda
        :param coords: [N, 3], float, cuda
        :return:
        """
        c_idxs = clusters_idx[:, 1].cuda()
        clusters_feats = feats[c_idxs.long()]
        clusters_coords = coords[c_idxs.long()]

        clusters_coords_mean = scatter_mean(clusters_coords, clusters_idx[:, 0].cuda().long(), dim=0) # [nCluster, 3], float

        clusters_coords_mean = torch.index_select(clusters_coords_mean, 0, clusters_idx[:, 0].cuda().long())  # [sum_points, 3], float
        clusters_coords -= clusters_coords_mean

        clusters_coords_min = scatter_min(clusters_coords, clusters_idx[:, 0].cuda().long(), dim=0)[0] # [nCluster, 3], float
        clusters_coords_max = scatter_max(clusters_coords, clusters_idx[:, 0].cuda().long(), dim=0)[0] # [nCluster, 3], float

        clusters_scale = 1 / ((clusters_coords_max - clusters_coords_min) / fullscale).max(1)[0] - 0.01  # [nCluster], float
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        min_xyz = clusters_coords_min * clusters_scale.unsqueeze(-1)  # [nCluster, 3], float
        max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

        clusters_scale = torch.index_select(clusters_scale, 0, clusters_idx[:, 0].cuda().long())

        clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

        range = max_xyz - min_xyz
        offset = - min_xyz
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        clusters_coords += offset
        assert clusters_coords.shape.numel() == ((clusters_coords >= 0) * (clusters_coords < fullscale)).sum()

        clusters_coords = clusters_coords.long()
        clusters_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), clusters_coords.cpu()], 1)  # [sum_points, 1 + 3]

        out_coords, inp_map, out_map = pointgroup_ops.voxelization_idx(clusters_coords, int(clusters_idx[-1, 0]) + 1, mode)
        # output_coords: M * [1 + 3] long
        # input_map: sum_points int
        # output_map: M * [maxActive + 1] int

        out_feats = pointgroup_ops.voxelization(clusters_feats, out_map.cuda(), mode)  # [M, C], float, cuda

        spatial_shape = [fullscale] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats, out_coords.int().cuda(), spatial_shape, int(clusters_idx[-1, 0]) + 1)

        return voxelization_feats, inp_map


    def forward(self, input, input_map, coords, batch_idxs, epoch, semantic_only=False):
        """
        :param input_map: [N], int, cuda
        :param coords: [N, 3], float, cuda
        :param batch_idxs: [N], int, cuda
        :param coords_offsets: [B, 3], int, cuda
        """
        ret = {}
        timer = gorilla.Timer()

        for m in self.fix_module:
            # self.module_map[m].eval()
            mod = self.module_map[m]
            mod.eval()

        output = self.input_conv(input)
        output = self.unet(output)

        output = self.output_layer(output)
        output_feats = output.features[input_map.long()] # [N, m]

        #### semantic segmentation
        semantic_scores = self.linear(output_feats)   # [N, nClass], float
        semantic_preds = semantic_scores.max(1)[1]  # [N], long

        ret["semantic_scores"] = semantic_scores

        #### offset
        pt_offsets_feats = self.offset(output_feats)
        pt_offsets = self.offset_linear(pt_offsets_feats) # [N, 3], float32
        ret["pt_offsets"] = pt_offsets

        # if True:
        if (epoch > self.prepare_epochs) and not semantic_only:
            #### get prooposal clusters
            semantic_preds_filter = semantic_preds
            object_idx_filter = torch.nonzero(semantic_preds_filter > 1).view(-1)

            if len(object_idx_filter) > 0:
                batch_idxs_ = batch_idxs[object_idx_filter]
                batch_offsets_ = self.get_batch_offsets(batch_idxs_, input.batch_size)
                coords_ = coords[object_idx_filter]
                pt_offsets_ = pt_offsets[object_idx_filter]
                shifted_coords = coords_ + pt_offsets_
                semantic_preds_cpu = semantic_preds_filter[object_idx_filter].int().cpu()

                #### shift coordinates region grow
                idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(shifted_coords, batch_idxs_, batch_offsets_, self.cluster_radius_shift, int(self.cluster_shift_mean_active/2))
                proposals_idx_shift, proposals_offset_shift = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx_shift.cpu(), start_len_shift.cpu(), self.cluster_npoint_thresh)
                proposals_idx_shift[:, 1] = object_idx_filter[proposals_idx_shift[:, 1].long()].int()
                # proposals_idx_shift: [sum_points, 2], int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset_shift: [num_prop + 1], int

                proposals_idx = proposals_idx_shift
                proposals_offset = proposals_offset_shift

                #### origin coordinates region grow
                idx, start_len = pointgroup_ops.ballquery_batch_p(coords_, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_mean_active)
                proposals_idx_origin, proposals_offset_origin = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx.cpu(), start_len.cpu(), self.cluster_npoint_thresh)
                proposals_idx_origin[:, 1] = object_idx_filter[proposals_idx_origin[:, 1].long()].int()
                # proposals_idx_origin, [sum_points, 2], int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset_origin, [num_prop + 1], int

                proposals_idx_origin[:, 0] += (proposals_offset.size(0) - 1)
                proposals_offset_origin += proposals_offset[-1]
                proposals_idx = torch.cat((proposals_idx, proposals_idx_origin), dim=0)
                proposals_offset = torch.cat((proposals_offset, proposals_offset_origin[1:]))
                # proposals_idx, [sum_points, 2]: int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset, [num_prop + 1]: int

            #### proposals voxelization again
            input_feats, inp_map = self.clusters_voxelization(proposals_idx, output_feats, coords, self.score_fullscale, self.score_scale, self.score_mode)

            #### score
            score = self.score_unet(input_feats)
            score = self.score_outputlayer(score)
            score_feats = score.features[inp_map.long()]  # [sum_points, C]
            score_feats = scatter_max(score_feats, proposals_idx[:, 0].cuda().long(), dim=0)[0] # [num_prop, C]
            scores = self.score_linear(score_feats) # [num_prop, 1]

            ret["proposal_scores"] = (scores, proposals_idx, proposals_offset)


        return ret

    @staticmethod
    def get_batch_offsets(batch_idxs, bs):
        """
        :param batch_idxs: (N), int
        :param bs: int
        :return: batch_offsets: (bs + 1)
        """
        batch_idxs_np = batch_idxs.cpu().numpy()
        batch_offsets = np.append(np.searchsorted(batch_idxs_np, range(bs)), len(batch_idxs_np))
        batch_offsets = torch.Tensor(batch_offsets).int().to(batch_idxs.device)
        return batch_offsets

