"""
PointGroup
Written by Li Jiang
"""
import sys
import functools

import spconv
import gorilla3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import cKDTree
from torch_scatter import scatter_min, scatter_mean, scatter_max

from ..lib.pointgroup_ops.functions import pointgroup_ops
from ..util import get_batch_offsets
from .func_helper import *


class PointGroup(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        input_c = cfg.model.input_channel
        blocks = cfg.model.blocks
        m = cfg.model.m
        classes = cfg.model.classes
        block_reps = cfg.model.block_reps

        self.cluster_radius = cfg.cluster.cluster_radius
        self.cluster_radius_shift = cfg.cluster.cluster_radius_shift
        self.cluster_meanActive = cfg.cluster.cluster_meanActive
        self.cluster_shift_meanActive = cfg.cluster.cluster_shift_meanActive
        self.cluster_npoint_thre = cfg.cluster.cluster_npoint_thre

        self.score_scale = cfg.model.score_scale
        self.score_fullscale = cfg.model.score_fullscale
        self.mode = cfg.model.score_mode

        self.prepare_epochs = cfg.model.prepare_epochs

        self.fix_module = cfg.model.fix_module

        try:
            self.aggregate_feat = cfg.aggregate_feat
        except:
            self.aggregate_feat = False
        
        try:
            self.overseg_pooling_type = cfg.overseg_pooling_type
        except:
            self.overseg_pooling_type = "max"


        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        block = gorilla3d.ResidualBlock

        if cfg.model.use_coords:
            input_c += 3

        #### backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key="subm1")
        )

        block_list = [m * (i + 1) for i in range(blocks)]
        self.unet = gorilla3d.UBlockBottom(block_list, norm_fn, block_reps, block, indice_key_id=1)

        #### self attention module
        # self.self_attn = nn.MultiheadAttention(m * blocks, 8)

        self.output_layer = spconv.SparseSequential(
            norm_fn(m),
            nn.ReLU()
        )

        #### semantic segmentation
        self.linear = nn.Linear(m, classes) # bias(default): True

        #### offset
        self.offset = nn.Sequential(
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU()
        )
        self.offset_linear = nn.Linear(m, 3, bias=True)

        #### overseg partial
        self.overseg_linear = nn.Sequential(
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU()
        )
        self.overseg_semantic = nn.Linear(m, classes)  # bias(default): True
        self.overseg_offsets = nn.Linear(m, 3)  # bias(default): True

        #### score branch
        self.score_unet = gorilla3d.UBlock([m, 2*m], norm_fn, 2, block, indice_key_id=1)
        self.score_outputlayer = spconv.SparseSequential(
            norm_fn(m),
            nn.ReLU()
        )
        self.score_linear = nn.Linear(m, 1)

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

    def aggregate_features(self, clusters_idx, feats, mode=0):
        """
        :param clusters_idx: (SumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
        :param feats: (N, C), float, cuda
        :mode: int (0:max-pooling 1:mean-pooling)
        :return:
        :cluster_feats: (nCluster, C)
        """
        c_idxs = clusters_idx[:, 1].cuda() # (sumNPoint)
        clusters_feats = feats[c_idxs.long()] # (sumNPoint, C)
        if mode == 0:  # max-pooling
            clusters_feats = scatter_max(clusters_feats, clusters_idx[:, 0].cuda().long(), dim=0)[0] # (nCluster, C)
        elif mode == 1: # mean-pooling
            clusters_feats = scatter_mean(clusters_feats, clusters_idx[:, 0].cuda().long(), dim=0) # (nCluster, C)
        else:
            raise ValueError("mode must be '0' or '1', but got {}".format(mode))
        return clusters_feats


    def build_overseg_graph_overseg(self, overseg_centers, overseg_semantic, overseg_offsets, overseg_batch_idxs, num_nearest=8, thr_outside=0.3):
        """
        :overseg_centers: (num_overseg, 3), float, cuda
        :overseg_semantic: (num_overseg), long, cuda
        :overseg_offsets: (num_overseg, 3), float, cuda
        :overseg_batch_idxs: (num_overseg), long, cuda
        :return:
        :adajency_matrix_list
        """
        overseg_shifts = overseg_centers + overseg_offsets  # (num_overseg, 3)

        # build graph according to centers
        adajency_matrix_list = []
        semantic_connect_map_list = []
        for batch_idx in range(overseg_batch_idxs.max() + 1):
            # slice according to batch
            ids = (overseg_batch_idxs == batch_idx)
            num_batch = ids.sum()
            batch_overseg_centers = overseg_centers[ids] # (num_batch, 3)
            batch_overseg_offsets = overseg_offsets[ids] # (num_batch, 3)
            batch_overseg_shifts = overseg_shifts[ids] # (num_batch, 3)
            batch_overseg_semantic = overseg_semantic[ids] # (num_batch)

            # build the semantic connect map
            semantic_connect_map = torch.eye(num_batch).bool().to(batch_overseg_semantic.device) # (num_batch, num_batch)
            # semantic_connect_map = batch_overseg_semantic.new_zeros([num_batch, num_batch]).bool() # (num_batch, num_batch)
            for semantic in torch.unique(batch_overseg_semantic):
                if semantic < 2:
                    continue
                semantic_ids = torch.where(batch_overseg_semantic == semantic)[0]
                x_ids, y_ids = torch.meshgrid(semantic_ids, semantic_ids)
                semantic_connect_map[x_ids.reshape(-1), y_ids.reshape(-1)] = True
            semantic_connect_map_list.append(semantic_connect_map)

            # calculate distance matrix and get distance matrix
            distance_matrix = torch.sqrt(gorilla3d.square_distance(batch_overseg_centers)) # (num_batch, num_batch)
            distance_matrix_shift = torch.sqrt(gorilla3d.square_distance(batch_overseg_shifts))  # (num_batch, num_batch)
            distance_squeeze_map = (distance_matrix_shift < distance_matrix) # (num_batch, num_batch)
            
            batch_overseg_semantic = overseg_semantic[ids]  # (num_batch)
            
            # adajency_matrix = batch_overseg_centers.new_zeros([num_batch, num_batch]).bool() # (num_batch, num_batch)
            # tree = cKDTree(batch_overseg_centers.detach().cpu().numpy())
            # dd, ii = tree.query(batch_overseg_centers.detach().cpu().numpy(), num_nearest)
            
            adajency_matrix = batch_overseg_shifts.new_zeros([num_batch, num_batch]).bool() # (num_batch, num_batch)
            tree = cKDTree(batch_overseg_shifts.detach().cpu().numpy())
            dd, ii = tree.query(batch_overseg_shifts.detach().cpu().numpy(), num_nearest)

            nearest_map = torch.Tensor(ii).long().to(adajency_matrix.device)  # (num_batch, num_nearest)
            filter_ids = torch.Tensor(dd < thr_outside).bool().to(adajency_matrix.device) # (num_batch, num_nearest)
            x = torch.arange(num_batch).to(adajency_matrix.device) # (num_batch)
            y = torch.arange(num_nearest).to(adajency_matrix.device) # (num_nearest)
            query_map, _ = torch.meshgrid(x, y)  # (num_batch, num_nearest)
            ids_x = query_map.reshape(-1)[filter_ids.view(-1)]
            ids_y = nearest_map.view(-1)[filter_ids.view(-1)]
            adajency_matrix[ids_x, ids_y] = True
            adajency_matrix[ids_y, ids_x] = True
            
            adajency_matrix *= semantic_connect_map
            adajency_matrix *= distance_squeeze_map
            # # outside(far centers filter)
            # outside_map = torch.Tensor(dd > thr_outside).long().to(adajency_matrix.device) # (num_batch, num_nearest)
            # adajency_matrix[query_map.reshape(-1), nearest_map.view(-1)] = True
            # adajency_matrix[nearest_map.view(-1), query_map.reshape(-1)] = True

            adajency_matrix = adajency_matrix.float()
            adajency_matrix_list.append(adajency_matrix)

        return adajency_matrix_list, semantic_connect_map_list


    def clusters_voxelization(self, clusters_idx, feats, coords, fullscale, scale, mode):
        """
        :param clusters_idx: (SumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
        :param feats: (N, C), float, cuda
        :param coords: (N, 3), float, cuda
        :return:
        """
        c_idxs = clusters_idx[:, 1].cuda()
        clusters_feats = feats[c_idxs.long()]
        clusters_coords = coords[c_idxs.long()]

        clusters_coords_mean = scatter_mean(clusters_coords, clusters_idx[:, 0].cuda().long(), dim=0) # (nCluster, 3), float

        clusters_coords_mean = torch.index_select(clusters_coords_mean, 0, clusters_idx[:, 0].cuda().long())  # (sumNPoint, 3), float
        clusters_coords -= clusters_coords_mean

        clusters_coords_min = scatter_min(clusters_coords, clusters_idx[:, 0].cuda().long(), dim=0)[0] # (nCluster, 3), float
        clusters_coords_max = scatter_max(clusters_coords, clusters_idx[:, 0].cuda().long(), dim=0)[0] # (nCluster, 3), float

        clusters_scale = 1 / ((clusters_coords_max - clusters_coords_min) / fullscale).max(1)[0] - 0.01  # (nCluster), float
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        min_xyz = clusters_coords_min * clusters_scale.unsqueeze(-1)  # (nCluster, 3), float
        max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

        clusters_scale = torch.index_select(clusters_scale, 0, clusters_idx[:, 0].cuda().long())

        clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

        range = max_xyz - min_xyz
        # offset = - min_xyz + torch.clamp(fullscale - range - 0.001, min=0) * torch.rand(3).cuda() + torch.clamp(fullscale - range + 0.001, max=0) * torch.rand(3).cuda()
        offset = - min_xyz
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        clusters_coords += offset
        assert clusters_coords.shape.numel() == ((clusters_coords >= 0) * (clusters_coords < fullscale)).sum()

        clusters_coords = clusters_coords.long()
        clusters_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), clusters_coords.cpu()], 1)  # (sumNPoint, 1 + 3)

        out_coords, inp_map, out_map = pointgroup_ops.voxelization_idx(clusters_coords, int(clusters_idx[-1, 0]) + 1, mode)
        # output_coords: M * (1 + 3) long
        # input_map: sumNPoint int
        # output_map: M * (maxActive + 1) int

        out_feats = pointgroup_ops.voxelization(clusters_feats, out_map.cuda(), mode)  # (M, C), float, cuda

        spatial_shape = [fullscale] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats, out_coords.int().cuda(), spatial_shape, int(clusters_idx[-1, 0]) + 1)

        return voxelization_feats, inp_map


    def normalize_mask_input(self, clusters_idx, feats, coords, global_feats):
        """
        :param clusters_idx: (SumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
        :param feats: (N, C), float, cuda
        :param coords: (N, 3), float, cuda
        :param global_feats: (nCluster, C), float, cuda
        :return:
        """
        c_idxs = clusters_idx[:, 1].cuda()
        clusters_feats = feats[c_idxs.long()] # (sumNPoint, C), float
        clusters_coords = coords[c_idxs.long()] # (sumNPoint, 3), float

        clusters_coords_mean = scatter_mean(clusters_coords, clusters_idx[:, 0].cuda().long(), dim=0) # (nCluster, 3), float
        clusters_coords_mean = torch.index_select(clusters_coords_mean, 0, clusters_idx[:, 0].cuda().long()) # (sumNPoint, 3), float
        clusters_coords -= clusters_coords_mean # (sumNPoint, 3), float

        clusters_global_feats = torch.index_select(global_feats, 0, clusters_idx[:, 0].cuda().long())  # (sumNPoint, C), float
        
        combine_feats = torch.cat([clusters_feats, clusters_global_feats, clusters_coords], dim=1)  # (sumNPoint, C+C+3), float
        return combine_feats


    def forward(self, input, input_map, coords, batch_idxs, batch_offsets, coords_offsets, scene_list, epoch, extra_data=None, mode="train", semantic_only=False):
        """
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda
        :param coords_offsets: (B, 3), int, cuda
        """
        ret = {}
            
        for m in self.fix_module:
            # self.module_map[m].eval()
            mod = self.module_map[m]
            mod.eval()

        output = self.input_conv(input)
        output, bottom = self.unet(output)

        output = self.output_layer(output)
        output_feats = output.features[input_map.long()] # (N, m)

        #### semantic segmentation
        semantic_scores = self.linear(output_feats)   # (N, nClass), float
        semantic_preds = semantic_scores.max(1)[1]  # (N), long

        ret["semantic_scores"] = semantic_scores

        #### offset
        pt_offsets_feats = self.offset(output_feats)
        pt_offsets = self.offset_linear(pt_offsets_feats) # (N, 3), float32
        ret["pt_offsets"] = pt_offsets
        
        #### analysis extra_data
        overseg = extra_data["overseg"]

        overseg_centers = scatter_mean(coords, overseg, dim=0)  # (num_overseg, 3)
        overseg_batch_idxs = scatter_mean(batch_idxs, overseg, dim=0).long()  # (num_overseg)

        #### overseg semantic/offsets
        overseg_feats = overseg_pooling(output_feats, overseg, self.overseg_pooling_type)
        overseg_feats = self.overseg_linear(overseg_feats) # (num_overseg, C)
        overseg_semantic_scores = self.overseg_semantic(overseg_feats) # (num_overseg, nClass)
        overseg_semantic_preds = overseg_semantic_scores.max(1)[1]  # (num_overseg), long
        ret["overseg_semantic_scores"] = overseg_semantic_scores
        overseg_pt_offsets = self.overseg_offsets(overseg_feats) # (num_overseg, 3)
        ret["overseg_pt_offsets"] = overseg_pt_offsets
        
        # conver the point-wise to overseg-wise
        overseg_semantic_preds = align_overseg_semantic_label(semantic_preds, overseg, 21)  # (num_overseg)
        semantic_preds_by_overseg = overseg_semantic_preds[overseg]  # (N)
        semantic_equal_ids = (semantic_preds_by_overseg == semantic_preds)
        overseg_pt_offsets = scatter_mean(pt_offsets, overseg, dim=0)  # (num_overseg, 3)

        # if mode=="test":
        #     # map the oversegs' semantic label to points' semantic label
        #     semantic_preds = overseg_semantic_scores.max(1)[1] # (num_overseg)
        #     semantic_preds_filter = semantic_preds[overseg]
        #     semantic_scores = F.one_hot(semantic_preds_filter.long(), 20).float()
        #     object_idx_filter = torch.nonzero(semantic_preds_filter > 1).view(-1)
        #     ret["semantic_scores"] = semantic_scores

        # if True:
        if (epoch > self.prepare_epochs) and not semantic_only:
            #### get prooposal clusters
            # from copy import deepcopy
            # semantic_preds_filter = refine_semantic_segmentation(semantic_preds, overseg)
            # semantic_scores = F.one_hot(semantic_preds_filter.long(), 20).float()
            # ret["semantic_scores"] = semantic_scores
            semantic_preds_filter = semantic_preds
            object_idx_filter = torch.nonzero(semantic_preds_filter > 1).view(-1)

            # use overseg directly
            adajency_matrix_list, semantic_connect_map_list = self.build_overseg_graph_overseg(overseg_centers, overseg_semantic_preds, overseg_pt_offsets, overseg_batch_idxs)

            proposals_idx, proposals_offset = overseg_graph_fusion(overseg, batch_idxs, overseg_semantic_preds, overseg_batch_idxs, adajency_matrix_list)
            _, proposals_idx[:, 0] = torch.unique(proposals_idx[:, 0], return_inverse=True)

            # import os.path as osp
            # np.save(osp.join("visual", "data", scene_list[0] + "_coords.npy"), coords.cpu().numpy())
            # np.save(osp.join("visual", "data", scene_list[0] + "_colors.npy"), input.features[input_map.long(), :3].cpu().numpy())
            # np.save(osp.join("visual", "data", scene_list[0] + "_overseg.npy"), overseg.cpu().numpy())
            # np.save(osp.join("visual", "data", scene_list[0] + "_overseg_semantic.npy"), overseg_semantic_preds.cpu().numpy())
            # np.save(osp.join("visual", "data", scene_list[0] + "_overseg_centers.npy"), overseg_centers.cpu().numpy())
            # np.save(osp.join("visual", "data", scene_list[0] + "_overseg_offsets.npy"), overseg_pt_offsets.cpu().numpy())
            # # np.save(osp.join("visual", "data", scene_list[0] + "_proposal_idx.npy"), proposals_idx_origin.cpu().numpy())
            # np.save(osp.join("visual", "data", scene_list[0] + "_proposal_idx.npy"), proposals_idx.cpu().numpy())

            # # visual
            # visual_tree(coords, overseg, batch_offsets, overseg_centers, overseg_batch_idxs, adajency_matrix_list, scene_list)
            # visual_tree(coords, overseg, batch_offsets, overseg_centers, overseg_batch_idxs, [connect_map], scene_list, suffix="connect")

            if len(object_idx_filter) > 0:
                batch_idxs_ = batch_idxs[object_idx_filter]
                batch_offsets_ = get_batch_offsets(batch_idxs_, input.batch_size)
                coords_ = coords[object_idx_filter]
                pt_offsets_ = pt_offsets[object_idx_filter]
                shifted_coords = coords_ + pt_offsets_
                # semantic_preds_cpu = semantic_preds[object_idx_filter].int().cpu()
                semantic_preds_cpu = semantic_preds_filter[object_idx_filter].int().cpu()

                # idx, start_len = pointgroup_ops.ballquery_batch_p(coords_, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_meanActive)
                # proposals_idx_origin, proposals_offset_origin = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx.cpu(), start_len.cpu(), self.cluster_npoint_thre)
                # proposals_idx_origin[:, 1] = object_idx_filter[proposals_idx_origin[:, 1].long()].int()
                # proposals_idx_origin, proposals_offset_origin = overseg_fusion(proposals_idx_origin, proposals_offset_origin, overseg, thr_filter=0.4, thr_fusion=0.4)
                # proposals_idx_origin: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset_origin: (nProposal + 1), int

                # idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(shifted_coords, batch_idxs_, batch_offsets_, self.cluster_radius_shift, int(self.cluster_shift_meanActive/2))
                # proposals_idx_shift, proposals_offset_shift = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx_shift.cpu(), start_len_shift.cpu(), self.cluster_npoint_thre)
                # proposals_idx_shift[:, 1] = object_idx_filter[proposals_idx_shift[:, 1].long()].int()
                # # proposals_idx_shift, proposals_offset_shift = overseg_fusion(proposals_idx_shift, proposals_offset_shift, overseg, thr_filter=0.25, thr_fusion=0.25)
                # # proposals_idx_shift: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # # proposals_offset_shift: (nProposal + 1), int

                # proposals_idx = proposals_idx_shift
                # proposals_offset = proposals_offset_shift

                # proposals_idx_origin[:, 0] += (proposals_offset.size(0) - 1)
                # proposals_offset_origin += proposals_offset[-1]
                # proposals_idx = torch.cat((proposals_idx, proposals_idx_origin), dim=0)
                # proposals_offset = torch.cat((proposals_offset, proposals_offset_origin[1:]))
                # # proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # # proposals_offset: (nProposal + 1), int

            #### proposals voxelization again
            input_feats, inp_map = self.clusters_voxelization(proposals_idx, output_feats, coords, self.score_fullscale, self.score_scale, self.mode)

            #### score (without voxelization to save time)
            if self.aggregate_feat:
                score_media = self.score_unet(output)
                score_media = self.score_outputlayer(score_media)
                score_media = score_media.features[input_map.long()] # (N, C)
                # max-pooling according to cluster proposals idx(without voxelization again to save time)
                score_feats = self.aggregate_features(proposals_idx, score_media) # (nProposal, C)
                scores = self.score_linear(score_feats) # (nProposal, 1)
            else:
                #### score
                score = self.score_unet(input_feats)
                score = self.score_outputlayer(score)
                score_feats = score.features[inp_map.long()]  # (sumNPoint, C)
                score_feats = scatter_max(score_feats, proposals_idx[:, 0].cuda().long(), dim=0)[0] # (nProposal, C)
                scores = self.score_linear(score_feats) # (nProposal, 1)

            ret["proposal_scores"] = (scores, proposals_idx, proposals_offset)
                

        return ret


