# -*- coding:utf-8 -*-
# author: Xinge

from typing import List
import gorilla
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

@gorilla.MODULES.register_module()
class CylinderFea(nn.Module):

    def __init__(self,
                 grid_size,
                 fea_dim=3,
                 out_pt_fea_dim=64,
                 max_pt_per_encode=64,
                 fea_compre=None):
        super(CylinderFea, self).__init__()

        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fea_dim),

            nn.Linear(fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, out_pt_fea_dim)
        )

        self.max_pt = max_pt_per_encode
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        kernel_size = 3
        self.local_pool_op = torch.nn.MaxPool2d(kernel_size, stride=1,
                                                padding=(kernel_size - 1) // 2,
                                                dilation=1)
        self.pool_dim = out_pt_fea_dim

        # point feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.fea_compre),
                nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

    def forward(self,
                pt_fea: List[torch.Tensor],
                xy_ind: List[torch.Tensor]):

        # concat indices with batch index on 0-dim
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):
            cat_pt_ind.append(F.pad(xy_ind[i_batch], (1, 0), "constant", value=i_batch))

        cat_pt_fea = torch.cat(pt_fea, dim=0) # [N, 8]
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0) # [N, 4]

        # NOTE: shuffle process maybe slow and seems useless
        # get the device
        cur_dev = pt_fea[0].get_device()
        # shuffle the data
        pt_num = cat_pt_ind.shape[0] # [N]
        shuffled_ind = torch.randperm(pt_num, device=cur_dev)
        cat_pt_fea = cat_pt_fea[shuffled_ind, :] # [N, 8]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :] # [N, 4]

        # unique xy grid index
        # unq - [num_valid_voxel, 4]
        # unq_inv - [N], the range is [0, num_valid_voxel)
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0) # [N]
        unq = unq.type(torch.int64)

        # process feature
        processed_cat_pt_fea = self.PPmodel(cat_pt_fea) # [N, C] process point-wise features(PointNet)
        pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0] # [num_valid_voxel, C'] dylindrical features

        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)
        else:
            processed_pooled_data = pooled_data

        return unq, processed_pooled_data
