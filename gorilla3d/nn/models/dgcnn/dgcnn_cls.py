# modified from https://github.com/AnTao97/dgcnn.pytorch

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from ...modules.dgcnn.util import *


class DGCNNCls(nn.Module):
    def __init__(self, cfg, output_channels=40):
        """Author: shi.xian
        dgcnn classification network

        Args:
            k (int): [Num of nearest neighbors]
            emb_dims (int): [Dimension of embeddings]
            dropout (List[int], optional): [layers to apply dropout]
        """
        super(DGCNNCls, self).__init__()
        self.cfg = cfg
        self.k = cfg.get("k")
        self.emb_dims = cfg.get("emb_dims")
        self.dropout = cfg.get("dropout", 0.5)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(self.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        """Author: shi.xian
        dgcnn cls forward

        Args:
            x (torch.Tensor, [batch_size, input_channels, num_points]): input points

        Returns:
            dict('input_pc', 'prediction')
            input_pc: [batch_size, input_channels, num_points]
            prediction: [batch_size, output_channels]: [the classification probability of each category]
        """
        results = dict(input_pc=x)

        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)      
        x = self.conv1(x)                      
        x1 = x.max(dim=-1, keepdim=False)[0]    

        x = get_graph_feature(x1, k=self.k)     
        x = self.conv2(x)                      
        x2 = x.max(dim=-1, keepdim=False)[0]    

        x = get_graph_feature(x2, k=self.k)    
        x = self.conv3(x)                   
        x3 = x.max(dim=-1, keepdim=False)[0]    

        x = get_graph_feature(x3, k=self.k)  
        x = self.conv4(x)                       
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        
        results["prediction"] = x
        return results



