# modified from https://github.com/AnTao97/dgcnn.pytorch

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from ...modules.dgcnn.util import *



class DGCNNSemSeg(nn.Module):
    def __init__(self, cfg):
        """Author: shi.xian
        dgcnn semantic segmentation network

        Args:
            k (int): [Num of nearest neighbors]
            emb_dims (int): [Dimension of embeddings]
            dropout (float): [dropout rate]
        """
        super().__init__()
        self.cfg = cfg
        self.k = cfg.k
        self.emb_dims = cfg.get("emb_dims")

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(self.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=cfg.dropout)
        self.conv9 = nn.Conv1d(256, 13, kernel_size=1, bias=False)
        

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
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=True)  
        x = self.conv1(x)                   
        x = self.conv2(x)                     
        x1 = x.max(dim=-1, keepdim=False)[0]   

        x = get_graph_feature(x1, k=self.k)    
        x = self.conv3(x)                      
        x = self.conv4(x)                       
        x2 = x.max(dim=-1, keepdim=False)[0]   

        x = get_graph_feature(x2, k=self.k)     
        x = self.conv5(x)                      
        x3 = x.max(dim=-1, keepdim=False)[0]  

        x = torch.cat((x1, x2, x3), dim=1)   

        x = self.conv6(x)                      
        x = x.max(dim=-1, keepdim=True)[0]   

        x = x.repeat(1, 1, num_points)       
        x = torch.cat((x, x1, x2, x3), dim=1)   

        x = self.conv7(x)                      
        x = self.conv8(x)                   
        x = self.dp1(x)
        x = self.conv9(x)                    
        
        results["prediction"] = x
        return results