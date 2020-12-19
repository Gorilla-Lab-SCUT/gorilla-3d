# modified from https://github.com/AnTao97/dgcnn.pytorch

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from ...modules.dgcnn.util import *
from ...modules.dgcnn.transformer import *



class DGCNNPartSeg(nn.Module):
    def __init__(self, cfg, seg_num_all):
        """Author: shi.xian
        dgcnn part segmentation network

        Args:
            k (int): [Num of nearest neighbors]
            emb_dims (int): [Dimension of embeddings]
            dropout (float): [dropout rate]
        """
        super().__init__()
        self.cfg = cfg
        self.k = cfg.get("k")
        self.emb_dims = cfg.get("emb_dims", 1024)
        self.dropout = cfg.get("dropout", 0.5)
        self.seg_num_all = seg_num_all

        self.with_trans = cfg.get("with_trans", True)

        if self.with_trans:
            self.transform_net = TransformNet(cfg)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(self.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
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
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(self.emb_dims + 256, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=self.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=self.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    def forward(self, x, l):
        """Author: shi.xian
        dgcnn cls forward

        Args:
            x (torch.Tensor, [batch_size, input_channels, num_points]): input points
            l (torch.Tensor, [batch_size, num_categories]): input shape category mask
        Returns:
            dict('input_pc', 'prediction')
            input_pc: [batch_size, input_channels, num_points]
            prediction: [batch_size, output_channels]: [the classification probability of each category]
        """

        results = dict(input_pc=x)

        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)
        
        if self.with_trans: 
            t = self.transform_net(x0)            
            x = x.transpose(2, 1)                 
            x = torch.bmm(x, t)                    
            x = x.transpose(2, 1)               

        x = get_graph_feature(x, k=self.k)     
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

        l = l.view(batch_size, -1, 1)         
        l = self.conv7(l)                      

        x = torch.cat((x, l), dim=1)        
        x = x.repeat(1, 1, num_points)        

        x = torch.cat((x, x1, x2, x3), dim=1)   

        x = self.conv8(x)                     
        x = self.dp1(x)
        x = self.conv9(x)                 
        x = self.dp2(x)
        x = self.conv10(x)                   
        x = self.conv11(x)                   
        
        results["prediction"] = x
        return results




