# Copyright (c) Gorilla-Lab. All rights reserved.
from typing import Dict, Optional

import gorilla
import torch
import torch.nn as nn
import torch.nn.functional as F

@gorilla.MODULES.register_module()
class RangeHead(nn.Module):
    def __init__(self,
                 dropout: float=0.01,
                 in_channels: int=64,
                 nclasses: int=20):
        super().__init__()
        self.head = nn.Sequential(nn.Dropout2d(p=dropout),
                                  nn.Conv2d(in_channels,
                                            nclasses,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1))
    def forward(self, x):
        return self.head(x)

@gorilla.MODELS.register_module()
class RangeNet(nn.Module):
    def __init__(self,
                 backbone_cfg: Dict,
                 decoder_cfg: Dict,
                 head_cfg: Dict,
                 crf_cfg: Optional[Dict]=None,
                 **kwargs):
        super().__init__()

        self.backbone = gorilla.build_module(backbone_cfg)
        self.decoder = gorilla.build_module(decoder_cfg)
        
        self.head = gorilla.build_module(head_cfg)
        
        self.CRF = None
        if crf_cfg is not None:
            self.CRF = gorilla.build_module(crf_cfg)

    def forward(self, x, mask=None):
        y, skips = self.backbone(x)
        y = self.decoder(y, skips)
        y = self.head(y)
        y = F.softmax(y, dim=1)
        if self.CRF:
            assert(mask is not None)
            y = self.CRF(x, y, mask)

        return y

