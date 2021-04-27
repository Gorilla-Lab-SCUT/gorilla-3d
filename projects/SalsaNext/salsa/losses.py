# Copyright (c) Gorilla-Lab. All rights reserved.
from typing import List

try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import gorilla

@gorilla.LOSSES.register_module()
class SalsaLoss(nn.Module):
    def __init__(self,
                 num_class: int=20,
                 ignore_label: int=-1,
                 loss_weight: List[float]=[1.0, 1.0],
                 **kwargs):
        super().__init__()
        self.ce_criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)
        self.lovasz_criterion = gorilla.losses.lovasz_loss

        self.num_class = num_class
        self.ignore_label = ignore_label
        self.ce_weight, self.lovasz_weight = loss_weight

        #### criterion
        self.semantic_criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_label)
        self.score_criterion = nn.BCELoss(reduction="none")

    def forward(self, loss_input):
        prediction = loss_input["prediction"] # [B, num_class, H, W, L]
        softmax_prediction = F.softmax(prediction, dim=1) # [B, num_class, H, W, L]
        labels = loss_input["labels"] # [B, H, W, L]

        # TODO: lovasz_criterion need to fix
        loss = self.ce_weight * self.ce_criterion(prediction, labels) + \
               self.lovasz_weight * self.lovasz_criterion(softmax_prediction, labels)

        return loss


