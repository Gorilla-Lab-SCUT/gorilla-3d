# Copyright (c) Gorilla-Lab. All rights reserved.
import torch
import torch.nn as nn
import gorilla

@gorilla.LOSSES.register_module()
class ClsLoss(nn.Module):
    def __init__(self,
                 ignore_label: int=-100,
                 loss_weight: float=1.0,
                 **kwargs):
        super().__init__()
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight

        #### criterion
        self.semantic_criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_label)

    def forward(self, loss_inp):
        loss_out = {}

        """semantic loss"""
        logits = loss_inp["logits"]
        labels = loss_inp["labels"]
        # logits: (B, N, nClass), float32, cuda
        # labels: (B, N), long, cuda

        cls_loss = self.semantic_criterion(logits, labels)
        # loss_out["cls_loss"] = (cls_loss, semantic_scores.shape[0]) # the second item is number of point (Optional)
        loss_out["cls_loss"] = cls_loss


        """total loss"""
        # loss = mask_loss
        loss = self.loss_weight * cls_loss

        return loss, loss_out


