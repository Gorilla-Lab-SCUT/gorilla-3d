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
        self.ce_criterion = SoftmaxHeteroscedasticLoss()
        self.lovasz_criterion = gorilla.losses.lovasz_loss

        self.num_class = num_class
        self.ignore_label = ignore_label
        self.ce_weight, self.lovasz_weight = loss_weight

    def forward(self, loss_input):
        prediction = loss_input["prediction"] # [B, num_class, H, W, L]
        softmax_prediction = F.softmax(prediction, dim=1) # [B, num_class, H, W, L]
        labels = loss_input["labels"] # [B, H, W, L]

        # TODO: lovasz_criterion need to fix
        loss = self.ce_weight * self.ce_criterion(prediction, labels) + \
               self.lovasz_weight * self.lovasz_criterion(softmax_prediction, labels)

        return loss


# modify from https://github.com/Halmstad-University/SalsaNext/blob/master/train/tasks/semantic/modules/trainer.py
class SoftmaxHeteroscedasticLoss(torch.nn.Module):
    def __init__(self):
        super(SoftmaxHeteroscedasticLoss, self).__init__()
        keep_variance_fn = lambda x: x + 1e-3
        self.adf_softmax = Softmax(dim=1, keep_variance_fn=keep_variance_fn)

    def forward(self, outputs, targets, eps=1e-5):
        mean, var = self.adf_softmax(*outputs)
        targets = torch.nn.functional.one_hot(targets, num_classes=20).permute(0,3,1,2).float()

        precision = 1 / (var + eps)
        return torch.mean(0.5 * precision * (targets - mean) ** 2 + 0.5 * torch.log(var + eps))

class Softmax(nn.Module):
    def __init__(self, dim=1, keep_variance_fn=None):
        super(Softmax, self).__init__()
        self.dim = dim
        self._keep_variance_fn = keep_variance_fn

    def forward(self, features_mean, features_variance, eps=1e-5):
        """Softmax function applied to a multivariate Gaussian distribution.
        It works under the assumption that features_mean and features_variance
        are the parameters of a the indepent gaussians that contribute to the
        multivariate gaussian.
        Mean and variance of the log-normal distribution are computed following
        https://en.wikipedia.org/wiki/Log-normal_distribution."""

        log_gaussian_mean = features_mean + 0.5 * features_variance
        log_gaussian_variance = 2 * log_gaussian_mean

        log_gaussian_mean = torch.exp(log_gaussian_mean)
        log_gaussian_variance = torch.exp(log_gaussian_variance)
        log_gaussian_variance = log_gaussian_variance * (torch.exp(features_variance) - 1)

        constant = torch.sum(log_gaussian_mean, dim=self.dim) + eps
        constant = constant.unsqueeze(self.dim)
        outputs_mean = log_gaussian_mean / constant
        outputs_variance = log_gaussian_variance / (constant ** 2)

        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance