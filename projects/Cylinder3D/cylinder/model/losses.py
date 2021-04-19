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
class CylinderLoss(nn.Module):
    def __init__(self,
                 num_class: int=20,
                 ignore_label: int=-100,
                 loss_weight: List[float]=[1.0, 1.0],
                 **kwargs):
        super().__init__()
        self.ce_criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)
        self.lovasz_criterion = lovasz_softmax

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

### lovasz_softmax
# TODO: merging into gorilla-core
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(lambda x: x!=x, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W, L] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W, L] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    # calculate loss for each class
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes == "present" and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None, sigmoid=False):
    """
    Flattens predictions in the batch more generally
    """
    if sigmoid:
        # assumes output of a sigmoid layer
        probas = probas.unsqueeze(1) # [B, 1, ...]
    C = probas.shape[1]
    ndim = len(probas.shape)
    probas = probas.permute(0, *range(2, ndim), 1).contiguous().view(-1, C) # [B*..., C]
    labels = labels.view(-1) # [B*...]
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()] # [num_valid, C]
    vlabels = labels[valid] # [num_valid, C]
    return vprobas, vlabels


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

