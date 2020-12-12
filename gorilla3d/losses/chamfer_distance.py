import torch
from torch import nn as nn
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss

from ..ops import cham_dist


def chamfer_distance(src,
                     dst,
                     src_weight=1.0,
                     dst_weight=1.0,
                     criterion_mode="l2",
                     reduction="mean"):
    r"""Calculate Chamfer Distance of two sets.
    Args:
        src (torch.Tensor): Source set with shape [B, N, C] to
            calculate Chamfer Distance.
        dst (torch.Tensor): Destination set with shape [B, M, C] to
            calculate Chamfer Distance.
        src_weight (torch.Tensor or float): Weight of source loss.
        dst_weight (torch.Tensor or float): Weight of destination loss.
        criterion_mode (str): Criterion mode to calculate distance.
            The valid modes are smooth_l1, l1 or l2.
        reduction (str): Method to reduce losses.
            The valid reduction method are 'none', 'sum' or 'mean'.
    Returns:
        tuple: Source and Destination loss with the corresponding indices.
            - loss_src (torch.Tensor): The min distance \
                from source to destination.
            - loss_dst (torch.Tensor): The min distance \
                from destination to source.
            - idx_src (torch.Tensor): Index the min distance point \
                for each point in source to destination.
            - idx_dst (torch.Tensor): Index the min distance point \
                for each point in destination to source.
    """

    dist_src, dist_dst, idx_src, idx_dst = cham_dist(src, dst)

    if criterion_mode == "l2":
        dist_src = torch.sqrt(dist_src)
        dist_dst = torch.sqrt(dist_dst)
    elif criterion_mode == "smooth_l1":
        dist_src = smooth_l1_loss(src, dst[idx_src])
        dist_dst = smooth_l1_loss(dst, src[idx_dst])
    elif criterion_mode == "l1":
        dist_src = l1_loss(src, dst[idx_src])
        dist_dst = l1_loss(dst, src[idx_dst])
    else:
        raise NotImplementedError

    loss_src = (dist_src * src_weight)  # [B, N]
    loss_dst = (dist_dst * dst_weight)  # [B, M]

    if reduction == "sum":
        loss_src = torch.sum(loss_src)
        loss_dst = torch.sum(loss_dst)
    elif reduction == "mean":
        loss_src = torch.mean(loss_src)
        loss_dst = torch.mean(loss_dst)
    elif reduction == "none":
        pass
    else:
        raise NotImplementedError

    return loss_src, loss_dst, idx_src, idx_dst
