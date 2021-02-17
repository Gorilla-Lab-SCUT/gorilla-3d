import torch, glob, os, numpy as np
import os.path as osp
import sys
sys.path.append("../")

import gorilla


def get_checkpoint(log_dir, epoch=0, checkpoint=""):
    if not checkpoint:
        if epoch > 0:
            checkpoint = osp.join(log_dir, "epoch_{0:05d}.pth".format(epoch))
            assert osp.isfile(checkpoint)
        else:
            checkpoint = sorted(glob.glob(osp.join(log_dir, "*.pth")))
            if len(checkpoint) > 0:
                checkpoint = checkpoint[-1]
                epoch = int(checkpoint.split("_")[-1].split(".")[0])

    return checkpoint, epoch + 1


def write_obj(points, colors, out_filename):
    N = points.shape[0]
    fout = open(out_filename, "w")
    for i in range(N):
        c = colors[i]
        fout.write("v %f %f %f %d %d %d\n" % (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]))
    fout.close()


def get_batch_offsets(batch_idxs, bs):
    """
    :param batch_idxs: (N), int
    :param bs: int
    :return: batch_offsets: (bs + 1)
    """
    batch_idxs_np = batch_idxs.cpu().numpy()
    batch_offsets = np.append(np.searchsorted(batch_idxs_np, range(bs)), len(batch_idxs_np))
    batch_offsets = torch.Tensor(batch_offsets).int().to(batch_idxs.device)
    return batch_offsets

