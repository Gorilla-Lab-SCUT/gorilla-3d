import torch, glob, os, numpy as np
import os.path as osp
import sys
sys.path.append("../")

import gorilla


def get_checkpoint(exp_path, exp_name, epoch=0, f=""):
    if not f:
        if epoch > 0:
            f = osp.join(exp_path, exp_name + "-%09d"%epoch + ".pth")
            assert osp.isfile(f)
        else:
            f = sorted(glob.glob(osp.join(exp_path, exp_name + "-*.pth")))
            if len(f) > 0:
                f = f[-1]
                epoch = int(f.split("-")[-1].split(".")[0])
                # epoch = int(f[len(exp_path) + len(exp_name) + 2 : -4])

    return f, epoch + 1


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
    batch_offsets = torch.Tensor(batch_offsets).to(batch_idxs.device)
    return batch_offsets

