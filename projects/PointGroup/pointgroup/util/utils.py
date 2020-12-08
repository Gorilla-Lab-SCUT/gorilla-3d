import torch, glob, os, numpy as np
import os.path as osp
import sys
sys.path.append("../")

import gorilla

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    # "K" classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))  # area_intersection: K, indicates the number of members in each class in intersection
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


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

