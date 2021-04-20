"""
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
"""
import os
import pickle

import gorilla
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

ID2CLASS = {0: "bed",
            1: "tv_stand",
            2: "xbox",
            3: "person",
            4: "night_stand",
            5: "curtain",
            6: "bottle",
            7: "bench",
            8: "mantel",
            9: "plant",
            10: "flower_pot",
            11: "tent",
            12: "stairs",
            13: "radio",
            14: "monitor",
            15: "guitar",
            16: "bathtub",
            17: "door",
            18: "piano",
            19: "cone",
            20: "keyboard",
            21: "bowl",
            22: "airplane",
            23: "dresser",
            24: "cup",
            25: "vase",
            26: "sofa",
            27: "range_hood",
            28: "glass_box",
            29: "car",
            30: "bookshelf",
            31: "lamp",
            32: "stool",
            33: "desk",
            34: "sink",
            35: "chair",
            36: "toilet",
            37: "table",
            38: "laptop",
            39: "wardrobe"}


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetCls(Dataset):
    def __init__(self,
                 root: str,
                 num_point: int,
                 num_category: int,
                 use_uniform_sample: bool,
                 use_normals: bool=False,
                 split: int="train",
                 process_data: bool=False):
        self.root = root
        self.npoints = num_point
        self.process_data = process_data
        self.uniform = use_uniform_sample
        self.use_normals = use_normals
        self.num_category = num_category

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, "modelnet10_shape_names.txt")
        else:
            self.catfile = os.path.join(self.root, "modelnet40_shape_names.txt")

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids["train"] = [line.rstrip() for line in open(os.path.join(self.root, "modelnet10_train.txt"))]
            shape_ids["test"] = [line.rstrip() for line in open(os.path.join(self.root, "modelnet10_test.txt"))]
        else:
            shape_ids["train"] = [line.rstrip() for line in open(os.path.join(self.root, "modelnet40_train.txt"))]
            shape_ids["test"] = [line.rstrip() for line in open(os.path.join(self.root, "modelnet40_test.txt"))]

        assert (split == "train" or split == "test")
        shape_names = ["_".join(x.split("_")[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + ".txt") for i
                         in range(len(shape_ids[split]))]
        print("The size of %s data is %d" % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root, "modelnet%d_%s_%dpts_fps.dat" % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, "modelnet%d_%s_%dpts.dat" % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print("Processing data %s (only running in the first time)..." % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=",").astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, "wb") as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print("Load processed data from %s..." % self.save_path)
                with open(self.save_path, "rb") as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=",").astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return {"point_set": point_set,
                "label": label[0]}


if __name__ == "__main__":
    import torch

    data = ModelNetCls("/data/modelnet40_normal_resampled/", split="train")
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)

