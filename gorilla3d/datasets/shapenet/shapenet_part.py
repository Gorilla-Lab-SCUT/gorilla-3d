# Copyright (c) Gorilla-Lab. All rights reserved.

import os
import os.path
import json
import numpy as np
import sys
from typing import List
from torch.utils.data import (Dataset, DataLoader)


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class ShapeNetPartNormal(Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 split='trainval',
                 normalize=True,
                 return_cls_label=True,
                 use_normal=True):
        """Author: wu.chaozheng
        dataloader of shapenet part.

        Args:
            root ([str]): [path to shapenet part dataset.]
            npoints (int, optional): [number points each object.]. Defaults to 2500.
            classification (bool, optional): [whether load the data for classification task.]. Defaults to False.
            split (str, optional): ['train', 'test', 'val' or 'trainval']. Defaults to 'trainval'.
            normalize (bool, optional): [whether to normalize the pc into unit sphere.]. Defaults to True.
            return_cls_label (bool, optional): [whether return the category label]. Defaults to True.
            use_normal (bool, optional): [whether use the normal]. Defaults to True.
        """
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}

        self.classification = classification
        self.normalize = normalize
        self.return_cls_label = return_cls_label
        self.use_normal = use_normal

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}

        self.meta = {}
        with open(
                os.path.join(self.root, 'train_test_split',
                             'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(
                os.path.join(self.root, 'train_test_split',
                             'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(
                os.path.join(self.root, 'train_test_split',
                             'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [
                    fn for fn in fns
                    if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))
                ]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {
            'Earphone': [16, 17, 18],
            'Motorbike': [30, 31, 32, 33, 34, 35],
            'Rocket': [41, 42, 43],
            'Car': [8, 9, 10, 11],
            'Laptop': [28, 29],
            'Cap': [6, 7],
            'Skateboard': [44, 45, 46],
            'Mug': [36, 37],
            'Guitar': [19, 20, 21],
            'Bag': [4, 5],
            'Lamp': [24, 25, 26, 27],
            'Table': [47, 48, 49],
            'Airplane': [0, 1, 2, 3],
            'Pistol': [38, 39, 40],
            'Chair': [12, 13, 14, 15],
            'Knife': [22, 23]
        }

        for cat in sorted(self.seg_classes.keys()):
            print(cat, self.seg_classes[cat])

    def __getitem__(self, index):
        """[summary]

        Args:
            index ([int]): [item index.]

        Returns:
            [numpy array]: [return data (point_set, normal, seg (opt), cls(opt))]
        """
        fn = self.datapath[index]
        cat = self.datapath[index][0]
        cls = self.classes[cat]
        cls = np.array([cls]).astype(np.int32)
        data = np.loadtxt(fn[1]).astype(np.float32)
        point_set = data[:, 0:3]
        if self.normalize:
            point_set = pc_normalize(point_set)
            data[:, :3] = point_set
        normal = data[:, 3:6]
        seg = data[:, 6].astype(np.int32)

        if len(seg) > self.npoints:
            choice = np.random.choice(len(seg), self.npoints, replace=False)
        else:
            choice = np.random.choice(len(seg), self.npoints, replace=True)

        #resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        normal = normal[choice, :]

        if self.use_normal:
            point_set = np.concatenate([point_set, normal], -1)

        data = {"point_set": point_set}
        # print(data)
        if self.classification:
            data.update({'cls_label': cls})
            return data
        else:
            if self.return_cls_label:
                data.update({'seg_label': seg, 'cls_label': cls})
                return data
            else:
                data.update({'seg_label': seg})
                return data

    def __len__(self):
        return len(self.datapath)
