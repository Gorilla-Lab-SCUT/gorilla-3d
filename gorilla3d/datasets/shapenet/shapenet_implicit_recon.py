# Copyright (c) Gorilla-Lab. All rights reserved.
import os
from typing import List

import numpy as np
from torch.utils import data


class ShapenetCommonDataset(data.Dataset):
    def __init__(
        self,
        h5_dataset_folder: str,
        split: str = 'train',
        categories: List[str] = ['03001627'],
        ###
        requires_xyzgt: bool = False,
        requires_pc: bool = False,
        requires_randpts: bool = False,
        requires_rgb: bool = False,
        requires_ctrmag: bool = False,
        ####
        sdf_num: int = None,
        pc_num: int = None,
        randpts_num: int = None,
        rgb_rand_num: int = 1,
        rgb_idx_lst: List = None,
        shuffle: bool = False,
    ):
        super(ShapenetCommonDataset, self).__init__()
        assert split in ['train', 'test']
        self.h5_dataset_folder = h5_dataset_folder
        self.split = split
        self.categories = categories
        ###########
        self.requires_xyzgt = requires_xyzgt
        self.requires_pc = requires_pc
        self.requires_randpts = requires_randpts
        self.requires_rgb = requires_rgb
        self.requires_ctrmag = requires_ctrmag
        ###########
        self.sdf_num = sdf_num
        self.pc_num = pc_num
        self.randpts_num = randpts_num
        self.rgb_rand_num = rgb_rand_num
        self.rgb_idx_lst = rgb_idx_lst
        self.shuffle = shuffle

        assert self.categories is not None and len(self.categories) != 0
        self.h5_list = [f"{c}_{self.split}.h5" for c in self.categories]
        self.field_list = [ShapenetCommonField(os.path.join(self.h5_dataset_folder, s)) for s in self.h5_list]

        self.load_configs = dict(
            requires_xyzgt=self.requires_xyzgt,
            requires_pc=self.requires_pc,
            requires_randpts=self.requires_randpts,
            requires_rgb=self.requires_rgb,
            requires_ctrmag=self.requires_ctrmag,
            sdf_num=self.sdf_num,
            pc_num=self.pc_num,
            randpts_num=self.randpts_num,
            rgb_rand_num=self.rgb_rand_num,
            rgb_idx_lst=self.rgb_idx_lst,
            shuffle=self.shuffle,
        )

        self.eachlen = [field.length for field in self.field_list]
        self.cumsum = list(np.cumsum(np.array(self.eachlen)))
        self.length = sum(self.eachlen)

    def parse_index(self, idx):
        class_index = None
        st = 0
        for i in range(len(self.cumsum)):
            ed = self.cumsum[i]
            if idx >= st and idx < ed:
                class_index = i
                break
            st = ed
        obj_index = idx - (self.cumsum[class_index - 1] if class_index != 0 else 0)
        return class_index, obj_index

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        class_index, obj_index = self.parse_index(idx)

        data = self.field_list[class_index].load(obj_index=obj_index, **self.load_configs)

        data['class_name'] = self.categories[class_index]

        return data

#############################################################################################################################################

import os
import h5py
import numpy as np

###############################################


def get_data(f, key, obj_index, num=None):
    if num is None:
        return f[key][obj_index, ...]
    elif isinstance(num, list):
        return f[key][obj_index, np.array(num), ...]
    else:
        start_index = np.random.randint(f[key].shape[1] - num + 1)
        return f[key][obj_index, start_index:(start_index + num), ...]


def to_float32(arr):
    if arr.dtype == np.float16:
        return arr.astype(np.float32) + 1e-4 * np.random.randn(*arr.shape).astype(np.float32)
    elif arr.dtype == np.float32:
        return arr
    else:
        return arr.astype(np.float32)


###############################################


class ShapenetCommonField:
    def __init__(self, file_name):
        '''
        Args:
            file_name (str): the huge h5 file name
        '''
        self.file_name = file_name
        self.f = h5py.File(self.file_name, 'r')
        self.obj_list = list(self.f['obj_list'][:])
        self.obj_list = [s.decode('utf-8') for s in self.obj_list]

    def close(self):
        self.f.close()

    @property
    def length(self):
        return len(self.obj_list)

    def load(
        self,
        obj_index,
        requires_xyzgt=False,
        requires_pc=False,
        requires_randpts=False,
        requires_rgb=False,
        requires_ctrmag=False,
        sdf_num=None,
        pc_num=None,
        randpts_num=None,
        rgb_rand_num=1,
        rgb_idx_lst=None,
        shuffle=False,
    ):
        """
        Args:
            obj_index (int): obj index

            requires_xyzgt (bool): requires xyzgt
            requires_pc (bool): requires pc (poisson sampling)
            requires_randpts (bool): requires randpts (random sampling)
            requires_rgb (bool): requires rgb image(s)
            requires_ctrmag (bool): requires ctrmag

            sdf_num (int): num of sdf samples, if None, sdf_num==all
            pc_num (int): num of pc, if None, pc_num==all
            randpts_num (int): num of randpts, if None, randpts_num==all
            rgb_rand_num (int): num of images
            rgb_idx_lst (list): list of images indices

            shuffle (bool): shuffle or not

        Returns:
            data (dict): it may contain:
                data['xyzgt'].shape==(500000, 4)
                data['pc'].shape==(10000, 3)
                data['randpts'].shape==(100000, 3)
                data['ctrmag'].shape==(4, )
                data['imgs'].shape==(rgb_rand_num or len(rgb_idx_lst), C, H, W)
                data['imgs_index'].shape==(rgb_rand_num or len(rgb_idx_lst), )
                data['obj_name']
        """
        assert requires_xyzgt or requires_pc or requires_randpts or requires_rgb or requires_ctrmag

        data = dict()

        if requires_ctrmag or requires_xyzgt or requires_pc or requires_randpts:
            ctrmag = to_float32(get_data(self.f, 'ctrmag', obj_index, None))
            if requires_ctrmag:
                data['ctrmag'] = ctrmag  # float32

        if requires_xyzgt:
            xyzgt = to_float32(get_data(self.f, 'pts_sdf', obj_index, sdf_num))
            if shuffle:
                np.random.shuffle(xyzgt)
            xyzgt = (xyzgt - np.array([*ctrmag[:3], 0], dtype=np.float32)) / ctrmag[3]
            data['xyzgt'] = xyzgt

        if requires_pc:
            pc = to_float32(get_data(self.f, 'poisson_pts', obj_index, pc_num))
            if shuffle:
                np.random.shuffle(pc)
            pc = (pc - ctrmag[:3]) / ctrmag[3]
            data['pc'] = pc

        if requires_randpts:
            randpts = to_float32(get_data(self.f, 'rand_pts', obj_index, randpts_num))
            if shuffle:
                np.random.shuffle(randpts)
            randpts = (randpts - ctrmag[:3]) / ctrmag[3]
            data['randpts'] = randpts

        if requires_rgb:
            assert (isinstance(rgb_rand_num, int) and rgb_idx_lst is None) or (rgb_rand_num is None
                                                                               and isinstance(rgb_idx_lst, list))
            if isinstance(rgb_rand_num, int) and rgb_idx_lst is None:
                image_index_list = list(np.random.choice(24, size=rgb_rand_num, replace=False))
            else:
                image_index_list = rgb_idx_lst

            imgs = to_float32(get_data(self.f, 'imgs', obj_index, image_index_list).squeeze())
            imgs = (imgs - 127.5) / 255.0

            if shuffle and imgs.ndim == 4:
                np.random.shuffle(imgs)
            data['imgs'] = imgs
            data['imgs_index'] = np.array(image_index_list)

        data['obj_name'] = self.obj_list[obj_index]

        return data
