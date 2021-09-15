# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import h5py
import numpy as np
from typing import List, Union
from torch.utils.data import Dataset
from gorilla import to_float32, find_vcs_root

_DEFAULT_H5_FOLDER = os.path.join(find_vcs_root(__file__), "data", "shapenet",
                                  "shapenet_implicit_recon", "data")


class ShapenetImplicitRecon(Dataset):
    def __init__(
        self,
        h5_dataset_folder: str = _DEFAULT_H5_FOLDER,
        split: str = "train",
        categories: List[str] = ["03001627"],
        requires_xyzgt: bool = True,
        requires_pc: bool = False,
        requires_randpts: bool = False,
        requires_rgb: bool = False,
        requires_ctrmag: bool = False,
        do_normalization: bool = True,
        sdf_num: Union[int, None] = 4096,
        pc_num: Union[int, None] = None,
        randpts_num: Union[int, None] = None,
        rgb_rand_num: Union[int, None] = 1,
        rgb_idx_lst: Union[None, List[int]] = None,
        shuffle: bool = False,
    ):
        r"""Author: lei.jiabao
        Shapenet dataset for implicit field reconstruction.
        It returns point samples, and optionally rgb images.

        Args:
            h5_dataset_folder (str): The folder containing dataset files (.h5 file). 
                These files should be named like "03001627_train.h5".
            split (str, optional): Which split to use. It should be "train", "val" or "test".
                Defaults to "train".
            categories (List[str], optional): Categories to use. Defaults to ["03001627"].
            requires_xyzgt (bool, optional): Requiring 3d point locations and their corresponding sdf values. 
                Sdf values are positive if outside. The num can be specified by `sdf_num`. 
                For example, data["xyzgt"].shape==(500000, 4).
                Defaults to True.
            requires_pc (bool, optional): Requiring surface points sampled by poisson disk sampling.
                The num can be specified by `pc_num`.
                For example, data["pc"].shape==(10000, 3).
                Defaults to False.
            requires_randpts (bool, optional): Requiring surface points sampled by random sampling.
                The num can be specified by `randpts_num`. 
                For example, data["randpts"].shape==(100000, 3).
                Defaults to False.
            requires_rgb (bool, optional): Requiring rgb images. The num can be specified by either `rgb_rand_num` or `rgb_idx_lst`. 
                For example, data["imgs"].shape==(3, 224, 224),    data["imgs_index"].shape==(1, ).
                         or: data["imgs"].shape==(5, 3, 224, 224), data["imgs_index"].shape==(5, ).
                Defaults to False.
            requires_ctrmag (bool, optional): Requiring normalizing factor.
                The first three entries are the center coordinates, the last entry is the magnitude.
                Center can be calculated by: ctr = np.mean(p, axis=0), where p are points sampled randomly from surface.
                Magnitude can be calculated by: mag = np.max(np.linalg.norm(p - ctr, axis=1)).
                For example, data["ctrmag"].shape==(4, )
                Defaults to False.
            do_normalization: (bool, optional): Doing normalization for the loaded data.
                For point cloud, normalizing point locations to unit sphere, and sdf value by the scaling factor.
                For images, normalizing images pixel values to -0.5~0.5.
                Defaults to True.
            sdf_num (int, optional): Num of xyzgt samples to return. If None, use all samples.
                Defaults to 4096.
            pc_num (int, optional): Num of pc samples to return. If None, use all samples.
                Defaults to None.
            randpts_num (int, optional): Num of randpts samples to return. If None, use all samples.
                Defaults to None.
            rgb_rand_num (Union[int, None], optional): Num of the randomly selected rgb images to return.
                Only one of `rgb_rand_num` and `rgb_idx_lst` can be used. Defaults to 1.
            rgb_idx_lst (Union[None, List[int]], optional): Fixed indices of the rgb images to return.
                Only one of `rgb_rand_num` and `rgb_idx_lst` can be used. Defaults to None.
            shuffle (bool, optional): Shuffling after loading from files. It is usually unnecessary because data should be 
                stored out of order in files naturally. Defaults to False.
        """
        super(ShapenetImplicitRecon, self).__init__()
        assert split in ["train", "val", "test"
                         ], "`split` must be \"train\" \"val\" or \"test\""
        self.h5_dataset_folder = h5_dataset_folder
        self.split = split
        self.categories = categories
        self.requires_xyzgt = requires_xyzgt
        self.requires_pc = requires_pc
        self.requires_randpts = requires_randpts
        self.requires_rgb = requires_rgb
        self.requires_ctrmag = requires_ctrmag
        self.do_normalization = do_normalization
        self.sdf_num = sdf_num
        self.pc_num = pc_num
        self.randpts_num = randpts_num
        self.rgb_rand_num = rgb_rand_num
        self.rgb_idx_lst = rgb_idx_lst
        self.shuffle = shuffle

        assert self.categories is not None and len(self.categories) != 0, \
               "`categories` should be a list of str containing all the needed categories"
        self.h5_list = [f"{c}_{self.split}.h5" for c in self.categories]
        self.field_list = [
            ShapenetImplicitReconField(os.path.join(self.h5_dataset_folder, s))
            for s in self.h5_list
        ]

        self.load_configs = dict(
            requires_xyzgt=self.requires_xyzgt,
            requires_pc=self.requires_pc,
            requires_randpts=self.requires_randpts,
            requires_rgb=self.requires_rgb,
            requires_ctrmag=self.requires_ctrmag,
            do_normalization=self.do_normalization,
            sdf_num=self.sdf_num,
            pc_num=self.pc_num,
            randpts_num=self.randpts_num,
            rgb_rand_num=self.rgb_rand_num,
            rgb_idx_lst=self.rgb_idx_lst,
            shuffle=self.shuffle,
        )

        self.eachlen = [len(field) for field in self.field_list]
        self.cumsum = list(np.cumsum(np.array(self.eachlen)))
        self.length = sum(self.eachlen)

    def _parse_index(self, idx):
        class_index = None
        st = 0
        for i in range(len(self.cumsum)):
            ed = self.cumsum[i]
            if idx >= st and idx < ed:
                class_index = i
                break
            st = ed
        obj_index = idx - (self.cumsum[class_index -
                                       1] if class_index != 0 else 0)
        return class_index, obj_index

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> dict:
        """ Get one item for the object specified by `idx`
        Args:
            idx (int): global object index

        Returns:
            data (dict): A dictionary containing all the things you need. 
                All the items are numpy.ndarray in float32 data type.
                Valid Keys are:
                    "ctrmag": the center and the magnitude to normalize
                    "xyzgt": point locations and gt sdf values
                    "pc": poisson points from surface
                    "randpts": randomly sampled points from surface
                    "imgs": images for the objects
                    "imgs_index": images index (in range: 0 <= imgs_index < 24)
                    "obj_name": object name
                    "class_name": category name
        """
        class_index, obj_index = self._parse_index(idx)

        data = self.field_list[class_index].load(obj_index=obj_index,
                                                 **self.load_configs)

        data["class_name"] = self.categories[class_index]

        return data


###########################################################################################################


def _get_data(f, key, obj_index, num=None):
    if num is None:
        return f[key][obj_index, ...]
    elif isinstance(num, list):
        return f[key][obj_index, np.array(num), ...]
    else:
        start_index = np.random.randint(f[key].shape[1] - num + 1)
        return f[key][obj_index, start_index:(start_index + num), ...]


class ShapenetImplicitReconField:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.f = h5py.File(self.file_name, "r")
        obj_list = list(self.f["obj_list"][:])
        self.obj_list = [s.decode("utf-8") for s in obj_list]

    def __del__(self):
        self.f.close()

    def __len__(self):
        return len(self.obj_list)

    def load(
        self,
        obj_index: int,
        requires_xyzgt: bool = True,
        requires_pc: bool = False,
        requires_randpts: bool = False,
        requires_rgb: bool = False,
        requires_ctrmag: bool = False,
        do_normalization: bool = True,
        sdf_num: Union[int, None] = 4096,
        pc_num: Union[int, None] = None,
        randpts_num: Union[int, None] = None,
        rgb_rand_num: Union[int, None] = 1,
        rgb_idx_lst: Union[None, List[int]] = None,
        shuffle: bool = False,
    ) -> dict:
        r"""Author: lei.jiabao
        Load data from file for one object

        Args:
            obj_index (int): The index of one object to load. The following requirements are for the object specified by `obj_index`.
                It should be in range 0 <= obj_index < len(self).
                For example, data["obj_name"]=="1c2b230840baac57df3c82bbe2e3ca02"
            requires_xyzgt (bool, optional): Requiring 3d point locations and their corresponding sdf values. 
                Sdf values are positive if outside. The num can be specified by `sdf_num`. 
                For example, data["xyzgt"].shape==(500000, 4).
                Defaults to True.
            requires_pc (bool, optional): Requiring surface points sampled by poisson disk sampling.
                The num can be specified by `pc_num`.
                For example, data["pc"].shape==(10000, 3).
                Defaults to False.
            requires_randpts (bool, optional): Requiring surface points sampled by random sampling.
                The num can be specified by `randpts_num`. 
                For example, data["randpts"].shape==(100000, 3).
                Defaults to False.
            requires_rgb (bool, optional): Requiring rgb images. The num can be specified by either `rgb_rand_num` or `rgb_idx_lst`. 
                For example, data["imgs"].shape==(3, 224, 224),    data["imgs_index"].shape==(1, ).
                         or: data["imgs"].shape==(5, 3, 224, 224), data["imgs_index"].shape==(5, ).
                Defaults to False.
            requires_ctrmag (bool, optional): Requiring normalizing factor.
                The first three entries are the center coordinates, the last entry is the magnitude.
                Center can be calculated by: ctr = np.mean(p, axis=0), where p are points sampled randomly from surface.
                Magnitude can be calculated by: mag = np.max(np.linalg.norm(p - ctr, axis=1)).
                For example, data["ctrmag"].shape==(4, )
                Defaults to False.
            do_normalization: (bool, optional): Doing normalization for data.
                For point cloud, normalizing point locations to unit sphere, and sdf value by the scaling factor.
                For images, normalizing images pixel values to -0.5~0.5.
                Defaults to True.
            sdf_num (int, optional): Num of xyzgt samples to return. If None, use all samples.
                Defaults to 4096.
            pc_num (int, optional): Num of pc samples to return. If None, use all samples.
                Defaults to None.
            randpts_num (int, optional): Num of randpts samples to return. If None, use all samples.
                Defaults to None.
            rgb_rand_num (Union[int, None], optional): Num of the randomly selected rgb images to return.
                Only one of `rgb_rand_num` and `rgb_idx_lst` can be used. Defaults to 1.
            rgb_idx_lst (Union[None, List[int]], optional): Fixed indices of the rgb images to return.
                Only one of `rgb_rand_num` and `rgb_idx_lst` can be used. Defaults to None.
            shuffle (bool, optional): Shuffling after loading from files. It is usually unnecessary because data should be 
                stored out of order in files naturally. Defaults to False.

        Returns:
            data (dict): A dictionary containing all the things you need. 
                All the items are numpy.ndarray in float32 data type.
                Valid Keys are:
                    "ctrmag": the center and the magnitude to normalize
                    "xyzgt": point locations and gt sdf values
                    "pc": poisson points from surface
                    "randpts": randomly sampled points from surface
                    "imgs": images for the objects
                    "imgs_index": images index (in range: 0 <= imgs_index < 24)
                    "obj_name": object name
        """

        assert requires_xyzgt or requires_pc or requires_randpts or requires_rgb or requires_ctrmag, \
               "must require at least one thing to return"

        data = dict()

        if requires_ctrmag or do_normalization:
            ctrmag = to_float32(_get_data(self.f, "ctrmag", obj_index, None))
            if requires_ctrmag:
                data["ctrmag"] = ctrmag

        if requires_xyzgt:
            xyzgt = to_float32(_get_data(self.f, "pts_sdf", obj_index,
                                         sdf_num))
            if shuffle:
                np.random.shuffle(xyzgt)
            if do_normalization:
                xyzgt = (xyzgt - np.array([*ctrmag[:3], 0], dtype=np.float32)
                         ) / ctrmag[3]  # normalize to unit sphere
            data["xyzgt"] = xyzgt

        if requires_pc:
            pc = to_float32(_get_data(self.f, "poisson_pts", obj_index,
                                      pc_num))
            if shuffle:
                np.random.shuffle(pc)
            if do_normalization:
                pc = (pc - ctrmag[:3]) / ctrmag[3]  # normalize to unit sphere
            data["pc"] = pc

        if requires_randpts:
            randpts = to_float32(
                _get_data(self.f, "rand_pts", obj_index, randpts_num))
            if shuffle:
                np.random.shuffle(randpts)
            if do_normalization:
                randpts = (randpts -
                           ctrmag[:3]) / ctrmag[3]  # normalize to unit sphere
            data["randpts"] = randpts

        if requires_rgb:
            assert (isinstance(rgb_rand_num, int) and rgb_idx_lst is None) or \
                   (rgb_rand_num is None and isinstance(rgb_idx_lst, list)), \
                   "only one of `rgb_rand_num` and `rgb_idx_lst` can be used"
            if isinstance(rgb_rand_num, int) and rgb_idx_lst is None:
                image_index_list = list(
                    np.random.choice(24, size=rgb_rand_num, replace=False))
            else:
                image_index_list = rgb_idx_lst

            imgs = to_float32(
                _get_data(self.f, "imgs", obj_index,
                          image_index_list).squeeze())
            if do_normalization:
                imgs = (imgs - 127.5) / 255.0  # to -0.5~0.5

            if shuffle and imgs.ndim == 4:
                np.random.shuffle(imgs)  # shuffle images order
            data["imgs"] = imgs
            data["imgs_index"] = np.array(image_index_list)

        data["obj_name"] = self.obj_list[obj_index]

        return data


if __name__ == "__main__":
    # testing
    dataset = ShapenetImplicitRecon()
    x = dataset.__getitem__(555)
    import ipdb
    ipdb.set_trace()
