# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import os.path as osp
from typing import List, Union

import numpy as np

from gorilla.evaluation import DatasetEvaluators

from .pattern import SemanticEvaluator, InstanceEvaluator


CLASS_LABELS = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub",
    "otherfurniture"
]
VALID_CLASS_IDS = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])


class ScanNetSemanticEvaluator(SemanticEvaluator):
    def __init__(self,
                 dataset_root,
                 num_classes: int=20,
                 avoid_zero: bool=False,
                 class_labels: List[str]=CLASS_LABELS,
                 valid_class_ids: Union[np.ndarray, List[int]]=VALID_CLASS_IDS,
                 **kwargs):
        super().__init__(num_classes,
                         avoid_zero,
                         class_labels,
                         valid_class_ids,
                         **kwargs)
        self.dataset_root = dataset_root

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        if not isinstance(inputs, List):
            inputs = [inputs]
        if not isinstance(outputs, List):
            outputs = [outputs]
        for input, output in zip(inputs, outputs):
            scene_name = input["scene_name"]
            semantic_gt = self.read_gt(osp.join(self.dataset_root, scene_name), scene_name)
            semantic_pred = output["semantic_pred"].cpu().clone().numpy()
            semantic_pred = self.valid_class_ids[semantic_pred]
            self.fill_confusion(semantic_pred, semantic_gt)

    @staticmethod
    def read_gt(origin_root, scene_name):
        label = np.load(
            os.path.join(origin_root, scene_name + ".txt_sem_label.npy"))
        return label


# ---------- Label info ---------- #
FOREGROUND_CLASS_LABELS = [
    "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator", "shower curtain",
    "toilet", "sink", "bathtub", "otherfurniture"
]
FOREGROUND_VALID_CLASS_IDS = np.array(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])


class ScanNetInstanceEvaluator(InstanceEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """
    def __init__(self,
                dataset_root: str,
                num_classes: int=18,
                avoid_zero: bool=False,
                class_labels: List[str]=FOREGROUND_CLASS_LABELS,
                valid_class_ids: List[int]=FOREGROUND_VALID_CLASS_IDS,
                **kwargs):
        """
        Args:
            num_classes, ignore_label: deprecated argument
        """
        super().__init__(num_classes,
                         avoid_zero,
                         class_labels,
                         valid_class_ids,
                         **kwargs)
        self._dataset_root = dataset_root

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts.
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        if not isinstance(inputs, List):
            inputs = [inputs]
        if not isinstance(outputs, List):
            outputs = [outputs]
        for input, output in zip(inputs, outputs):
            scene_name = input["scene_name"]
            gt_file = osp.join(self._dataset_root, scene_name + ".txt")
            gt_ids = np.loadtxt(gt_file)
            self.assign(scene_name, output, gt_ids)


ScanNetEvaluator = DatasetEvaluators(
    [ScanNetSemanticEvaluator, ScanNetInstanceEvaluator])
