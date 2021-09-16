# Copyright (c) Gorilla-Lab. All rights reserved.
import os
from typing import List

import numpy as np

from .pattern import SemanticEvaluator, InstanceEvaluator

CLASS_LABELS = [
    "unlabeled", "car", "bicycle", "motorcycle", "truck", "bus", "person",
    "bicyclist", "motorcyclist", "road", "parking", "sidewalk", "other-ground",
    "building", "fence", "vegetation", "trunk", "terrain", "pole",
    "traffic-sign"
]

CLASS_IDS = list(range(len(CLASS_LABELS)))


class KittiSemanticEvaluator(SemanticEvaluator):
    def __init__(self,
                 class_labels: List[str] = CLASS_LABELS,
                 class_ids: List[int] = CLASS_IDS,
                 ignore: List[int] = [0],
                 **kwargs):
        super().__init__(class_labels=class_labels,
                         class_ids=class_ids,
                         ignore=ignore,
                         **kwargs)

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
            semantic_pred = output["semantic_pred"].cpu().clone().numpy()
            semantic_gt = output["semantic_gt"].cpu().clone().numpy()
            self.fill_confusion(semantic_pred, semantic_gt)


# ---------- Label info ---------- #
FOREGROUND_CLASS_LABELS = [
    "car", "bicycle", "motorcycle", "truck", "bus", "person", "bicyclist",
    "motorcyclist"
]
FOREGROUND_CLASS_IDS = np.array(range(1, 9))


class KittiInstanceInstanceEvaluator(InstanceEvaluator):
    r"""
    Evaluate instance segmentation metrics.
    """
    def __init__(self,
                 dataset_root: str,
                 class_labels: List[str] = FOREGROUND_CLASS_LABELS,
                 class_ids: List[int] = FOREGROUND_CLASS_IDS,
                 **kwargs):
        r"""
        Args:
            ignore_label: deprecated argument
        """
        super().__init__(class_labels=class_labels,
                         class_ids=class_ids,
                         **kwargs)
        self._dataset_root = dataset_root

    def process(self, inputs, outputs):
        r"""
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
            scene_id, frame_id = scene_name.split("_")
            gt_file = os.path.join(self._dataset_root, scene_id,
                                   frame_id + ".txt")
            gt_ids = np.loadtxt(gt_file)
            self.assign(scene_name, output, gt_ids)
