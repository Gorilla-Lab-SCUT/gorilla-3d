# Copyright (c) Gorilla-Lab. All rights reserved.
from typing import List, Union

import numpy as np

from .pattern import SemanticEvaluator


CLASS_LABELS = [
    "car", "bicycle", "motorcycle", "truck", "bus", "person", "bicyclist",
    "motorcyclist", "road", "parking", "sidewalk", "other-ground",
    "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign"
]


VALID_CLASS_IDS = np.arange(len(CLASS_LABELS))

class KittiSemanticEvaluator(SemanticEvaluator):
    def __init__(self,
                 num_classes: int=19,
                 avoid_zero: bool=True,
                 class_labels: List[str]=CLASS_LABELS,
                 valid_class_ids: Union[np.ndarray, List[int]]=VALID_CLASS_IDS,
                 **kwargs):
        super().__init__(num_classes,
                         avoid_zero,
                         class_labels,
                         valid_class_ids,
                         **kwargs)

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            semantic_pred = output["semantic_pred"].cpu().clone().numpy()
            semantic_gt = output["semantic_gt"].cpu().clone().numpy()
            self.fill_confusion(semantic_pred, semantic_gt)


