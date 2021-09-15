# Copyright (c) Gorilla-Lab. All rights reserved.
from typing import List, Tuple, Union

import numpy as np

from .pattern import ClassificationEvaluator

CLASS_LABELS = [
    "bed",
    "tv_stand",
    "xbox",
    "person",
    "night_stand",
    "curtain",
    "bottle",
    "bench",
    "mantel",
    "plant",
    "flower_pot",
    "tent",
    "stairs",
    "radio",
    "monitor",
    "guitar",
    "bathtub",
    "door",
    "piano",
    "cone",
    "keyboard",
    "bowl",
    "airplane",
    "dresser",
    "cup",
    "vase",
    "sofa",
    "range_hood",
    "glass_box",
    "car",
    "bookshelf",
    "lamp",
    "stool",
    "desk",
    "sink",
    "chair",
    "toilet",
    "table",
    "laptop",
    "wardrobe",
]

CLASS_IDS = np.arange(len(CLASS_LABELS))


class ModelNetClassificationEvaluator(ClassificationEvaluator):
    def __init__(self,
                 class_labels: List[str] = CLASS_LABELS,
                 class_ids: Union[np.ndarray, List[int]] = CLASS_IDS,
                 top_k: Tuple[int] = (1, 5),
                 **kwargs):
        super().__init__(class_labels, class_ids, top_k, **kwargs)

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
            prediction = output["prediction"].cpu().clone()
            labels = output["labels"].cpu().clone()
            self.match(prediction, labels)
