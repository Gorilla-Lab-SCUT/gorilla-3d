# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import os.path as osp
import logging

import torch
import numpy as np

from gorilla.evaluation import DatasetEvaluator
from ..metric import (evaluate_semantic_kitti)


class KittiSemanticEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """
    def __init__(self):
        """
        Args:
            num_classes, ignore_label: deprecated argument
        """
        self.reset()

    def reset(self):
        self._predictions = {}
        self._gt = {}

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            scene_name = input["scene_name"]
            semantic_pred = output["semantic_pred"].cpu().numpy()
            semantic_gt = output["semantic_gt"].cpu().numpy()
            self._gt[scene_name] = semantic_gt
            self._predictions[scene_name] = semantic_pred

    def evaluate(self):
        r"""
        """
        matches = {}
        for scene_name in self._gt.keys():
            matches[scene_name] = {}
            matches[scene_name]["semantic_gt"] = self._gt[scene_name]
            matches[scene_name]["semantic_pred"] = self._predictions[scene_name]

        evaluate_semantic_kitti(matches)

    @staticmethod
    def read_gt(origin_root, scene_name):
        label = np.load(
            os.path.join(origin_root, scene_name + ".txt_sem_label.npy"))
        return label


