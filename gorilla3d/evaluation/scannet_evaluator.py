# Copyright (c) Gorilla-Lab. All rights reserved.
import os.path as osp

import torch
import numpy as np

from gorilla.evaluation import DatasetEvaluator, DatasetEvaluators
from .metric import read_gt, evaluate

class ScanNetSemanticEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self, dataset_root, num_classes=None, ignore_label=None, logger=None
    ):
        """
        Args:
            num_classes, ignore_label: deprecated argument
        """
        self._dataset_root = dataset_root
        self._num_classes = num_classes
        self._ignore_label = ignore_label if ignore_label is not None else ignore_label
        self.logger = logger
        self.reset()

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._predictions = {}
        self._gt = {}

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts.
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            scene_name = input["scene_name"]
            semantic_gt = read_gt(osp.join(self._dataset_root, scene_name), scene_name)
            semantic_pred = output.cpu().numpy()
            self._gt[scene_name] = semantic_gt
            self._predictions[scene_name] = semantic_pred


    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):
        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        matches = {}
        for scene_name in self._gt.keys():
            matches[scene_name] = {}
            matches[scene_name]["semantic_gt"] = self._gt[scene_name]
            matches[scene_name]["semantic_pred"] = self._predictions[scene_name]

        evaluate(matches, self.logger)


