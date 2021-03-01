# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import os.path as osp

import torch
import numpy as np

from gorilla.evaluation import DatasetEvaluator, DatasetEvaluators
from ..metric import (evaluate_semantic_scannet, assign_instances_for_scan_scannet,
                      evaluate_matches_scannet, compute_averages_scannet, print_results_scannet,
                      print_prec_recall_scannet)


class ScanNetSemanticEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """
    def __init__(self, dataset_root, logger=None):
        """
        Args:
            num_classes, ignore_label: deprecated argument
        """
        self._dataset_root = dataset_root
        self.logger = logger
        self.reset()

    def reset(self):
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
            semantic_gt = self.read_gt(osp.join(self._dataset_root, scene_name), scene_name)
            semantic_pred = output["semantic_pred"].cpu().numpy()
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
            matches[scene_name]["semantic_pred"] = self._predictions[
                scene_name]

        evaluate_semantic_scannet(matches, self.logger)

    @staticmethod
    def read_gt(origin_root, scene_name):
        label = np.load(
            os.path.join(origin_root, scene_name + ".txt_sem_label.npy"))
        return label

class ScanNetInstanceEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """
    def __init__(self, dataset_root, logger=None):
        """
        Args:
            num_classes, ignore_label: deprecated argument
        """
        self._dataset_root = dataset_root
        self.logger = logger
        self.reset()

    def reset(self):
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
            gt_file = osp.join(self._dataset_root, scene_name + ".txt")
            gt2pred, pred2gt = assign_instances_for_scan_scannet(
                scene_name, output, gt_file)
            self._gt[scene_name] = gt2pred
            self._predictions[scene_name] = pred2gt

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
            matches[scene_name]["gt"] = self._gt[scene_name]
            matches[scene_name]["pred"] = self._predictions[scene_name]

        ap_scores, prec_recall_total = evaluate_matches_scannet(matches)
        avgs = compute_averages_scannet(ap_scores)
        print_results_scannet(avgs, self.logger)
        print_prec_recall_scannet(matches, logger=self.logger)


ScanNetEvaluator = DatasetEvaluators(
    [ScanNetSemanticEvaluator, ScanNetInstanceEvaluator])
