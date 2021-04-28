# Copyright (c) Gorilla-Lab. All rights reserved.
import os.path as osp
from typing import List, Union

import numpy as np

from gorilla.evaluation import DatasetEvaluators

from .pattern import SemanticEvaluator, InstanceEvaluator


CLASS_LABELS = [
    "ceiling", "floor", "wall", "beam", "column", "window", "door",
    "table", "chair", "sofa", "bookcase", "board", "clutter"
]
VALID_CLASS_IDS = np.arange(len(CLASS_LABELS))


class S3DISSemanticEvaluator(SemanticEvaluator):
    def __init__(self,
                 num_classes: int=13,
                 avoid_zero: bool=False,
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



class S3DISInstanceEvaluator(InstanceEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """
    def __init__(self,
                dataset_root: str,
                num_classes: int=13,
                avoid_zero: bool=False,
                class_labels: List[str]=CLASS_LABELS,
                valid_class_ids: List[int]=VALID_CLASS_IDS,
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
        for input, output in zip(inputs, outputs):
            scene_name = input["scene_name"]
            gt_file = osp.join(self._dataset_root, scene_name + ".txt")
            gt_ids = np.loadtxt(gt_file)
            self.assign(scene_name, output, gt_ids)


# class S3DISInstanceEvaluator(DatasetEvaluator):
#     """
#     Evaluate semantic segmentation metrics.
#     """
#     def __init__(self, dataset_root):
#         """
#         Args:
#             num_classes, ignore_label: deprecated argument
#         """
#         self._dataset_root = dataset_root
#         self.reset()

#     def reset(self):
#         self.matches = {}

#     def process(self, inputs, outputs):
#         """
#         Args:
#             inputs: the inputs to a model.
#                 It is a list of dicts.
#             outputs: the outputs of a model. It is either list of semantic segmentation predictions
#                 or list of dicts with key "sem_seg" that contains semantic
#                 segmentation prediction in the same format.
#         """
#         for input, output in zip(inputs, outputs):
#             scene_name = input["scene_name"]
#             gt_file = osp.join(self._dataset_root, scene_name + ".txt")
#             gt2pred, pred2gt = assign_instances_for_scan_s3dis(
#                 scene_name, output, gt_file)
#             self.matches[scene_name] = {
#                 "instance_pred": pred2gt,
#                 "instance_gt": gt2pred
#             }

#     def evaluate(self, ap=True, prec_rec=True):
#         """TODO: modify it
#         Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):
#         * Mean intersection-over-union averaged across classes (mIoU)
#         * Frequency Weighted IoU (fwIoU)
#         * Mean pixel accuracy averaged across classes (mACC)
#         * Pixel Accuracy (pACC)
#         """
#         if ap:
#             ap_scores, prec_recall_total = evaluate_matches_s3dis(self.matches)
#             avgs = compute_averages_s3dis(ap_scores)
#             print_results_s3dis(avgs)
#         if prec_rec:
#             print_prec_recall_s3dis(self.matches)


S3DISEvaluator = DatasetEvaluators(
    [S3DISSemanticEvaluator, S3DISInstanceEvaluator])
