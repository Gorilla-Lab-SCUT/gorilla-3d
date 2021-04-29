# Copyright (c) Gorilla-Lab. All rights reserved.
from typing import List

import numpy as np

import gorilla


class SemanticEvaluator(gorilla.evaluation.DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """
    def __init__(self,
                 num_classes: int,
                 avoid_zero: bool,
                 class_labels: List[str],
                 valid_class_ids: List[int],
                 **kwargs,):
        """
        Args:
            num_classes, ignore_label: deprecated argument
        """
        super().__init__() # init logger
        self.num_classes = num_classes
        self.avoid_zero = avoid_zero
        self.class_labels = class_labels
        self.valid_class_ids = valid_class_ids
        # avoid the zero class (NOTE: unlabel_id in semantic-kitti)
        if self.avoid_zero:
            self.num_classes += 1
            self.valid_class_ids += 1
        self.reset()

    def reset(self):
        max_id = self.valid_class_ids.max() + 1
        self.confusion = np.zeros((max_id + 1, max_id + 1), dtype=np.int64)

    def fill_confusion(self,
                       pred: np.ndarray,
                       gt: np.ndarray):
        np.add.at(self.confusion, (gt.flatten(), pred.flatten()), 1)
        # # equivalent operation
        # self.confusion += np.bincount(
        #     (self.num_classes + 1) * pred.flatten() + gt.flatten(),
        #     minlength=self.confusion.size,
        # ).reshape(self.confusion.shape)

    def evaluate(self, return_confusion: bool=False):
        ### get iou
        not_ignored = [l for l in self.valid_class_ids]
        filter_confusion = self.confusion[not_ignored, :][:, not_ignored] # [num_class, num_class]

        # print semantic segmentation result(IoU)
        self.print_result(filter_confusion)

        # return confusion matrix
        if return_confusion:
            return filter_confusion

    def print_result(self, filter_confusion):
        # calculate ious
        tp = np.diag(filter_confusion) # [num_class]
        denom = filter_confusion.sum(1) + filter_confusion.sum(0) - np.diag(filter_confusion) # [num_class]
        ious = tp / denom # [num_class]

        #### print
        self.logger.info("classes          IoU")
        self.logger.info("-" * 45)
        mean_iou = 0
        for i, class_label in enumerate(self.class_labels):
            self.logger.info(f"{class_label:<14s}: {ious[i]:>5.3f}   ({tp[i]:>6d}/{denom[i]:<6d})")
        mean_iou = np.nanmean(ious)
        self.logger.info(f"mean: {mean_iou:>5.3f}")


