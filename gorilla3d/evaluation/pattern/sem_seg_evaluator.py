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
                 class_labels: List[str],
                 class_ids: List[int],
                 ignore: List[int]=[],
                 **kwargs,):
        """
        Args:
            num_classes, ignore_label: deprecated argument
        """
        super().__init__() # init logger
        self.num_classes = num_classes
        self.class_labels = class_labels
        self.class_ids = class_ids
        self.ignore = ignore
        assert len(self.class_labels) == len(self.class_ids), (
            f"all classe labels are {self.class_labels}, length is {len(self.class_labels)}\n"
            f"all class ids are {self.class_ids}, length is {len(self.class_ids)}\n"
            f"their length do not match")
        self.id_to_label = {i : name for (i, name) in zip(self.class_ids, self.class_labels)}
        self.reset()

    @property
    def filter_confusion(self):
        ### get iou of not interesting categories
        not_ignored = [l for l in self.class_ids if l not in self.ignore]
        return self.confusion[not_ignored, :][:, not_ignored] # [num_class, num_class]

    def reset(self):
        max_id = self.class_ids.max() + 1
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

        # print semantic segmentation result(IoU)
        self.print_result()

        # return confusion matrix
        if return_confusion:
            return self.filter_confusion

    def print_result(self):
        # calculate ious
        tp = np.diag(self.filter_confusion) # [num_class]
        denom = self.filter_confusion.sum(1) + self.filter_confusion.sum(0) - np.diag(self.filter_confusion) # [num_class]
        ious = (tp / denom) * 100 # [num_class]

        # build IoU table
        haeders = ["class", "IoU", "TP/(TP+FP+FN)"]
        results = []
        self.logger.info("Evaluation results for semantic segmentation:")
        
        max_length = max(15, max(map(lambda x: len(x), self.class_labels)))
        filterd_class_labels = [v for k, v in self.id_to_label.items() if k not in self.ignore]
        for i, class_label in enumerate(filterd_class_labels):
            results.append((class_label.ljust(max_length, " "), ious[i], f"({tp[i]:>6d}/{denom[i]:<6d})"))
        acc_table = gorilla.table(
            results,
            headers=haeders,
            stralign="left"
        )
        for line in acc_table.split("\n"):
            self.logger.info(line)
        self.logger.info(f"mean: {np.nanmean(ious):.1f}")
        self.logger.info("")



