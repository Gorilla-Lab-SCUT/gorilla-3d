# Copyright (c) Gorilla-Lab. All rights reserved.
import copy
from typing import List, Tuple
from collections import OrderedDict

import torch
import numpy as np

import gorilla

# modify from https://github.com/Megvii-BaseDetection/cvpods/blob/master/cvpods/evaluation/classification_evaluation.py
class ClassificationEvaluator(gorilla.evaluation.DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """
    def __init__(self,
                 num_classes: int,
                 class_labels: List[str],
                 valid_class_ids: List[int],
                 top_k: Tuple[int],
                 **kwargs,):
        """
        Args:
            num_classes, ignore_label: deprecated argument
        """
        super().__init__() # init logger
        self.num_classes = num_classes
        self.class_labels = class_labels
        self.valid_class_ids = valid_class_ids
        self._top_k = top_k
        self.reset()

    def reset(self):
        self._predictions = []
        self._labels = []

    def match(self,
              prediction: np.ndarray,
              label: np.ndarray):
        self._predictions.append(prediction)
        self._labels.append(label)
        

    def evaluate(self, return_acc: bool=False):
        self._predictions = torch.cat(self._predictions).view(-1, self.num_classes) # [N, num_classes]
        self._labels = torch.cat(self._labels).view(-1) # [N]

        # calcualate instance accuracy
        acc = gorilla.evaluation.accuracy(self._predictions, self._labels, self._top_k)
        
        acc_dict = {}
        for i, k in enumerate(self._top_k):
            acc_dict[f"Top_{k} Acc"] = acc[i]
            
        acc_table = gorilla.create_small_table(acc_dict)
        self.logger.info("Evaluation results for classification:")
        self.logger.info("\n"+acc_table)

        totals, corrects = gorilla.accuracy_for_each_class(self._predictions, self._labels.view(-1, 1), self.num_classes) # [num_classes]
        corrects_per_class = (corrects * 100)/ totals # [num_classes]
        headers = ("classes", "Top_1 Acc")
        data = [("mean", str(float(corrects_per_class.mean())))]
        for class_label, correct in zip(self.class_labels, corrects_per_class):
            data.append((str(class_label), str(float(correct))))

        acc_table = gorilla.table(data, headers)
        self.logger.info("\n"+acc_table)
        
        if return_acc:
            return acc

