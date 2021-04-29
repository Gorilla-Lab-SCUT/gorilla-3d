# Copyright (c) Gorilla-Lab. All rights reserved.
import itertools
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
        

    def evaluate(self, show_per_class: bool=True, return_acc: bool=False):
        self._predictions = torch.cat(self._predictions).view(-1, self.num_classes) # [N, num_classes]
        self._labels = torch.cat(self._labels).view(-1) # [N]

        # calcualate instance accuracy
        acc = gorilla.evaluation.accuracy(self._predictions, self._labels, self._top_k)
        
        acc_dict = {}
        for i, k in enumerate(self._top_k):
            acc_dict[f"Top_{k} Acc"] = acc[i]
            
        acc_table = gorilla.create_small_table(acc_dict, tablefmt="psql",)
        self.logger.info("Evaluation results for classification:")
        for line in acc_table.split("\n"):
            self.logger.info(line)

        if show_per_class:
            totals, corrects = gorilla.accuracy_for_each_class(self._predictions, self._labels.view(-1, 1), self.num_classes) # [num_classes]
            corrects_per_class = (corrects * 100)/ totals # [num_classes]

            self.logger.info("Top_1 Acc of each class")
            # tabulate it
            N_COLS = min(8, len(self.class_labels) * 2)
            acc_per_class = [(self.class_labels[i], float(corrects_per_class[i])) for i in range(len(self.class_labels))]
            acc_flatten = gorilla.concat_list(acc_per_class)
            results_2d = itertools.zip_longest(*[acc_flatten[i::N_COLS] for i in range(N_COLS)])
            acc_table = gorilla.table(
                results_2d,
                headers=["class", "Acc"] * (N_COLS // 2),
                tablefmt="psql",
            )
            for line in acc_table.split("\n"):
                self.logger.info(line)
            self.logger.info(f"mean: {corrects_per_class.mean():.4f}")
        
        if return_acc:
            return acc

