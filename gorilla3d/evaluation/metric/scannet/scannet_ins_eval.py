# Modified from ScanNet evaluation script: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py

import sys
import logging
from functools import partial
from typing import Dict, List, Optional

import numpy as np

from ..instance_utils import evaluate_matches, compute_averages, assign_instances_for_scan, print_results, print_prec_recall

# ---------- Label info ---------- #
CLASS_LABELS = [
    "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator", "shower curtain",
    "toilet", "sink", "bathtub", "otherfurniture"
]
VALID_CLASS_IDS = np.array(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
ID_TO_LABEL = {}
for i in range(len(VALID_CLASS_IDS)):
    ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]


evaluate_matches_scannet = partial(evaluate_matches, class_labels=CLASS_LABELS)
compute_averages_scannet = partial(compute_averages, class_labels=CLASS_LABELS)
assign_instances_for_scan_scannet = partial(assign_instances_for_scan,
                                            valid_class_ids=VALID_CLASS_IDS,
                                            class_labels=CLASS_LABELS,
                                            id_to_label=ID_TO_LABEL)
print_results_scannet = partial(print_results, class_labels=CLASS_LABELS)
print_prec_recall_scannet = partial(print_prec_recall, valid_class_ids=VALID_CLASS_IDS, class_labels=CLASS_LABELS)
