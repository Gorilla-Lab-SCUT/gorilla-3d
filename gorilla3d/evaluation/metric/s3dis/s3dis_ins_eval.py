# Modified from ScanNet evaluation script: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py

import sys
import logging
from functools import partial
from typing import Dict, List, Optional

import numpy as np

from ..instance_utils import evaluate_matches, compute_averages, assign_instances_for_scan, print_results, print_prec_recall

# ---------- Label info ---------- #
CLASS_LABELS = [
    "ceiling", "floor", "wall", "beam", "column", "window", "door",
    "table", "chair", "sofa", "bookcase", "board", "clutter"
]
VALID_CLASS_IDS = np.array(range(len(CLASS_LABELS)))
ID_TO_LABEL = {}
for i in range(len(VALID_CLASS_IDS)):
    ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]


evaluate_matches_s3dis = partial(evaluate_matches, class_labels=CLASS_LABELS)
compute_averages_s3dis = partial(compute_averages, class_labels=CLASS_LABELS)
assign_instances_for_scan_s3dis = partial(assign_instances_for_scan,
                                          valid_class_ids=VALID_CLASS_IDS,
                                          class_labels=CLASS_LABELS,
                                          id_to_label=ID_TO_LABEL)
print_results_s3dis = partial(print_results, class_labels=CLASS_LABELS)
print_prec_recall_s3dis = partial(print_prec_recall, valid_class_ids=VALID_CLASS_IDS, class_labels=CLASS_LABELS)
