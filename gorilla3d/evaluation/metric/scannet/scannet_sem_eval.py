# Modified from ScanNet evaluation script: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_label.py
from functools import partial

import numpy as np

from ..semantic_utils import get_iou, evaluate_scan, evaluate_semantic


CLASS_LABELS = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub",
    "otherfurniture"
]
VALID_CLASS_IDS = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

evaluate_semantic_scannet = partial(evaluate_semantic, valid_class_ids=VALID_CLASS_IDS, class_labels=CLASS_LABELS)

