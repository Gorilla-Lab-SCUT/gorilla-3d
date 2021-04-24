# Modified from ScanNet evaluation script: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_label.py
from functools import partial

import numpy as np

from ..semantic_utils import evaluate_semantic

CLASS_LABELS = [
    "car", "bicycle", "motorcycle", "truck", "bus", "person", "bicyclist",
    "motorcyclist", "road", "parking", "sidewalk", "other-ground",
    "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign"
]


VALID_CLASS_IDS = np.arange(len(CLASS_LABELS))

evaluate_semantic_kitti = partial(evaluate_semantic, valid_class_ids=VALID_CLASS_IDS, class_labels=CLASS_LABELS, avoid_zero=True)

