# modify from ScanNet # Modified from ScanNet evaluation script: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_label.py
from functools import partial

import numpy as np

from ..semantic_utils import evaluate_semantic

CLASS_LABELS = [
    "ceiling", "floor", "wall", "beam", "column", "window", "door",
    "table", "chair", "sofa", "bookcase", "board", "clutter"
]
VALID_CLASS_IDS = np.array(range(len(CLASS_LABELS)))

evaluate_semantic_s3dis = partial(evaluate_semantic, valid_class_ids=VALID_CLASS_IDS, class_labels=CLASS_LABELS)

