# Modified from ScanNet evaluation script: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_label.py

import os
import sys
import inspect
from typing import Dict, List, Optional

import gorilla
import numpy as np


currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


# TODO: move out
def get_iou(label_id: int,
            confusion: np.ndarray,
            valid_class_ids: np.ndarray):
    if not label_id in valid_class_ids:
        return float("nan")
    # #true positives
    tp = np.longlong(confusion[label_id, label_id])
    # #false negatives
    fn = np.longlong(confusion[label_id, :].sum()) - tp
    # #false positives
    not_ignored = [l for l in valid_class_ids if not l == label_id]
    fp = np.longlong(confusion[not_ignored, label_id].sum())

    denom = (tp + fp + fn)
    if denom == 0:
        return float("nan")
    return (float(tp) / denom, tp, denom)


def evaluate_scan(data: Dict,
                  confusion: np.ndarray,
                  valid_class_ids: np.ndarray):
    pred_ids = data["semantic_pred"]
    gt_ids = data["semantic_gt"]
    # sanity checks
    if not pred_ids.shape == gt_ids.shape:
        message = f"{pred_ids.shape}: number of predicted values does not match number of vertices"
        sys.stderr.write("ERROR: " + str(message) + "\n")
        sys.exit(2)

    pred_ids = valid_class_ids[pred_ids]
    np.add.at(confusion, (gt_ids, pred_ids), 1)


def evaluate_semantic(matches: Dict,
                      valid_class_ids: Optional[np.ndarray]=None,
                      class_labels: List[str]=["class"]):
    
    max_id = int(valid_class_ids.max() + 1)
    confusion = np.zeros((max_id + 1, max_id + 1), dtype=np.ulonglong)

    logger = gorilla.derive_logger(__name__)

    message = f"evaluating {len(matches)} scans..."
    logger.info(message)
    for i, (scene, data) in enumerate(matches.items()):
        evaluate_scan(data, confusion, valid_class_ids)
        sys.stdout.write(f"\rscans processed: {i + 1}")
        sys.stdout.flush()
    print("")

    class_ious = {}
    for i in range(len(valid_class_ids)):
        label_name = class_labels[i]
        label_id = valid_class_ids[i]
        class_ious[label_name] = get_iou(label_id, confusion, valid_class_ids)
    # print
    logger.info("classes          IoU")
    logger.info("-" * 45)
    mean_iou = 0
    for i in range(len(valid_class_ids)):
        label_name = class_labels[i]
        #print(f"{{label_name:<14s}: class_ious[label_name][0]:>5.3f}")
        logger.info(f"{label_name:<14s}: {class_ious[label_name][0]:>5.3f}   "
             f"({class_ious[label_name][1]:>6d}/{class_ious[label_name][2]:<6d})")
        mean_iou += class_ious[label_name][0]
    mean_iou = mean_iou / len(valid_class_ids)
    logger.info(f"mean: {mean_iou:>5.3f}")
