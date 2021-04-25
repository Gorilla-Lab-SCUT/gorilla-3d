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


def evaluate_scan(data: Dict,
                  confusion: np.ndarray,
                  valid_class_ids: np.ndarray):
    pred_ids = data["semantic_pred"].flatten()
    gt_ids = data["semantic_gt"].flatten()
    # sanity checks
    if not pred_ids.shape == gt_ids.shape:
        message = f"{pred_ids.shape}: number of predicted values does not match number of vertices"
        sys.stderr.write("ERROR: " + str(message) + "\n")
        sys.exit(2)
    # avoid invalid gt ids
    valid_ids = (gt_ids >= 0) & (gt_ids <= valid_class_ids.max())
    pred_ids = pred_ids[valid_ids]
    gt_ids = gt_ids[valid_ids]

    # map prediction ids
    pred_ids = valid_class_ids[pred_ids]

    np.add.at(confusion, (gt_ids, pred_ids), 1)


def evaluate_semantic(matches: Dict,
                      valid_class_ids: np.ndarray=np.array([0]),
                      class_labels: List[str]=["class"],
                      avoid_zero: bool=False):
    r"""
    evaluate the semantic segmentation IoU using confusion matrix

    Args:
        matches (Dict): prediction-gt match for each scene
        valid_class_ids (np.ndarray): class ids for valid classes (maybe not dense, like ScanNet). Defaults to np.array([0])
        class_labels (List[str]): name of each class. Defaults to ["class"]
        avoid_zero (bool): avoid the zero-class prediction(SemanticKitti). Defaults to False
    """

    # avoid the zero class (NOTE: unlabel_id in semantic-kitti)
    if avoid_zero:
        valid_class_ids += 1
        valid_class_ids = np.concatenate([np.zeros(1, dtype=valid_class_ids.dtype), valid_class_ids])

    # initialize
    logger = gorilla.derive_logger(__name__)

    message = f"evaluating {len(matches)} scans..."
    logger.info(message)
    ### define the confusion matrix and process match for each scan
    max_id = int(valid_class_ids.max() + 1)
    confusion = np.zeros((max_id + 1, max_id + 1), dtype=np.ulonglong)
    for scene, data in gorilla.track(matches.items()):
        evaluate_scan(data, confusion, valid_class_ids)

    ### get iou
    # avoid the zero class
    if avoid_zero:
        valid_class_ids = valid_class_ids[1:]
    
    # print semantic segmentation result(IoU)
    print_semantic_result(confusion, valid_class_ids, class_labels)


def print_semantic_result(confusion: np.ndarray,
                          valid_class_ids: np.ndarray=np.array([0]),
                          class_labels: List[str]=["class"],):
    # initialize
    logger = gorilla.derive_logger(__name__)

    not_ignored = [l for l in valid_class_ids]
    filter_confusion = confusion[not_ignored, :][:, not_ignored] # [num_class, num_class]
        
    tp = np.diag(filter_confusion) # [num_class]
    denom = filter_confusion.sum(1) + filter_confusion.sum(0) - np.diag(filter_confusion) # [num_class]
    ious = tp / denom # [num_class]

    #### print
    logger.info("classes          IoU")
    logger.info("-" * 45)
    mean_iou = 0
    for i in range(len(valid_class_ids)):
        label_name = class_labels[i]
        logger.info(f"{label_name:<14s}: {ious[i]:>5.3f}   ({tp[i]:>6d}/{denom[i]:<6d})")
    mean_iou = np.nanmean(ious)
    logger.info(f"mean: {mean_iou:>5.3f}")

