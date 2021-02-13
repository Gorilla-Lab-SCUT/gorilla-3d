# modify from ScanNet function

import os, sys
import inspect
import numpy as np
import logging

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

CLASS_LABELS = [
    "ceiling", "floor", "wall", "beam", "column", "window", "door",
    "table", "chair", "sofa", "bookcase", "board", "clutter"
]
VALID_CLASS_IDS = np.array(range(len(CLASS_LABELS)))
UNKNOWN_ID = np.max(VALID_CLASS_IDS) + 1


# TODO: move out
def get_iou(label_id, confusion):
    if not label_id in VALID_CLASS_IDS:
        return float("nan")
    # #true positives
    tp = np.longlong(confusion[label_id, label_id])
    # #false negatives
    fn = np.longlong(confusion[label_id, :].sum()) - tp
    # #false positives
    not_ignored = [l for l in VALID_CLASS_IDS if not l == label_id]
    fp = np.longlong(confusion[not_ignored, label_id].sum())

    denom = (tp + fp + fn)
    if denom == 0:
        return float("nan")
    return (float(tp) / denom, tp, denom)


def evaluate_scan(data, confusion):
    pred_ids = data["semantic_pred"]
    gt_ids = data["semantic_gt"]
    # sanity checks
    if not pred_ids.shape == gt_ids.shape:
        message = f"{pred_ids.shape}: number of predicted values does not match number of vertices"
        sys.stderr.write("ERROR: " + str(message) + "\n")
        sys.exit(2)

    pred_ids = VALID_CLASS_IDS[pred_ids]
    np.add.at(confusion, (gt_ids, pred_ids), 1)


def evaluate_s3dis(matches, logger=None):
    if logger is not None:
        assert isinstance(logger, logging.Logger)
    max_id = UNKNOWN_ID
    confusion = np.zeros((max_id + 1, max_id + 1), dtype=np.ulonglong)

    def info(message):
        if logger is not None:
            logger.info(message)
        else:
            print(message)

    message = f"evaluating {len(matches)} scans..."
    info(message)
    for i, (scene, data) in enumerate(matches.items()):
        evaluate_scan(data, confusion)
        sys.stdout.write(f"\rscans processed: {i + 1}")
        sys.stdout.flush()
    print("")

    class_ious = {}
    for i in range(len(VALID_CLASS_IDS)):
        label_name = CLASS_LABELS[i]
        label_id = VALID_CLASS_IDS[i]
        class_ious[label_name] = get_iou(label_id, confusion)
    # print
    info("classes          IoU")
    info("----------------------------")
    mean_iou = 0
    for i in range(len(VALID_CLASS_IDS)):
        label_name = CLASS_LABELS[i]
        #print(f"{{label_name:<14s}: class_ious[label_name][0]:>5.3f}")
        info(f"{label_name:<14s}: {class_ious[label_name][0]:>5.3f}   "
             f"({class_ious[label_name][1]:>6d}/{class_ious[label_name][2]:<6d})")
        mean_iou += class_ious[label_name][0]
    mean_iou = mean_iou / len(VALID_CLASS_IDS)
    info(f"mean: {mean_iou:>5.3f}")

