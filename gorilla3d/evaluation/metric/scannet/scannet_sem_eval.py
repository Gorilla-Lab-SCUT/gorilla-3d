# Evaluates semantic label task
# Input:
#   - path to .txt prediction files
#   - path to .txt ground truth files
#   - output file to write results to
# Note that only the valid classes are used for evaluation,
# i.e., any ground truth label not in the valid label set
# is ignored in the evaluation.
#
# example usage: evaluate_semantic_label.py --scan_path [path to scan data] --output_file [output file]

# python imports
import os, sys, argparse
import inspect
import numpy as np
import logging

try:
    from itertools import izip
except ImportError:
    izip = zip

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

CLASS_LABELS = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub",
    "otherfurniture"
]
VALID_CLASS_IDS = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
CAT_MAP = {i: cls_id for i, cls_id in enumerate(VALID_CLASS_IDS)}
UNKNOWN_ID = np.max(VALID_CLASS_IDS) + 1


def read_gt(origin_root, scene_name):
    label = np.load(
        os.path.join(origin_root, scene_name + ".txt_sem_label.npy"))
    return label


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
        message = "{}: number of predicted values does not match number of vertices".format(
            pred_ids.shape)
        sys.stderr.write("ERROR: " + str(message) + "\n")
        sys.exit(2)

    pred_ids = VALID_CLASS_IDS[pred_ids]
    np.add.at(confusion, (gt_ids, pred_ids), 1)


def evaluate(matches, logger=None):
    if logger is not None:
        assert isinstance(logger, logging.Logger)
    max_id = UNKNOWN_ID
    confusion = np.zeros((max_id + 1, max_id + 1), dtype=np.ulonglong)

    def info(message):
        if logger is not None:
            logger.info(message)
        else:
            print(message)

    message = "evaluating {} scans...".format(len(matches))
    info(message)
    for i, (scene, data) in enumerate(matches.items()):
        evaluate_scan(data, confusion)
        sys.stdout.write("\rscans processed: {}".format(i + 1))
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
        #print("{{0:<14s}: 1:>5.3f}".format(label_name, class_ious[label_name][0]))
        info("{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})".format(
            label_name, class_ious[label_name][0], class_ious[label_name][1],
            class_ious[label_name][2]))
        mean_iou += class_ious[label_name][0]
    mean_iou = mean_iou / len(VALID_CLASS_IDS)
    info("mean: {:>5.3f}".format(mean_iou))
