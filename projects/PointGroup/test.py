"""
PointGroup test.py
Written by Li Jiang
"""
import open3d as o3d
import argparse
import time
import numpy as np
import os
import os.path as osp

import torch
import gorilla
import gorilla3d
import scipy.stats as stats

from pointgroup import (model_fn_decorator, PointGroup as Network)


def get_parser():
    parser = argparse.ArgumentParser(description="Point Cloud Instance Segmentation")
    parser.add_argument("--config",
                        type=str,
                        default="config/pointgroup_default_scannet.yaml",
                        help="path to config file")
    ### pretrain
    parser.add_argument("--pretrain",
                        type=str,
                        default="",
                        help="path to pretrain model")
    ### semantic only
    parser.add_argument("--semantic",
                        action="store_true",
                        help="only evaluate semantic segmentation")

    args_cfg = parser.parse_args()

    return args_cfg


def init():
    args = get_parser()
    exp_name = args.config.split("/")[-1][:-5]
    cfg = gorilla.Config.fromfile(args.config)
    cfg.pretrain = args.pretrain
    cfg.semantic = args.semantic
    cfg.exp_path = osp.join("exp", exp_name)
    cfg.task = cfg.data.split # the task of test is defined in as data.split

    gorilla.set_random_seed(cfg.data.test_seed)

    #### get logger file
    if cfg.data.split == "test":
        log_file = os.path.join(
            cfg.exp_path, "result", "epoch{}_nmst{}_scoret{}_npointt{}".format(cfg.data.test_epoch, cfg.data.TEST_NMS_THRESH, cfg.data.TEST_SCORE_THRESH, cfg.data.TEST_NPOINT_THRESH),
            cfg.data.split, "test-{}.log".format(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        )
    else:
        log_file = osp.join(
            cfg.exp_path,
            "{}-{}.log".format(cfg.data.split, time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        )
    if not gorilla.is_filepath(osp.dirname(log_file)):
        gorilla.mkdir_or_exist(log_file)
    logger = gorilla.get_root_logger(log_file)
    logger.info(
        "************************ Start Logging ************************")

    # log the config
    logger.info(cfg)

    global result_dir
    result_dir = osp.join(
        cfg.exp_path, "result",
        "epoch{}_nmst{}_scoret{}_npointt{}".format(cfg.data.test_epoch,
                                                   cfg.data.TEST_NMS_THRESH,
                                                   cfg.data.TEST_SCORE_THRESH,
                                                   cfg.data.TEST_NPOINT_THRESH),
        cfg.data.split)
    os.makedirs(osp.join(result_dir, "predicted_masks"), exist_ok=True)

    global semantic_label_idx
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    semantic_label_idx = torch.tensor([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
    ]).cuda()

    return logger, cfg


def test(model, model_fn, cfg, logger):
    logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

    epoch = cfg.data.test_epoch
    semantic = cfg.semantic

    test_dataset = gorilla3d.ScanNetV2InstTest(cfg, logger)
    test_dataloader = test_dataset.dataloader

    with torch.no_grad():
        model = model.eval()
        start = time.time()

        semantic_dataset_root = osp.join(cfg.data.data_root, "scannetv2", "scans")
        instance_dataset_root = osp.join(cfg.data.data_root, "scannetv2", cfg.data.split + "_gt")
        evaluator = gorilla3d.ScanNetSemanticEvaluator(semantic_dataset_root,
                                                       logger=logger)
        inst_evaluator = gorilla3d.ScanNetInstanceEvaluator(
            instance_dataset_root, logger=logger)

        for i, batch in enumerate(test_dataloader):
            N = batch["feats"].shape[0]
            test_scene_name = batch["scene_list"][0]

            start1 = time.time()
            preds = model_fn(batch, model, epoch, semantic)
            end1 = time.time() - start1

            ##### get predictions (#1 semantic_pred, pt_offsets; #2 scores, proposals_pred)
            semantic_scores = preds["semantic"]  # (N, nClass=20) float32, cuda
            semantic_pred = semantic_scores.max(1)[1]  # (N) long, cuda

            pt_offsets = preds["pt_offsets"]  # (N, 3), float32, cuda

            ##### semantic segmentation evaluation
            if cfg.data.eval:
                inputs = [{"scene_name": test_scene_name}]
                outputs = [{"semantic_pred": semantic_pred}]
                evaluator.process(inputs, outputs)
            
            prepare_flag = (epoch > cfg.model.prepare_epochs)
            if prepare_flag and not semantic:
                scores = preds["score"]  # (nProposal, 1) float, cuda
                scores_pred = torch.sigmoid(scores.view(-1))

                proposals_idx, proposals_offset = preds["proposals"]
                # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (nProposal + 1), int, cpu
                proposals_pred = torch.zeros(
                    (proposals_offset.shape[0] - 1, N),
                    dtype=torch.int,
                    device=scores_pred.device)  # (nProposal, N), int, cuda
                proposals_pred[proposals_idx[:, 0].long(),
                               proposals_idx[:, 1].long()] = 1
                semantic_pred_list = []
                for start, end in zip(proposals_offset[:-1],
                                      proposals_offset[1:]):
                    semantic_label, _ = stats.mode(
                        semantic_pred[proposals_idx[start:end,
                                                    1].long()].cpu().numpy())
                    semantic_label = semantic_label[0]
                    semantic_pred_list.append(semantic_label)

                semantic_id = semantic_label_idx[semantic_pred_list]
                # semantic_id = semantic_label_idx[semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]] # (nProposal), long

                ##### score threshold
                score_mask = (scores_pred > cfg.data.TEST_SCORE_THRESH)
                scores_pred = scores_pred[score_mask]
                proposals_pred = proposals_pred[score_mask]
                semantic_id = semantic_id[score_mask]

                ##### npoint threshold
                proposals_pointnum = proposals_pred.sum(1)
                npoint_mask = (proposals_pointnum > cfg.data.TEST_NPOINT_THRESH)
                scores_pred = scores_pred[npoint_mask]
                proposals_pred = proposals_pred[npoint_mask]
                semantic_id = semantic_id[npoint_mask]

                ##### nms
                if semantic_id.shape[0] == 0:
                    pick_idxs = np.empty(0)
                else:
                    proposals_pred_f = proposals_pred.float(
                    )  # (nProposal, N), float, cuda
                    intersection = torch.mm(
                        proposals_pred_f, proposals_pred_f.t(
                        ))  # (nProposal, nProposal), float, cuda
                    proposals_pointnum = proposals_pred_f.sum(
                        1)  # (nProposal), float, cuda
                    proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(
                        1, proposals_pointnum.shape[0])
                    proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(
                        proposals_pointnum.shape[0], 1)
                    cross_ious = intersection / (proposals_pn_h +
                                                 proposals_pn_v - intersection)
                    pick_idxs = non_max_suppression(
                        cross_ious.cpu().numpy(),
                        scores_pred.cpu().numpy(),
                        cfg.data.TEST_NMS_THRESH)  # int, (nCluster, N)
                clusters = proposals_pred[pick_idxs]
                cluster_scores = scores_pred[pick_idxs]
                cluster_semantic_id = semantic_id[pick_idxs]

                nclusters = clusters.shape[0]

                ##### prepare for evaluation
                if cfg.data.eval:
                    pred_info = {}
                    pred_info["scene_name"] = test_scene_name
                    pred_info["conf"] = cluster_scores.cpu().numpy()
                    pred_info["label_id"] = cluster_semantic_id.cpu().numpy()
                    pred_info["mask"] = clusters.cpu().numpy()
                    inst_evaluator.process(inputs, [pred_info])

            ##### save files
            start3 = time.time()
            if cfg.data.save_semantic:
                os.makedirs(osp.join(result_dir, "semantic"), exist_ok=True)
                semantic_np = semantic_pred.cpu().numpy()
                np.save(
                    osp.join(result_dir, "semantic", test_scene_name + ".npy"),
                    semantic_np)

            if cfg.data.save_pt_offsets:
                os.makedirs(osp.join(result_dir, "coords_offsets"),
                            exist_ok=True)
                pt_offsets_np = pt_offsets.cpu().numpy()
                coords_np = batch["locs_float"].numpy()
                coords_offsets = np.concatenate((coords_np, pt_offsets_np),
                                                1)  # (N, 6)
                np.save(
                    osp.join(result_dir, "coords_offsets",
                             test_scene_name + ".npy"), coords_offsets)

            if (prepare_flag and cfg.data.save_instance):
                f = open(osp.join(result_dir, test_scene_name + ".txt"), "w")
                for proposal_id in range(nclusters):
                    clusters_i = clusters[proposal_id].cpu().numpy()  # (N)
                    semantic_label = np.argmax(
                        np.bincount(
                            semantic_pred[np.where(clusters_i == 1)[0]].cpu()))
                    score = cluster_scores[proposal_id]
                    f.write("predicted_masks/{}_{:03d}.txt {} {:.4f}".format(
                        test_scene_name, proposal_id,
                        semantic_label_idx[semantic_label], score))
                    if proposal_id < nclusters - 1:
                        f.write("\n")
                    content = list(map(lambda x: str(x), clusters_i.tolist()))
                    content = "\n".join(content)
                    with open(
                            osp.join(
                                result_dir, "predicted_masks",
                                test_scene_name + "_%03d.txt" % (proposal_id)),
                            "w") as cf:
                        cf.write(content)
                    # np.savetxt(osp.join(result_dir, "predicted_masks", test_scene_name + "_%03d.txt" % (proposal_id)), clusters_i, fmt="%d")
                f.close()
            end3 = time.time() - start3
            end = time.time() - start
            start = time.time()

            ##### print
            if semantic:
                logger.info(
                    "instance iter: {}/{} point_num: {} time: total {:.2f}s inference {:.2f}s save {:.2f}s".format(
                        i + 1, len(test_dataset), N, end, end1, end3))
            else:
                logger.info(
                    "instance iter: {}/{} point_num: {} ncluster: {} time: total {:.2f}s inference {:.2f}s save {:.2f}s".format(
                        i + 1, len(test_dataset), N, nclusters, end, end1, end3))

        ##### evaluation
        if cfg.data.eval:
            if not semantic:
                inst_evaluator.evaluate()
            evaluator.evaluate()


def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


if __name__ == "__main__":
    logger, cfg = init()

    ##### model
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(cfg.model.classes))

    model = Network(cfg)

    use_cuda = torch.cuda.is_available()
    logger.info("cuda available: {}".format(use_cuda))
    assert use_cuda
    model = model.cuda()

    # logger.info(model)
    logger.info("#classifier parameters (model): {}".format(
        sum([x.nelement() for x in model.parameters()])))

    ##### model_fn (criterion)
    model_fn = model_fn_decorator(cfg, test=True)

    ##### load model
    gorilla.load_checkpoint(
        model, cfg.pretrain
    )  # resume from the latest epoch, or specify the epoch to restore

    ##### evaluate
    test(model, model_fn, cfg, logger)
