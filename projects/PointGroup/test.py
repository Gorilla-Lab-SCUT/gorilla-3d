"""
PointGroup test.py
Written by Li Jiang
"""
import argparse
import numpy as np
import os

import torch
import spconv
import scipy.stats as stats

import pointgroup_ops
import gorilla
import gorilla3d
import gorilla3d.datasets as g3d
import pointgroup

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
    ### visualize
    parser.add_argument("--visual",
                        type=str,
                        default=None,
                        help="visual path")
    ### log file path
    parser.add_argument("--log-file",
                        type=str,
                        default=None,
                        help="log_file path")

    args_cfg = parser.parse_args()

    return args_cfg


def init():
    args = get_parser()
    cfg = gorilla.Config.fromfile(args.config)
    cfg.pretrain = args.pretrain
    cfg.semantic = args.semantic
    cfg.task = cfg.data.split # the task of test is defined in as data.split
    cfg.visual = args.visual

    gorilla.set_random_seed(cfg.data.test_seed)

    #### get logger file
    params_dict = dict(
        epoch=cfg.data.test_epoch,
        optim=cfg.optimizer.type,
        lr=cfg.optimizer.lr,
        scheduler=cfg.lr_scheduler.type
    )
    if cfg.data.split == "test":
        params_dict["suffix"] = "test"

    log_dir, logger = gorilla.collect_logger(
        prefix=os.path.splitext(args.config.split("/")[-1])[0],
        log_name="test",
        log_file=args.log_file,
        # **params_dict
    )

    logger.info(
        "************************ Start Logging ************************")

    # log the config
    logger.info(cfg)

    global result_dir
    result_dir = os.path.join(
        log_dir, "result",
        "epoch{}_nmst{}_scoret{}_npointt{}".format(cfg.data.test_epoch,
                                                   cfg.data.TEST_NMS_THRESH,
                                                   cfg.data.TEST_SCORE_THRESH,
                                                   cfg.data.TEST_NPOINT_THRESH),
        cfg.data.split)
    os.makedirs(os.path.join(result_dir, "predicted_masks"), exist_ok=True)

    global semantic_label_idx
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    semantic_label_idx = torch.tensor([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
    ]).cuda()

    return logger, cfg


def test(model, cfg, logger):
    logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

    epoch = cfg.data.test_epoch
    semantic = cfg.semantic

    cfg.dataset.task = cfg.data.split  # change task
    cfg.dataset.test_mode = True
    cfg.dataloader.batch_size = 1
    test_dataloader = gorilla.build_dataloader(cfg.dataset,
                                               cfg.dataloader)

    with torch.no_grad():
        model = model.eval()

        # init timer to calculate time
        timer = gorilla.Timer()

        # define evaluator
        # get the real data root
        data_root = os.path.join(os.path.dirname(__file__), cfg.dataset.data_root)
        if "test" in cfg.data.split:
            split = "scans_test"
        else:
            split = "scans"

        semantic_dataset_root = os.path.join(data_root, "scans")
        instance_dataset_root = os.path.join(data_root, cfg.data.split + "_gt")
        evaluator = gorilla3d.ScanNetSemanticEvaluator(semantic_dataset_root)
        inst_evaluator = gorilla3d.ScanNetInstanceEvaluator(instance_dataset_root)

        for i, batch in enumerate(test_dataloader):
            torch.cuda.empty_cache()
            timer.reset()
            N = batch["feats"].shape[0]
            test_scene_name = batch["scene_list"][0]

            coords = batch["locs"].cuda()                # [N, 1 + 3], long, cuda, dimension 0 for batch_idx
            locs_offset = batch["locs_offset"].cuda() # [B, 3], long, cuda
            voxel_coords = batch["voxel_locs"].cuda()    # [M, 1 + 3], long, cuda
            p2v_map = batch["p2v_map"].cuda()            # [N], int, cuda
            v2p_map = batch["v2p_map"].cuda()            # [M, 1 + maxActive], int, cuda

            coords_float = batch["locs_float"].cuda()  # [N, 3], float32, cuda
            feats = batch["feats"].cuda()              # [N, C], float32, cuda

            batch_offsets = batch["offsets"].cuda()    # [B + 1], int, cuda
            scene_list = batch["scene_list"]

            extra_data = {"locs_offset": locs_offset,
                          "scene_list": scene_list}

            spatial_shape = batch["spatial_shape"]

            if cfg.model.use_coords:
                feats = torch.cat((feats, coords_float), 1)
            voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.data.mode)  # [M, C], float, cuda

            input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.dataloader.batch_size)

            data_time = timer.since_last()

            ret = model(input_,
                        p2v_map,
                        coords_float,
                        coords[:, 0].int(),
                        epoch,
                        semantic_only=semantic)
            semantic_scores = ret["semantic_scores"]  # [N, nClass] float32, cuda
            pt_offsets = ret["pt_offsets"]            # [N, 3], float32, cuda
            if (epoch > cfg.model.prepare_epochs) and not semantic:
                scores, proposals_idx, proposals_offset = ret["proposal_scores"]

            ##### preds
            with torch.no_grad():
                preds = {}
                preds["semantic"] = semantic_scores
                preds["pt_offsets"] = pt_offsets
                if (epoch > cfg.model.prepare_epochs) and not semantic:
                    preds["score"] = scores
                    preds["proposals"] = (proposals_idx, proposals_offset)


            ##### get predictions (#1 semantic_pred, pt_offsets; #2 scores, proposals_pred)
            semantic_scores = preds["semantic"]  # [N, nClass=20] float32, cuda
            semantic_pred = semantic_scores.max(1)[1]  # [N] long, cuda

            pt_offsets = preds["pt_offsets"]  # [N, 3], float32, cuda

            ##### semantic segmentation evaluation
            if cfg.data.eval:
                inputs = [{"scene_name": test_scene_name}]
                outputs = [{"semantic_pred": semantic_pred}]
                evaluator.process(inputs, outputs)
            
            prepare_flag = (epoch > cfg.model.prepare_epochs)
            if prepare_flag and not semantic:
                scores = preds["score"]  # [num_prop, 1] float, cuda
                scores_pred = torch.sigmoid(scores.view(-1))

                proposals_idx, proposals_offset = preds["proposals"]
                # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (num_prop + 1), int, cpu
                proposals_pred = torch.zeros(
                    (proposals_offset.shape[0] - 1, N),
                    dtype=torch.int,
                    device=scores_pred.device)  # [num_prop, N], int, cuda
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
                # semantic_id = semantic_label_idx[semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]] # [num_prop], long

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
                    )  # [num_prop, N], float, cuda
                    intersection = torch.mm(
                        proposals_pred_f, proposals_pred_f.t(
                        ))  # [num_prop, num_prop], float, cuda
                    proposals_pointnum = proposals_pred_f.sum(
                        1)  # [num_prop], float, cuda
                    proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(
                        1, proposals_pointnum.shape[0])
                    proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(
                        proposals_pointnum.shape[0], 1)
                    cross_ious = intersection / (proposals_pn_h +
                                                 proposals_pn_v - intersection)
                                                 
                    pick_idxs = gorilla3d.non_max_suppression(
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

            inference_time = timer.since_last()

            ##### visual
            if cfg.visual is not None:
                # visual semantic result
                gorilla.check_dir(cfg.visual)
                if cfg.semantic:
                    pass
                # visual instance result
                else:
                    gorilla3d.visualize_instance_mask(clusters.cpu().numpy(),
                                                      test_scene_name,
                                                      cfg.visual,
                                                      os.path.join(data_root, split),
                                                      cluster_scores.cpu().numpy(),
                                                      semantic_pred.cpu().numpy(),)

            ##### save files
            if cfg.data.save_semantic:
                os.makedirs(os.path.join(result_dir, "semantic"), exist_ok=True)
                semantic_np = semantic_pred.cpu().numpy()
                np.save(
                    os.path.join(result_dir, "semantic", test_scene_name + ".npy"),
                    semantic_np)

            if cfg.data.save_pt_offsets:
                os.makedirs(os.path.join(result_dir, "coords_offsets"),
                            exist_ok=True)
                pt_offsets_np = pt_offsets.cpu().numpy()
                coords_np = batch["locs_float"].numpy()
                coords_offsets = np.concatenate((coords_np, pt_offsets_np),
                                                1)  # [N, 6]
                np.save(
                    os.path.join(result_dir, "coords_offsets",
                             test_scene_name + ".npy"), coords_offsets)

            if (prepare_flag and cfg.data.save_instance):
                f = open(os.path.join(result_dir, test_scene_name + ".txt"), "w")
                for proposal_id in range(nclusters):
                    clusters_i = clusters[proposal_id].cpu().numpy()  # [N]
                    semantic_label = np.argmax(
                        np.bincount(
                            semantic_pred[np.where(clusters_i == 1)[0]].cpu()))
                    score = cluster_scores[proposal_id]
                    f.write(f"predicted_masks/{test_scene_name}_{proposal_id:03d}.txt "
                            f"{semantic_label_idx[semantic_label]} {score:.4f}")
                    if proposal_id < nclusters - 1:
                        f.write("\n")
                    content = list(map(lambda x: str(x), clusters_i.tolist()))
                    content = "\n".join(content)
                    with open(
                            os.path.join(
                                result_dir, "predicted_masks",
                                test_scene_name + "_%03d.txt" % (proposal_id)),
                            "w") as cf:
                        cf.write(content)
                    # np.savetxt(os.path.join(result_dir, "predicted_masks", test_scene_name + "_%03d.txt" % (proposal_id)), clusters_i, fmt="%d")
                f.close()

            save_time = timer.since_last()
            total_time = timer.since_start()

            ##### print
            if semantic:
                logger.info(
                    f"instance iter: {i + 1}/{len(test_dataloader)} point_num: {N} "
                    f"time: total {total_time:.2f}s data: {data_time:.2f}s "
                    f"inference {inference_time:.2f}s save {save_time:.2f}s")
            else:
                logger.info(
                    f"instance iter: {i + 1}/{len(test_dataloader)} point_num: {N} "
                    f"ncluster: {nclusters} time: total {total_time:.2f}s data: {data_time:.2f}s "
                    f"inference {inference_time:.2f}s save {save_time:.2f}s")

        ##### evaluation
        if cfg.data.eval:
            if not semantic:
                inst_evaluator.evaluate(prec_rec=False)
            evaluator.evaluate()


if __name__ == "__main__":
    logger, cfg = init()

    ##### model
    logger.info("=> creating model ...")
    logger.info(f"Classes: {cfg.model.classes}")

    model = gorilla.build_model(cfg.model)

    use_cuda = torch.cuda.is_available()
    logger.info(f"cuda available: {use_cuda}")
    assert use_cuda
    model = model.cuda()

    # logger.info(model)
    logger.info(f"#classifier parameters (model): {sum([x.nelement() for x in model.parameters()])}")

    ##### load model
    gorilla.load_checkpoint(
        model, cfg.pretrain
    )  # resume from the latest epoch, or specify the epoch to restore

    ##### evaluate
    test(model, cfg, logger)
