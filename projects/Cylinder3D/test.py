# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py

import os.path as osp
import glob
import argparse
import sys
import gorilla
import torch
import numpy as np

import cylinder

import warnings

warnings.filterwarnings("ignore")



def get_parser():
    parser = argparse.ArgumentParser(
        description="Point Cloud Instance Segmentation")
    parser.add_argument("--config",
                        type=str,
                        default="config/kitti.yaml",
                        help="path to config file")
    ### pretrain
    parser.add_argument("--pretrain",
                        type=str,
                        default="",
                        help="path to pretrain model")
    ### log file path
    parser.add_argument("--log-file",
                        type=str,
                        default=None,
                        help="log_file path")

    parser.add_argument("--preload-labels",
                        action="store_true",
                        help="preload labels or not")

    # runing test set or not
    # TODO: add the test set result storage
    parser.add_argument("--test",
                        action="store_true",
                        help="running test set or not")

    args_cfg = parser.parse_args()

    return args_cfg


def init():
    args = get_parser()
    cfg = gorilla.Config.fromfile(args.config)
    cfg.pretrain = args.pretrain
    cfg.dataset.preload_labels = args.preload_labels

    log_dir, logger = gorilla.collect_logger(
        prefix=osp.splitext(args.config.split("/")[-1])[0],
        log_name="test",
        log_file=args.log_file,
    )

    #### get logger file
    log_dir, logger = gorilla.collect_logger(
        prefix=osp.splitext(args.config.split("/")[-1])[0])
    backup_list = ["train.py", "test.py", "cylinder", args.config]
    gorilla.backup(log_dir, backup_list, logger)

    cfg.log_dir = log_dir
    
    seed = cfg.get("seed", 0)
    gorilla.set_random_seed(seed, logger=logger)

    logger.info("****************** Start Logging *******************")

    # log the config
    logger.info(cfg)

    return logger, cfg


def test(model, cfg, logger):
    logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
    cfg.dataset.task = "val"  # change task
    cfg.dataloader.batch_size = 1
    test_dataloader = gorilla.build_dataloader(cfg.dataset,
                                               cfg.dataloader)

    label_name = test_dataloader.dataset.label_name
    unique_label = np.asarray(sorted(list(label_name.keys())))[1:] - 1
    unique_label_str = [label_name[x] for x in unique_label + 1]

    hist_list = []
    with torch.no_grad():
        model = model.eval()

        # init timer to calculate time
        timer = gorilla.Timer()
        for i, batch in enumerate(test_dataloader):
            data_time = timer.since_last()
            (_, voxel_labels, grids, pt_labels, pt_features) = batch
            # voxel_label: [H, W, L], the class labels of voxels
            # grids: list of [N, 3], the voxel indices of points
            # pt_labels: list of [N], the label of points
            # pt_feature: list of [N, 9], features of points, generating from coordinates
            batch_size = len(pt_features)
            pt_features = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in pt_features]
            voxel_indices = [torch.from_numpy(i).cuda() for i in grids]
            voxel_labels = voxel_labels.type(torch.LongTensor).cuda()

            prediction = model(pt_features, voxel_indices, batch_size)
            predict_labels = torch.argmax(prediction, dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()
            for count, grid in enumerate(grids):
                hist_list.append(cylinder.fast_hist_crop(
                    predict_labels[count, grid[:, 0], grid[:, 1], grid[:, 2]],
                    pt_labels[count],
                    unique_label))
            total_time = timer.since_start()
            logger.info(
                f"instance iter: {i + 1}/{len(test_dataloader)} point_num: {len(pt_features[0])} "
                f"time: total {total_time:.2f}s data: {data_time:.2f}s ")
            timer.reset()
        
        # TODO: package in gorilla3d
        iou = cylinder.per_class_iu(sum(hist_list))
        logger.info("Validation per class iou: ")
        for class_name, class_iou in zip(unique_label_str, iou):
            logger.info(f"{class_name:<14s}: {class_iou * 100:>5.3f}")
        val_miou = np.nanmean(iou) * 100
        logger.info(f"Current val miou is {val_miou:>5.3f}")


if __name__ == "__main__":
    ##### init
    logger, cfg = init()

    ##### model
    logger.info("=> creating model ...")

    # TODO: fix here
    model = gorilla.build_model(cfg.model)

    use_cuda = torch.cuda.is_available()
    logger.info(f"cuda available: {use_cuda}")
    assert use_cuda
    model = model.cuda()

    count_parameters = sum(gorilla.parameter_count(model).values())
    logger.info(f"#classifier parameters new: {count_parameters}")

    ##### load model
    gorilla.load_checkpoint(
        model, cfg.pretrain
    )  # resume from the latest epoch, or specify the epoch to restore

    ##### dataset
    cfg.dataset.task = "val"  # change task
    cfg.dataloader.batch_size = 1
    test_dataloader = gorilla.build_dataloader(cfg.dataset,
                                               cfg.dataloader)

    ##### evaluate
    test(model, cfg, logger)



