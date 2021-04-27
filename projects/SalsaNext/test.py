# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py

import os
import argparse

import cv2
import gorilla
import gorilla3d
import torch
import numpy as np
import matplotlib.pyplot as plt

import salsa


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

    log_dir, logger = gorilla.collect_logger(
        prefix=os.path.splitext(args.config.split("/")[-1])[0],
        log_name="test",
        log_file=args.log_file,
    )

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
    cfg.dataloader.batch_size = 16
    cfg.dataloader.num_workers = 32
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
        for i_iter, batch in enumerate(test_dataloader):
            data_time = timer.since_last()
            proj_in = batch["proj"].cuda()
            proj_labels = batch["proj_labels"]
            p_xs = batch["proj_x"].cuda()
            p_ys = batch["proj_y"].cuda()
            npoints = batch["npoints"]

            # forward
            proj_output = model(proj_in)
            proj_argmax = proj_output.argmax(dim=1)

            ## calculate iou for evaluation
            # scan batch
            for i in range(proj_in.shape[0]):
                # first cut to rela size (batch size one allows it)
                npoint = npoints[i]
                p_x = p_xs[i, :npoint]
                p_y = p_ys[i, :npoint]

                unproj_pred_ids = proj_argmax[i, p_y, p_x].cpu().numpy() # [N]
                proj_label = proj_labels[i, p_y, p_x].cpu().numpy() # [N]
                proj_label[proj_label == -1] = 0
                hist_list.append(
                    salsa.fast_hist_crop(
                        unproj_pred_ids, proj_label, unique_label
                    )
                )

            total_time = timer.since_start()
            logger.info(
                f"instance iter: {i_iter + 1}/{len(test_dataloader)} point_num: {npoints.sum()} "
                f"time: total {total_time:.2f}s data: {data_time:.2f}s ")
            timer.reset()

        # avgs = evaluator.evaluate()
        # TODO: package in gorilla3d
        iou = salsa.per_class_iu(sum(hist_list))
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
    gorilla.solver.load_checkpoint(
        model, cfg.pretrain
    )  # resume from the latest epoch, or specify the epoch to restore

    ##### evaluate
    test(model, cfg, logger)



