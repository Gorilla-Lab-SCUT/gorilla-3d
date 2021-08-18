# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py

import os
import argparse
import torch

import gorilla
import gorilla3d
import gorilla3d.nn as g3n
import gorilla3d.datasets as g3d
import network


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
    cfg.dataset.task = "test"  # change task
    dataset = gorilla.build_dataset(cfg.dataset)
    test_dataloader = gorilla.build_dataloader(dataset,
                                               cfg.dataloader)

    evaluator = gorilla3d.ModelNetClassificationEvaluator()
    with torch.no_grad():
        model = model.eval()

        # init timer to calculate time
        timer = gorilla.Timer()
        for i_iter, batch in enumerate(test_dataloader):
            data_time = timer.since_last()
            point_sets = batch["point_set"].cuda() # [B, N, C]
            labels = batch["label"].long().cuda() # [B, N]

            # model forward and calculate loss
            logits = model(point_sets) # [B, N, num_classes]
            inputs = {"scene_name": i_iter}
            outputs = {"prediction": logits,
                       "labels": labels}
            evaluator.process(inputs, outputs)
            total_time = timer.since_start()
            logger.info(
                f"instance iter: {i_iter + 1}/{len(test_dataloader)} "
                f"time: total {total_time:.2f}s data: {data_time:.2f}s ")
            timer.reset()

        avgs = evaluator.evaluate()


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

    ##### evaluate
    test(model, cfg, logger)



