# Copyright (c) Gorilla-Lab. All rights reserved.
import sys
import glob
import argparse
import os.path as osp

import torch
import gorilla
import gorilla3d
from tensorboardX import SummaryWriter

import network

def get_parser():
    parser = argparse.ArgumentParser(
        description="Point Cloud Instance Segmentation")
    parser.add_argument("--config",
                        type=str,
                        default="config/default.yaml",
                        help="path to config file")

    args_cfg = parser.parse_args()

    return args_cfg


def init():
    # get the args and read the config file
    args = get_parser()
    cfg = gorilla.Config.fromfile(args.config)

    # get logger file
    log_dir, logger = gorilla.collect_logger(
        prefix=osp.splitext(args.config.split("/")[-1])[0])
    #### NOTE: can initlize the logger manually
    # logger = gorilla.get_logger(log_file)

    # backup the necessary file and directory(Optional, details for source code)
    backup_list = ["train.py", "test.py", "network", args.config]
    backup_dir = osp.join(log_dir, "backup")
    gorilla.backup(backup_dir, backup_list, logger)

    cfg.log_dir = log_dir
    
    # set random seed
    seed = cfg.get("seed", 0)
    gorilla.set_random_seed(seed, logger=logger)

    # log the config
    logger.info("****************** Start Logging *******************")
    logger.info(cfg)

    return logger, cfg


def get_checkpoint(log_dir, epoch=0, checkpoint=""):
    if not checkpoint:
        if epoch > 0:
            checkpoint = osp.join(log_dir, "epoch_{0:05d}.pth".format(epoch))
            assert osp.isfile(checkpoint)
        else:
            latest_checkpoint = glob.glob(osp.join(log_dir, "*latest*.pth"))
            if len(latest_checkpoint) > 0:
                checkpoint = latest_checkpoint[0]
            else:
                checkpoint = sorted(glob.glob(osp.join(log_dir, "*.pth")))
                if len(checkpoint) > 0:
                    checkpoint = checkpoint[-1]
                    epoch = int(checkpoint.split("_")[-1].split(".")[0])

    return checkpoint, epoch + 1

# realize the train process
def do_train(model, cfg, logger):
    model.train()
    # initilize optimizer and scheduler (scheduler is optional-adjust learning rate manually)
    optimizer = gorilla.build_optimizer(model, cfg.optimizer)
    lr_scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)

    # initialize criterion (Optional, can calculate in model forward)
    criterion = gorilla.build_loss(cfg.loss)
    
    # resume model/optimizer/scheduler
    iter = 1
    checkpoint, epoch = get_checkpoint(cfg.log_dir)
    if gorilla.is_filepath(checkpoint): # read valid checkpoint file
        logger.info(f"resume from: {checkpoint}")
        # meta is the dict save some necessary information (last epoch/iteration, acc, loss)
        meta = gorilla.resume(model=model,
                              checkpoint=checkpoint,
                              optimizer=optimizer,     # optimizer and scheduler is optional
                              scheduler=lr_scheduler,  # to resume (can not give these paramters)
                              resume_optimize=True,
                              resume_scheduler=True,
                              strict=False,
                              )
        # get epoch from meta (Optional)
        epoch = meta.get("epoch", epoch)
        iter = meta.get("iter", iter)
    
    # initialize train dataset
    cfg.dataset.split = "train" # change split manually
    train_dataloader = gorilla.build_dataloader(cfg.dataset,
                                                cfg.dataloader,
                                                shuffle=True,
                                                drop_last=True)

    # initialize tensorboard (Optional)
    writer = SummaryWriter(log_dir=cfg.log_dir) # tensorboard writer

    # initialize time buffer and timers (Optional)
    iter_time = gorilla.HistoryBuffer()
    data_time = gorilla.HistoryBuffer()
    iter_timer = gorilla.Timer()

    # loss/time buffer for epoch record (Optional)
    loss_buffer = gorilla.HistoryBuffer()
    epoch_timer = gorilla.Timer()

    while epoch <= cfg.epochs:
        torch.cuda.empty_cache() # (empty cuda cache, Optional)
        for i, batch in enumerate(train_dataloader):
            # calculate data loading time
            data_time.update(iter_timer.since_last())
            # cuda manually (TODO: integrating the data cuda operation)
            point_sets = batch["point_set"].cuda() # [B, N, C]
            labels = batch["label"].long().cuda() # [B, N]

            # model forward and calculate loss
            logits = model(point_sets)
            loss_inp = {"logits": logits,
                        "labels": labels}
            loss, loss_out = criterion(loss_inp) # [B, N, num_class]
            loss_buffer.update(loss)

            # sample the learning rate(Optional)
            lr = optimizer.param_groups[0]["lr"]
            # write tensorboard
            with torch.no_grad():
                writer.add_scalar(f"train/loss", loss, iter)
                writer.add_scalar(f"lr", lr, iter)
                # (NOTE: the `loss_out` is work for multi losses, which saves each loss item)
                # for k, v in loss_out.items():
                #     writer.add_scalar(f"train/{k}", v, iter)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter += 1

            # calculate time and clear timer and buffer(Optional)
            iter_time.update(iter_timer.since_start())
            iter_timer.reset() # record the iteration time and reset timer

            # TODO: the time manager will be integrated into gorilla-core
            # calculate remain time(Optional)
            current_iter = (epoch - 1) * len(train_dataloader) + i + 1
            max_iter = cfg.epochs * len(train_dataloader)
            remain_iter = max_iter - current_iter

            remain_time = remain_iter * iter_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = f"{int(t_h):02d}:{int(t_m):02d}:{int(t_s):02d}"

            sys.stdout.write(
                f"epoch: {epoch}/{cfg.epochs} iter: {i + 1}/{len(train_dataloader)} "
                f"lr: {lr:4f} loss: {loss_buffer.latest:.4f}({loss_buffer.avg:.4f}) "
                f"data_time: {data_time.latest:.2f}({data_time.avg:.2f}) "
                f"iter_time: {iter_time.latest:.2f}({iter_time.avg:.2f}) remain_time: {remain_time}\n")
        
        # updata learning rate scheduler and epoch
        lr_scheduler.step()
        epoch += 1

        # log the epoch information
        logger.info(f"epoch: {epoch}/{cfg.epochss}, train loss: {loss_buffer.avg}, time: {epoch_timer.since_start()}s")

        # write the important information into meta
        meta = {"epoch": epoch,
                "iter": iter}
    
        # save checkpoint
        checkpoint = osp.join(cfg.log_dir, "epoch_{0:05d}.pth".format(epoch))
        gorilla.save_checkpoint(model=model,
                                filename=checkpoint,
                                optimizer=optimizer,
                                scheduler=lr_scheduler,
                                meta=meta)
        logger.info("Saving " + checkpoint)
        # save as latest checkpoint
        latest_checkpoint = osp.join(cfg.log_dir, "epoch_latest.pth")
        gorilla.save_checkpoint(mdoel=model,
                                filename=latest_checkpoint,
                                optimizer=optimizer,
                                scheduler=lr_scheduler,
                                meta=meta)


# realize the test process
def do_test():
    pass


if __name__ == "__main__":
    # init
    logger, cfg = init()

    # model
    logger.info("=> creating model ...")

    # create model
    model = gorilla.build_model(cfg.model)
    model = model.cuda()
    # logger.info("Model:\n{}".format(model)) (Optional print model)

    # count the paramters of model (Optional)
    count_parameters = sum(gorilla.parameter_count(model).values())
    logger.info(f"#classifier parameters new: {count_parameters}")

    # start training
    do_train(model, cfg, logger)

