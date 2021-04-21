# Copyright (c) Gorilla-Lab. All rights reserved.
import glob
import argparse
import os.path as osp

import torch
import gorilla
import gorilla3d
from tensorboardX import SummaryWriter

import network

def get_parser():
    # the default argument parser contains some 
    # essential parameters for distributed
    parser = gorilla.default_argument_parser()
    parser.add_argument("--config",
                        type=str,
                        default="config/default.yaml",
                        help="path to config file")

    args_cfg = parser.parse_args()

    return args_cfg

# FIXME: you can define the checkpoint rule by yourself
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
                              filename=checkpoint,
                              optimizer=optimizer,     # optimizer and scheduler is optional
                              scheduler=lr_scheduler,  # to resume (can not give these paramters)
                              resume_optimizer=True,
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

    # initialize tensorboard (Optional) TODO: integrating the tensorborad manager
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
            # FIXME: your should define the forward part by yourself(inplace if)
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
            # FIXME: correspond to the above FIXME

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

            print(f"epoch: {epoch}/{cfg.epochs} iter: {i + 1}/{len(train_dataloader)} "
                  f"lr: {lr:4f} loss: {loss_buffer.latest:.4f}({loss_buffer.avg:.4f}) "
                  f"data_time: {data_time.latest:.2f}({data_time.avg:.2f}) "
                  f"iter_time: {iter_time.latest:.2f}({iter_time.avg:.2f}) remain_time: {remain_time}")
        
        # synchronize for distributed training
        gorilla.synchronize()
        # updata learning rate scheduler and epoch
        lr_scheduler.step()

        # log the epoch information
        logger.info(f"epoch: {epoch}/{cfg.epochs}, train loss: {loss_buffer.avg}, time: {epoch_timer.since_start()}s")

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
        gorilla.save_checkpoint(model=model,
                                filename=latest_checkpoint,
                                optimizer=optimizer,
                                scheduler=lr_scheduler,
                                meta=meta)

        epoch += 1


# realize the test process
def do_test():
    pass


def main(args):
    # get the args and read the config file
    cfg = gorilla.Config.fromfile(args.config)

    # get logger file
    log_dir, logger = gorilla.collect_logger(
        prefix=osp.splitext(osp.basename(args.config))[0])
    #### NOTE: can initlize the logger manually
    # logger = gorilla.get_logger(log_file)

    # backup the necessary file and directory(Optional, details for source code)
    # FIXME: if using backup func, you should define backup_list by yourself
    backup_list = ["plain_train.py", "test.py", "network", args.config]
    backup_dir = osp.join(log_dir, "backup")
    gorilla.backup(backup_dir, backup_list, logger)

    # merge the paramters in args into cfg
    cfg = gorilla.config.merge_cfg_and_args(cfg, args)

    cfg.log_dir = log_dir
    
    # set random seed
    seed = cfg.get("seed", 0)
    gorilla.set_random_seed(seed, logger=logger)

    # model
    logger.info("=> creating model ...")

    # create model
    model = gorilla.build_model(cfg.model) # NOTE: can define model manually(do not use the build function)
    model = model.cuda()
    if args.num_gpus > 1:
        # convert the BatchNorm in model as SyncBatchNorm (NOTE: this will be error for low-version pytorch!!!)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # DDP wrap model
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gorilla.get_local_rank()])

    # model = model.cuda()
    # logger.info("Model:\n{}".format(model)) (Optional print model)

    # count the paramters of model (Optional)
    count_parameters = sum(gorilla.parameter_count(model).values())
    logger.info(f"#classifier parameters new: {count_parameters}")

    # start training
    do_train(model, cfg, logger)


if __name__ == "__main__":
    # get the args and read the config file
    args = get_parser()

    # auto using the free gpus
    gorilla.set_cuda_visible_devices(num_gpu=args.num_gpus)

    gorilla.launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,) # use tuple to wrap
    )

