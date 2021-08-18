# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import glob

import torch

import gorilla
import gorilla3d
import gorilla3d.datasets as g3d
import cylinder

def get_parser():
    # FIXME: you should add your argument here
    #        the `default_argument_parser` contains some
    #        essential parameters for distributed
    parser = gorilla.default_argument_parser()
    parser.add_argument("--config",
                        type=str,
                        default="config/kitti.yaml",
                        help="path to config file")
    parser.add_argument("--not-val",
                        action="store_true",
                        help="preload labels or not")

    args_cfg = parser.parse_args()

    return args_cfg

# FIXME: you can define the checkpoint rule by yourself
def get_checkpoint(log_dir, epoch=0, checkpoint=""):
    if not checkpoint:
        if epoch > 0:
            checkpoint = os.path.join(log_dir, "epoch_{0:05d}.pth".format(epoch))
            assert os.path.isfile(checkpoint)
        else:
            latest_checkpoint = glob.glob(os.path.join(log_dir, "*latest*.pth"))
            if len(latest_checkpoint) > 0:
                checkpoint = latest_checkpoint[0]
            else:
                checkpoint = sorted(glob.glob(os.path.join(log_dir, "*.pth")))
                if len(checkpoint) > 0:
                    checkpoint = checkpoint[-1]
                    epoch = int(checkpoint.split("_")[-1].split(".")[0])

    return checkpoint

# realize the train process
def do_train(model, cfg, logger):
    model.train()
    # initilize optimizer and scheduler (scheduler is optional-adjust learning rate manually)
    optimizer = gorilla.build_optimizer(model, cfg.optimizer)
    lr_scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)

    # initialize criterion (Optional, can calculate in model forward)
    criterion = gorilla.build_loss(cfg.loss)
    
    # resume model/optimizer/scheduler
    checkpoint = get_checkpoint(cfg.log_dir)
    meta = {}
    if gorilla.is_filepath(checkpoint): # read valid checkpoint file
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
    epoch = meta.get("epoch", 0) + 1
    iter = meta.get("iter", 0) + 1
    
    # initialize train dataset
    cfg.dataset.split = "train" # change split manually
    train_dataloader = gorilla.build_dataloader(cfg.dataset,
                                                cfg.dataloader,
                                                shuffle=True,
                                                drop_last=True)

    # initialize tensorboard (Optional)
    # NOTE: this is a wrapper of `tensorboardX.SummaryWriter`
    #       it support the `add_scalar` and `add_scalars`
    #       API, which are the same as SummaryWriter.
    #       Otherwise, it contain a data buffer for recording
    writer = gorilla.TensorBoardWriter(cfg.log_dir)

    # initialize timers (Optional)
    iter_timer = gorilla.Timer()
    epoch_timer = gorilla.Timer()

    # loss/time buffer for epoch record (Optional)
    # NOTE: HistoryBuffer can be seen as a List with "clear", `avg` and `sum` API
    loss_buffer = gorilla.HistoryBuffer()
    iter_time = gorilla.HistoryBuffer()
    data_time = gorilla.HistoryBuffer()

    while epoch <= cfg.solver.epochs:
        torch.cuda.empty_cache() # (empty cuda cache, Optional)
        for i, batch in enumerate(train_dataloader):
            # calculate data loading time (`update` for HistoryBuffer can be seen as `append` for List)
            data_time.update(iter_timer.since_last())
            ### FIXME: your should define the forward part by yourself(inplace if)
            voxel_centers = batch["voxel_centers"]
            voxel_labels = batch["voxel_labels"]
            grid_inds = batch["grid_inds"]
            pt_labels = batch["point_labels"]
            pt_xyzs = batch["point_xyzs"]
            pt_features = batch["point_features"]
            # voxel_labels: [H, W, L], the class labels of voxels
            # voxel_label_conuts: [H, W, L, num_class], the class labels count voxels
            # grid_inds: [N, 4], the voxel indices
            # pt_xyzs: [N, 3], coordinates of points, generating from coordinates
            # pt_features: [N, 9], features of points, generating from coordinates
            pt_features = pt_features.cuda()
            voxel_indices = grid_inds.cuda()
            labels = voxel_labels.cuda()

            prediction = model(pt_features, voxel_indices)
            ret = {
                "prediction": prediction,
                "labels": labels}
            loss = criterion(ret)
            # FIXME: correspond to the above FIXME

            loss_buffer.update(loss)
            # sample the learning rate(Optional)
            lr = optimizer.param_groups[0]["lr"]
            #### write tensorboard (NOTE: 3 equivalent solutions)
            # # solution1: (record the data dict into buffer)
            # writer.update({"train/loss": loss, "lr": lr})
            # writer.write(iter)
            # solution2:
            writer.update({"train/loss": loss, "lr": lr}, iter) # given the `global_step` means write immediately
            # # # solution3:
            # writer.add_scalar(f"train/loss", loss, iter)
            # writer.add_scalar(f"lr", lr, iter)
            # # (NOTE: the `loss_out` is work for multi losses, which saves each loss item)
            # for k, v in loss_out.items():
            #     writer.add_scalar(f"train/{k}", v, iter)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter += 1

            # calculate time and reset timer(Optional)
            iter_time.update(iter_timer.since_start())
            iter_timer.reset() # record the iteration time and reset timer

            # TODO: the time manager will be integrated into gorilla-core
            # calculate remain time(Optional)
            remain_iter = (cfg.solver.epochs - epoch + 1) * len(train_dataloader) + i + 1
            remain_time = gorilla.convert_seconds(remain_iter * iter_time.avg) # convert seconds into "hours:minutes:sceonds"

            print(f"epoch: {epoch}/{cfg.solver.epochs} iter: {i + 1}/{len(train_dataloader)} "
                  f"lr: {lr:4f} loss: {loss_buffer.latest:.4f}({loss_buffer.avg:.4f}) "
                  f"data_time: {data_time.latest:.2f}({data_time.avg:.2f}) "
                  f"iter_time: {iter_time.latest:.2f}({iter_time.avg:.2f}) eta: {remain_time}")
        
        # synchronize for distributed training
        gorilla.synchronize()
        # updata learning rate scheduler and epoch
        lr_scheduler.step()

        # log the epoch information
        logger.info(f"epoch: {epoch}/{cfg.solver.epochs}, train loss: {loss_buffer.avg}, time: {epoch_timer.since_start()}s")
        # clear all buffer
        iter_time.clear()
        data_time.clear()
        loss_buffer.clear()

        # write the important information into meta
        meta = {"epoch": epoch,
                "iter": iter,
                "loss": loss_buffer.avg}
    
        # save checkpoint
        checkpoint = os.path.join(cfg.log_dir, "epoch_{0:05d}.pth".format(epoch))
        gorilla.save_checkpoint(model=model,
                                filename=checkpoint,
                                meta=meta)
        logger.info("Saving " + checkpoint)
        # save as latest checkpoint (contain optimizer and lr_scheduler)
        latest_checkpoint = os.path.join(cfg.log_dir, "epoch_latest.pth")
        gorilla.save_checkpoint(model=model,
                                filename=latest_checkpoint,
                                optimizer=optimizer,
                                scheduler=lr_scheduler,
                                meta=meta)

        epoch += 1


def main(args):
    # read config file
    cfg = gorilla.Config.fromfile(args.config)
    # merge the paramters in args into cfg
    cfg = gorilla.config.merge_cfg_and_args(cfg, args)

    # get logger file
    log_dir, logger = gorilla.collect_logger(
        prefix=os.path.splitext(os.path.basename(args.config))[0])
    #### NOTE: can initlize the logger manually
    # logger = gorilla.get_logger(log_file)
    cfg.log_dir = log_dir

    # backup the necessary file and directory(Optional, details for source code)
    # FIXME: if using backup func, you should define backup_list by yourself
    backup_list = ["plain_train.py", "test.py", "cylinder", args.config]
    backup_dir = os.path.join(log_dir, "backup")
    gorilla.backup(backup_dir, backup_list)

    
    # set random seed
    seed = cfg.get("seed", 0)
    gorilla.set_random_seed(seed)

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

    # logger.info("Model:\n{}".format(model)) (Optional print model)

    # count the paramters of model (Optional)
    count_parameters = sum(gorilla.parameter_count(model).values())
    logger.info(f"#classifier parameters new: {count_parameters}")

    # start training
    do_train(model, cfg, logger)


if __name__ == "__main__":
    # get the args
    args = get_parser()

    # # auto using the free gpus(NOTE: need to fix)
    # gorilla.set_cuda_visible_devices(num_gpu=args.num_gpus)

    # launcher (necessary for distributed)
    gorilla.launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,) # use tuple to wrap
    )

