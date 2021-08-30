# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py

import os
import glob
import argparse
import sys

import torch

import gorilla
import gorilla3d
import gorilla3d.datasets as g3d
import cylinder


def get_parser():
    parser = argparse.ArgumentParser(
        description="Point Cloud Instance Segmentation")
    parser.add_argument("--config",
                        type=str,
                        default="config/kitti.yaml",
                        help="path to config file")

    args_cfg = parser.parse_args()

    return args_cfg


def init():
    args = get_parser()
    cfg = gorilla.Config.fromfile(args.config)

    #### get logger file
    log_dir, logger = gorilla.collect_logger(
        prefix=os.path.splitext(args.config.split("/")[-1])[0])
    backup_list = ["train.py", "test.py", "cylinder", args.config]
    backup_dir = os.path.join(log_dir, "backup")
    gorilla.backup(backup_dir, backup_list, logger)

    cfg.log_dir = log_dir
    
    seed = cfg.get("seed", 0)
    gorilla.set_random_seed(seed, logger=logger)

    logger.info("****************** Start Logging *******************")

    # log the config
    logger.info(cfg)

    return logger, cfg


class CylinderSolver(gorilla.BaseSolver):
    @property
    def val_flag(self):
        return self.iter % self.cfg.solver.save_freq == 0

    def build_criterion(self):
        self.criterion = gorilla.build_loss(self.cfg.loss)
    
    def build_dataloaders(self):
        self.train_data_loader = self.dataloaders[0]
        self.val_data_loader = self.dataloaders[1]

    def step(self, batch, mode="train"):
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

        prediction = self.model(pt_features, voxel_indices)
        ret = {
            "prediction": prediction,
            "labels": labels}
        loss = self.criterion(ret)

        ##### accuracy / meter_dict
        with torch.no_grad():
            meter_dict = {}
            meter_dict[f"loss_{mode}"] = (loss.item(), sum(map(lambda x: len(x), pt_features)))

        self.log_buffer.update(meter_dict)

        return loss

    # TODO: modify to support save according to iteration
    def solve(self):
        self.build_criterion()
        self.train_data_loader = self.dataloaders[0]
        self.val_data_loader = self.dataloaders[1]
        while self.epoch <= self.cfg.solver.epochs:
            self.train()
            self.epoch += 1

    def train(self):
        self.clear()
        iter_time = gorilla.HistoryBuffer()
        data_time = gorilla.HistoryBuffer()
        model.train()

        epoch_timer = gorilla.Timer()
        iter_timer = gorilla.Timer()

        for i, batch in enumerate(self.train_data_loader):
            # torch.cuda.empty_cache()
            # calculate data loading time
            data_time.update(iter_timer.since_last())
            # model step forward and return loss
            loss = self.step(batch)

            ##### backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            lr = self.optimizer.param_groups[0]["lr"]

            ##### time and print
            current_iter = (self.epoch - 1) * len(
                self.train_data_loader) + i + 1
            max_iter = self.cfg.solver.epochs * len(self.train_data_loader)
            remain_iter = max_iter - current_iter

            iter_time.update(iter_timer.since_start())

            # reset_timer
            iter_timer.reset()

            remain_time = remain_iter * iter_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = f"{int(t_h):02d}:{int(t_m):02d}:{int(t_s):02d}"

            loss_buffer = self.log_buffer.get("loss_train")
            sys.stdout.write(
                f"epoch: {self.epoch}/{self.cfg.solver.epochs} iter: {i + 1}/{len(self.train_data_loader)} "
                f"lr: {lr:4f} loss: {loss_buffer.latest:.4f}({loss_buffer.avg:.4f}) "
                f"data_time: {data_time.latest:.2f}({data_time.avg:.2f}) "
                f"iter_time: {iter_time.latest:.2f}({iter_time.avg:.2f}) eta: {remain_time}\n")
                
            if (i == len(self.train_data_loader) - 1): print()

            self.iter += 1
            if self.val_flag:
                if self.cfg.solver.val:
                    self.evaluate()
                meta = {"epoch": self.epoch, "iter": self.iter}
                checkpoint = os.path.join(self.cfg.log_dir, "iter_{0:8d}.pth".format(self.epoch))
                latest_checkpoint = os.path.join(self.cfg.log_dir, "iter_latest.pth")
                gorilla.save_checkpoint(self.model, checkpoint, self.optimizer,
                                        self.lr_scheduler, meta)
                # save as latest checkpoint
                gorilla.save_checkpoint(self.model, latest_checkpoint, self.optimizer,
                                        self.lr_scheduler, meta)

        self.lr_scheduler.step()

        self.logger.info(
            "epoch: {}/{}, train loss: {:.4f}, time: {}s".format(
                self.epoch, self.cfg.solver.epochs, loss_buffer.avg,
                epoch_timer.since_start()))

        self.logger.info("Saving " + checkpoint)
        self.write()

    def evaluate(self):
        self.clear()
        self.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        label_name = self.val_data_loader.dataset.label_name
        with torch.no_grad():
            model.eval()
            epoch_timer = gorilla.Timer()
            for i, batch in enumerate(self.val_data_loader):
                loss = self.step(batch, mode="eval")

                loss_buffer = self.log_buffer.get("loss_eval")
                ##### print
                sys.stdout.write(f"\riter: {i + 1}/{len(self.val_data_loader)} "
                                 f"loss: {loss_buffer.latest:.4f}({loss_buffer.avg:.4f})")
                if (i == len(self.val_data_loader) - 1): print()

            logger.info(f"epoch: {self.epoch}/{self.cfg.solver.epochs}, "
                        f"val loss: {loss_buffer.avg:.4f}, time: {epoch_timer.since_start()}s")

            self.write()


# check checkpoint and auto load the latest
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

    return checkpoint, epoch + 1


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

    ##### dataset
    # get the real data root
    cfg.dataset.task = "train"  # change task
    train_dataloader = gorilla.build_dataloader(cfg.dataset,
                                                cfg.dataloader,
                                                prefetch=True,
                                                shuffle=True,
                                                drop_last=True)
    cfg.dataset.task = "val"  # change task
    cfg.dataloader.batch_size = 1
    val_dataloader = gorilla.build_dataloader(cfg.dataset,
                                              cfg.dataloader,
                                              prefetch=True)

    Trainer = CylinderSolver(model,
                             [train_dataloader, val_dataloader],
                             cfg)

    checkpoint, epoch = get_checkpoint(cfg.log_dir)
    Trainer.epoch = epoch
    if gorilla.is_filepath(checkpoint):
        Trainer.resume(
            checkpoint,
            strict=False,
            # # choice
            # resume_optimizer=False,
            # resume_scheduler=False
        )
    Trainer.solve()



