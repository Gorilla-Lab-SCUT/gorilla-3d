# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import sys
import glob
import argparse

import torch
import spconv

import pointgroup_ops
import gorilla
import gorilla3d
import gorilla3d.datasets as g3d
import pointgroup


def get_parser():
    parser = argparse.ArgumentParser(
        description="Point Cloud Instance Segmentation")
    parser.add_argument("--config",
                        type=str,
                        default="config/default.yaml",
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
    cfg = gorilla.Config.fromfile(args.config)
    cfg.pretrain = args.pretrain
    cfg.semantic = args.semantic

    #### get logger file
    log_dir, logger = gorilla.collect_logger(
        prefix=os.path.splitext(args.config.split("/")[-1])[0])
    backup_list = ["train.py", "test.py", "pointgroup", args.config]
    gorilla.backup(log_dir, backup_list, logger)

    cfg.log_dir = log_dir
    
    seed = cfg.get("seed", 0)
    gorilla.set_random_seed(seed, logger=logger)

    logger.info("****************** Start Logging *******************")

    # log the config
    logger.info(cfg)

    return logger, cfg


class PointGroupSolver(gorilla.BaseSolver):
    @property
    def val_flag(self):
        return gorilla.is_multiple(
            self.epoch, self.cfg.solver.save_freq) or gorilla.is_power2(
                self.epoch)

    def build_criterion(self):
        self.criterion = gorilla.build_loss(self.cfg.loss)
    
    def build_dataloaders(self):
        self.train_data_loader = self.dataloaders[0]
        self.val_data_loader = self.dataloaders[1]

    def step(self, batch, mode="train"):
        # model_fn defined in PointGroup
        ##### prepare input and forward
        coords = batch["locs"].cuda() # [N, 1 + 3], long, cuda, dimension 0 for batch_idx
        locs_offset = batch["locs_offset"].cuda()  # [B, 3], long, cuda
        voxel_coords = batch["voxel_locs"].cuda()  # [M, 1 + 3], long, cuda
        p2v_map = batch["p2v_map"].cuda()  # [N], int, cuda
        v2p_map = batch["v2p_map"].cuda()  # [M, 1 + maxActive], int, cuda

        coords_float = batch["locs_float"].cuda()  # [N, 3], float32, cuda
        feats = batch["feats"].cuda()  # [N, C], float32, cuda
        semantic_labels = batch["semantic_labels"].cuda()  # [N], long, cuda
        instance_labels = batch["instance_labels"].cuda(
        )  # [N], long, cuda, 0~total_num_inst, -100

        instance_info = batch["instance_info"].cuda(
        )  # [N, 9], float32, cuda, (meanxyz, minxyz, maxxyz)
        instance_pointnum = batch["instance_pointnum"].cuda(
        )  # [total_num_inst], int, cuda

        batch_offsets = batch["offsets"].cuda()  # [B + 1], int, cuda

        prepare_flag = (self.epoch > self.cfg.model.prepare_epochs)
        scene_list = batch["scene_list"]
        spatial_shape = batch["spatial_shape"]

        if self.cfg.model.use_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = pointgroup_ops.voxelization(
            feats, v2p_map, self.cfg.data.mode)  # [M, C], float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats,
                                         voxel_coords.int(),
                                         spatial_shape,
                                         self.cfg.dataloader.batch_size)

        ret = self.model(input_,
                         p2v_map,
                         coords_float,
                         coords[:, 0].int(),
                         self.epoch)

        semantic_scores = ret["semantic_scores"]  # [N, nClass] float32, cuda
        pt_offsets = ret["pt_offsets"]  # [N, 3], float32, cuda

        loss_inp = {}
        loss_inp["batch_idxs"] = coords[:, 0].int()
        loss_inp["feats"] = feats
        loss_inp["scene_list"] = scene_list

        loss_inp["semantic_scores"] = (semantic_scores, semantic_labels)
        loss_inp["pt_offsets"] = (pt_offsets,
                                  coords_float,
                                  instance_info,
                                  instance_labels)


        if prepare_flag:
            scores, proposals_idx, proposals_offset = ret["proposal_scores"]
            # scores: (num_prop, 1) float, cuda
            # proposals_idx: (sum_points, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (num_prop + 1), int, cpu

            loss_inp["proposal_scores"] = (scores,
                                           proposals_idx,
                                           proposals_offset,
                                           instance_pointnum)

        loss, loss_out = self.criterion(loss_inp, self.epoch)

        ##### accuracy / meter_dict
        with torch.no_grad():
            meter_dict = {}
            meter_dict[f"loss_{mode}"] = (loss.item(), coords.shape[0])
            for k, v in loss_out.items():
                meter_dict[f"{k}_{mode}"] = (float(v[0]), v[1])

        self.log_buffer.update(meter_dict)

        return loss

    def solve(self):
        self.build_criterion()
        self.train_data_loader = self.dataloaders[0]
        self.val_data_loader = self.dataloaders[1]
        while self.epoch <= self.cfg.solver.epochs:
            self.train()
            if self.val_flag:
                self.evaluate()
            self.epoch += 1

    def train(self):
        self.clear()
        iter_time = gorilla.HistoryBuffer()
        data_time = gorilla.HistoryBuffer()
        model.train()

        epoch_timer = gorilla.Timer()
        iter_timer = gorilla.Timer()

        for i, batch in enumerate(self.train_data_loader):
            torch.cuda.empty_cache()
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

        self.lr_scheduler.step()

        self.logger.info(
            "epoch: {}/{}, train loss: {:.4f}, time: {}s".format(
                self.epoch, self.cfg.solver.epochs, loss_buffer.avg,
                epoch_timer.since_start()))

        meta = {"epoch": self.epoch}
        checkpoint = os.path.join(self.cfg.log_dir, "epoch_{0:05d}.pth".format(self.epoch))
        latest_checkpoint = os.path.join(self.cfg.log_dir, "epoch_latest.pth")
        gorilla.save_checkpoint(self.model, checkpoint, self.optimizer,
                                self.lr_scheduler, meta)
        # save as latest checkpoint
        gorilla.save_checkpoint(self.model, latest_checkpoint, self.optimizer,
                                self.lr_scheduler, meta)

        self.logger.info("Saving " + checkpoint)
        self.write()

    def evaluate(self):
        self.clear()
        self.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

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

            self.logger.info(f"epoch: {self.epoch}/{self.cfg.solver.epochs}, "
                             f"val loss: {loss_buffer.avg:.4f}, time: {epoch_timer.since_start()}s")

            self.write()


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
    train_dataset = gorilla.build_dataset(cfg.dataset)
    train_dataloader = gorilla.build_dataloader(cfg.dataset,
                                                cfg.dataloader,
                                                shuffle=True,
                                                drop_last=True)
    cfg.dataset.task = "val"  # change task
    val_dataloader = gorilla.build_dataloader(cfg.dataset,
                                              cfg.dataloader)

    Trainer = PointGroupSolver(model,
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
