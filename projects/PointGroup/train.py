"""
PointGroup train.py
Written by Li Jiang
"""
import ipdb
import open3d as o3d
import sys
import time
import argparse
import os.path as osp

import torch
import gorilla
import gorilla3d
import spconv
from torch_scatter import scatter_mean

from pointgroup import (get_checkpoint, pointgroup_ops, PointGroupLoss,
                        align_overseg_semantic_label, PointGroup)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Point Cloud Instance Segmentation")
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
    exp_name = osp.splitext(args.config.split("/")[-1])[0]
    cfg = gorilla.Config.fromfile(args.config)
    cfg.exp_name = exp_name
    cfg.pretrain = args.pretrain
    cfg.semantic = args.semantic
    cfg.exp_path = osp.join("exp", exp_name)

    #### get logger file
    log_file = osp.join(
        cfg.exp_path,
        "{}-{}.log".format(cfg.task,
                           time.strftime("%Y%m%d_%H%M%S", time.localtime())))
    if not gorilla.is_filepath(osp.dirname(log_file)):
        gorilla.mkdir_or_exist(log_file)
    logger = gorilla.get_root_logger(log_file)
    logger.info(
        "************************ Start Logging ************************")

    # log the config
    logger.info(cfg)

    return logger, cfg


class PointGroupSolver(gorilla.BaseSolver):
    @property
    def val_flag(self):
        return gorilla.is_multiple(
            self.epoch, self.cfg.data.save_freq) or gorilla.is_power2(
                self.epoch)

    def build_criterion(self):
        self.criterion = PointGroupLoss(self.cfg)

    def step(self, batch, mode="train"):
        # model_fn defined in PointGroup
        ##### prepare input and forward
        coords = batch["locs"].cuda(
        )  # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        coords_offsets = batch["locs_offset"].cuda()  # (B, 3), long, cuda
        voxel_coords = batch["voxel_locs"].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch["p2v_map"].cuda()  # (N), int, cuda
        v2p_map = batch["v2p_map"].cuda()  # (M, 1 + maxActive), int, cuda

        coords_float = batch["locs_float"].cuda()  # (N, 3), float32, cuda
        feats = batch["feats"].cuda()  # (N, C), float32, cuda
        labels = batch["labels"].cuda()  # (N), long, cuda
        instance_labels = batch["instance_labels"].cuda(
        )  # (N), long, cuda, 0~total_nInst, -100

        instance_info = batch["instance_info"].cuda(
        )  # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        instance_pointnum = batch["instance_pointnum"].cuda(
        )  # (total_nInst), int, cuda

        batch_offsets = batch["offsets"].cuda()  # (B + 1), int, cuda
        overseg = batch["overseg"].cuda()  # (N), long, cuda
        _, overseg = torch.unique(overseg, return_inverse=True)  # (N), long, cuda

        extra_data = {"overseg": overseg,}

        prepare_flag = (self.epoch > cfg.model.prepare_epochs)
        scene_list = batch["scene_list"]
        spatial_shape = batch["spatial_shape"]

        if self.cfg.model.use_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = pointgroup_ops.voxelization(
            feats, v2p_map, cfg.data.mode)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats,
                                         voxel_coords.int(),
                                         spatial_shape,
                                         cfg.data.batch_size)

        ret = self.model(input_,
                         p2v_map,
                         coords_float,
                         coords[:, 0].int(),
                         batch_offsets,
                         coords_offsets,
                         scene_list,
                         epoch,
                         extra_data)

        semantic_scores = ret["semantic_scores"]  # (N, nClass) float32, cuda
        pt_offsets = ret["pt_offsets"]  # (N, 3), float32, cuda

        # overseg semantic align
        overseg_semantic_scores = ret["overseg_semantic_scores"]  # (num_overseg, nClass)
        overseg_labels = align_overseg_semantic_label(labels,
                                                      overseg,
                                                      21)  # (num_overseg)

        overseg_pt_offsets = ret["overseg_pt_offsets"]  # (num_overseg, 3)
        overseg_centers = scatter_mean(coords_float, overseg, dim=0)  # (num_overseg, 3)
        overseg_instance_labels = align_overseg_semantic_label(instance_labels,
                                                               overseg,
                                                               int(instance_labels.max() + 1))  # (num_overseg)

        if prepare_flag:
            scores, proposals_idx, proposals_offset = ret["proposal_scores"]
            # scores: (nProposal, 1) float, cuda
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu

        loss_inp = {}
        loss_inp["batch_idxs"] = coords[:, 0].int()
        loss_inp["overseg"] = overseg
        loss_inp["feats"] = feats
        loss_inp["scene_list"] = scene_list
        loss_inp["batch_offsets"] = batch_offsets

        loss_inp["semantic_scores"] = (semantic_scores, labels)
        loss_inp["pt_offsets"] = (pt_offsets, coords_float, instance_info,
                                  instance_labels)

        loss_inp["overseg_semantic_scores"] = (overseg_semantic_scores, overseg_labels)
        loss_inp["overseg_pt_offsets"] = (overseg_centers,
                                          overseg_pt_offsets,
                                          overseg_instance_labels)

        if prepare_flag:
            loss_inp["proposal_scores"] = (scores,
                                           proposals_idx,
                                           proposals_offset,
                                           instance_pointnum)

        loss, loss_out = self.criterion(loss_inp, self.epoch)

        ##### accuracy / meter_dict
        with torch.no_grad():
            meter_dict = {}
            meter_dict["loss_{}".format(mode)] = (loss.item(), coords.shape[0])
            for k, v in loss_out.items():
                meter_dict["{}_{}".format(k, mode)] = (float(v[0]), v[1])

        self.log_buffer.update(meter_dict)

        return loss

    def solve(self):
        self.build_criterion()
        self.train_data_loader = self.dataloaders[0]
        self.val_data_loader = self.dataloaders[1]
        while self.epoch <= self.cfg.data.epochs:
            self.train()
            if self.val_flag:
                self.evaluate()
            self.epoch += 1

    def train(self):
        self.clear()
        torch.cuda.empty_cache()
        iter_time = gorilla.HistoryBuffer()
        data_time = gorilla.HistoryBuffer()
        model.train()

        epoch_timer = gorilla.Timer()
        iter_timer = gorilla.Timer()

        ##### adjust learning rate
        for i, batch in enumerate(self.train_data_loader):
            # calculate data loading time
            data_time.update(iter_timer.since_last())
            # model step forward and return loss
            loss = self.step(batch)

            ##### backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            ##### time and print
            current_iter = (self.epoch - 1) * len(
                self.train_data_loader) + i + 1
            max_iter = self.cfg.data.epochs * len(self.train_data_loader)
            remain_iter = max_iter - current_iter

            iter_time.update(iter_timer.since_start())

            # reset_timer
            iter_timer.reset()

            remain_time = remain_iter * iter_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = "{:02d}:{:02d}:{:02d}".format(
                int(t_h), int(t_m), int(t_s))

            loss_buffer = self.log_buffer.get("loss_train")
            sys.stdout.write(
                "epoch: {}/{} iter: {}/{} lr: {:4f} loss: {:.4f}({:.4f}) data_time: {:.2f}({:.2f}) iter_time: {:.2f}({:.2f}) remain_time: {remain_time}\n"
                .format(self.epoch,
                        self.cfg.data.epochs,
                        i + 1,
                        len(self.train_data_loader),
                        lr,
                        loss_buffer.latest,
                        loss_buffer.avg,
                        data_time.latest,
                        data_time.avg,
                        iter_time.latest,
                        iter_time.avg,
                        remain_time=remain_time))
                
            if (i == len(self.train_data_loader) - 1): print()

        max_mem = self.get_max_memory()
        logger.info(
            "epoch: {}/{}, train loss: {:.4f}, time: {}s, max_mem: {}M".format(
                self.epoch, self.cfg.data.epochs, loss_buffer.avg,
                epoch_timer.since_start(), max_mem))

        meta = {"epoch": self.epoch}
        filename = osp.join(self.cfg.exp_path,
                            self.cfg.exp_name + "-%09d" % self.epoch + ".pth")
        gorilla.save_checkpoint(self.model, filename, self.optimizer,
                                self.lr_scheduler, meta)

        self.logger.info("Saving " + filename)
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
                sys.stdout.write("\riter: {}/{} loss: {:.4f}({:.4f})".format(
                    i + 1, len(self.val_data_loader), loss_buffer.latest,
                    loss_buffer.avg))
                if (i == len(self.val_data_loader) - 1): print()

            logger.info("epoch: {}/{}, val loss: {:.4f}, time: {}s".format(
                self.epoch, self.cfg.data.epochs, loss_buffer.avg,
                epoch_timer.since_start()))

            self.write()


if __name__ == "__main__":
    ##### init
    logger, cfg = init()

    ##### model
    logger.info("=> creating model ...")

    model = PointGroup(cfg)

    use_cuda = torch.cuda.is_available()
    logger.info("cuda available: {}".format(use_cuda))
    assert use_cuda
    model = model.cuda()

    count_parameters = sum(gorilla.parameter_count(model).values())
    logger.info("#classifier parameters new: {}".format(count_parameters))

    ##### dataset
    train_dataset = gorilla3d.ScanNetV2InstTrainVal(cfg, logger)
    train_dataloader = train_dataset.dataloader
    cfg.task = "val"  # change task
    val_dataset = gorilla3d.ScanNetV2InstTrainVal(cfg, logger)
    val_dataloader = val_dataset.dataloader

    cfg.log = cfg.exp_path
    Trainer = PointGroupSolver(model, [train_dataloader, val_dataloader], cfg,
                               logger)
    checkpoint, epoch = get_checkpoint(cfg.exp_path, cfg.exp_name)
    Trainer.set_epoch(epoch)
    if gorilla.is_filepath(checkpoint):
        Trainer.resume(checkpoint)
    Trainer.solve()
