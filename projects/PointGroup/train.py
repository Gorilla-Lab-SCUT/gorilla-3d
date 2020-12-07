"""
PointGroup train.py
Written by Li Jiang
"""

import open3d as o3d
import os
import os.path as osp
import sys
import time
import argparse

import torch
import gorilla

from pointgroup import (get_log_file, is_multiple, is_power2, checkpoint_save,
                        get_checkpoint, model_fn_decorator, Dataset,
                        PointGroup as Network)


def get_parser():
    parser = argparse.ArgumentParser(description="Point Cloud Segmentation")
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
    log_file = get_log_file(cfg)
    logger = gorilla.get_root_logger(log_file)
    logger.info("************************ Start Logging ************************")

    # log the config
    logger.info(cfg)
    gorilla.set_cuda_visible_devices()

    return logger, cfg


class PointGroupSolver(gorilla.BaseSolver):
    @property
    def val_flag(self):
        return is_multiple(self.epoch, self.cfg.save_freq) or is_power2(
            self.epoch)

    def solve(self, model_fn):
        self.model_fn = model_fn
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
        start_epoch = time.time()
        end = time.time()

        ##### adjust learning rate
        for i, batch in enumerate(self.train_data_loader):
            data_time.update(time.time() - end)

            ##### prepare input and forward
            loss, _, visual_dict, meter_dict = self.model_fn(
                batch, self.model, self.epoch)

            ##### meter_dict
            train_meter_dict = {}
            for key, value in meter_dict.items():
                train_meter_dict["{}_train".format(key)] = value
            self.log_buffer.update(train_meter_dict)

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

            iter_time.update(time.time() - end)
            end = time.time()

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

        logger.info("epoch: {}/{}, train loss: {:.4f}, time: {}s".format(
            self.epoch, self.cfg.data.epochs, loss_buffer.avg,
            time.time() - start_epoch))

        checkpoint_save(self.model, self.optimizer, self.lr_scheduler,
                        self.cfg.exp_path,
                        self.cfg.config.split("/")[-1][:-5], self.epoch,
                        self.cfg.save_freq,
                        logger=self.logger)
        self.write()

    def evaluate(self):
        self.clear()
        self.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        with torch.no_grad():
            model.eval()
            start_epoch = time.time()
            for i, batch in enumerate(self.val_data_loader):

                ##### prepare input and forward
                loss, preds, visual_dict, meter_dict = self.model_fn(
                    batch, self.model, self.epoch)

                ##### meter_dict
                eval_meter_dict = {}
                for key, value in meter_dict.items():
                    eval_meter_dict["{}_eval".format(key)] = value
                self.log_buffer.update(eval_meter_dict)

                loss_buffer = self.log_buffer.get("loss_eval")
                ##### print
                sys.stdout.write("\riter: {}/{} loss: {:.4f}({:.4f})".format(
                    i + 1, len(self.val_data_loader), loss_buffer.latest,
                    loss_buffer.avg))
                if (i == len(self.val_data_loader) - 1): print()

            logger.info("epoch: {}/{}, val loss: {:.4f}, time: {}s".format(
                self.epoch, self.cfg.data.epochs, loss_buffer.avg,
                time.time() - start_epoch))

            self.write()


if __name__ == "__main__":
    ##### init
    logger, cfg = init()

    ##### model
    logger.info("=> creating model ...")

    model = Network(cfg)

    use_cuda = torch.cuda.is_available()
    logger.info("cuda available: {}".format(use_cuda))
    assert use_cuda
    model = model.cuda()

    count_parameters = sum(gorilla.parameter_count(model).values())
    logger.info("#classifier parameters new: {}".format(count_parameters))

    ##### model_fn (criterion)
    model_fn = model_fn_decorator(cfg)

    ##### dataset
    dataset = Dataset(cfg, logger)
    dataset.trainLoader()
    dataset.valLoader()

    cfg.log = cfg.exp_path
    Trainer = PointGroupSolver(
        model, [dataset.train_data_loader, dataset.val_data_loader], cfg,
        logger)
    checkpoint, epoch = get_checkpoint(cfg.exp_path,
                                       cfg.exp_name)
    Trainer.set_epoch(epoch)
    if gorilla.is_filepath(checkpoint):
        Trainer.resume(checkpoint)
    Trainer.solve(model_fn)
