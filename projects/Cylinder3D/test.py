
import os
import argparse

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
    cfg.dataloader.batch_size = 4
    cfg.dataloader.num_workers = 8
    test_dataloader = gorilla.build_dataloader(cfg.dataset,
                                               cfg.dataloader)

    evaluator = gorilla3d.KittiSemanticEvaluator()
    with torch.no_grad():
        model = model.eval()

        # init timer to calculate time
        timer = gorilla.Timer()
        for i_iter, batch in enumerate(test_dataloader):
            data_time = timer.since_last()
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
            predict_labels = torch.argmax(prediction, dim=1)
            inputs = [{"scene_name": i_iter}]
            outputs = [{"semantic_pred": predict_labels[grid_inds[:, 0], grid_inds[:, 1], grid_inds[:, 2], grid_inds[:, 3]],
                        "semantic_gt": pt_labels}]
            evaluator.process(inputs, outputs)
            total_time = timer.since_start()
            logger.info(
                f"instance iter: {i_iter + 1}/{len(test_dataloader)} point_num: {len(pt_features)} "
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



