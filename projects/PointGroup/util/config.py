
import argparse
import os

import gorilla

def get_parser():
    parser = argparse.ArgumentParser(description='Point Cloud Segmentation')
    parser.add_argument('--config', type=str, default='config/pointgroup_default_scannet.yaml', help='path to config file')

    ### pretrain
    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')
    ### semantic only
    parser.add_argument('--semantic', action="store_true", help="only evaluate semantic segmentation")

    args_cfg = parser.parse_args()
    assert args_cfg.config is not None
    config = gorilla.load(args_cfg.config)
    # import ipdb; ipdb.set_trace()
    a = gorilla.Config()
    cfg_origin = gorilla.Config.fromfile(args_cfg.config)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)
        # if key == "SOLVER":
        #     setattr(args_cfg, key, config[key])
        # else:
        #     for k, v in config[key].items():
        #         setattr(args_cfg, k, v)

    return args_cfg

cfg = get_parser()
setattr(cfg, 'exp_path', os.path.join('exp', cfg.dataset, cfg.model_name, cfg.config.split('/')[-1][:-5]))
