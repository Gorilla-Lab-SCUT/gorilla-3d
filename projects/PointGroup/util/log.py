'''
Logger
Written by Li Jiang
'''

import logging
import os
import sys
import time

import gorilla

def get_log_file(cfg):
    if cfg.task == 'train':
        log_file = os.path.join(
            cfg.exp_path,
            'train-{}.log'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        )
    elif cfg.task == 'trainval':
        log_file = os.path.join(
            cfg.exp_path,
            'trainval-{}.log'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        )
    elif cfg.task == 'val':
        log_file = os.path.join(
            cfg.exp_path,
            'val-{}.log'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        )
    elif cfg.task == 'val_mini':
        log_file = os.path.join(
            cfg.exp_path,
            'val_mini-{}.log'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        )
    elif cfg.task == 'test':
        log_file = os.path.join(
            cfg.exp_path, 'result', 'epoch{}_nmst{}_scoret{}_npointt{}'.format(cfg.test_epoch, cfg.TEST_NMS_THRESH, cfg.TEST_SCORE_THRESH, cfg.TEST_NPOINT_THRESH),
            cfg.split, 'test-{}.log'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        )

    if not gorilla.is_filepath(os.path.dirname(log_file)):
        gorilla.mkdir_or_exist(log_file)
    return log_file

