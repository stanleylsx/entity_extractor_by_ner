# -*- coding: utf-8 -*-
# @Time : 2020/9/9 6:14 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : main.py
# @Software: PyCharm
import argparse
import random
import numpy as np
import os
from engines.train import train
from engines.data import DataManager
from engines.configure import Configure
from engines.utils.logger import get_logger
from engines.predict import Predictor


def set_env(configures):
    random.seed(configures.seed)
    np.random.seed(configures.seed)


def fold_check(configures):
    datasets_fold = 'datasets_fold'
    assert hasattr(configures, datasets_fold), 'item datasets_fold not configured'

    if not os.path.exists(configures.datasets_fold):
        print('datasets fold not found')
        exit(1)

    checkpoints_dir = 'checkpoints_dir'
    if not os.path.exists(configures.checkpoints_dir) or not hasattr(configures, checkpoints_dir):
        print('checkpoints fold not found, creating...')
        paths = configures.checkpoints_dir.split('/')
        if len(paths) == 2 and os.path.exists(paths[0]) and not os.path.exists(configures.checkpoints_dir):
            os.mkdir(configures.checkpoints_dir)
        else:
            os.mkdir('checkpoints')

    vocabs_dir = 'vocabs_dir'
    if not os.path.exists(configures.vocabs_dir):
        print('vocabs fold not found, creating...')
        if hasattr(configures, vocabs_dir):
            os.mkdir(configures.vocabs_dir)
        else:
            os.mkdir(configures.datasets_fold + '/vocabs')

    log_dir = 'log_dir'
    if not os.path.exists(configures.log_dir):
        print('log fold not found, creating...')
        if hasattr(configures, log_dir):
            os.mkdir(configures.log_dir)
        else:
            os.mkdir(configures.datasets_fold + '/vocabs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with BiLSTM+CRF')
    parser.add_argument('--config_file', default='system.config', help='Configuration File')
    args = parser.parse_args()
    configs = Configure(config_file=args.config_file)

    fold_check(configs)
    logger = get_logger(configs.log_dir)
    configs.show_data_summary(logger)
    set_env(configs)
    mode = configs.mode.lower()
    dataManager = DataManager(configs, logger)

    if mode == 'train':
        logger.info('mode: train')
        train(configs, dataManager, logger)
    elif mode == 'test':
        logger.info('mode: test')
        predictor = Predictor(configs, dataManager, logger)
        predictor.test_batch_predict()
    elif mode == 'interactive_predict':
        logger.info('mode: predict_one')
        predictor = Predictor(configs, dataManager, logger)
        while True:
            logger.info('please input a sentence (enter [exit] to exit.)')
            sentence = input()
            if sentence == 'exit':
                break
            results = predictor.predict(sentence)
            print(results)
