# -*- coding: utf-8 -*-
# @Time : 2020/9/13 3:18 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : configure.py
# @Software: PyCharm
import sys


class Configure:
    def __init__(self, config_file='system.config'):
        config = self.config_file_to_dict(config_file)

        # Status:
        the_item = 'mode'
        if the_item in config:
            self.mode = config[the_item]

        # Datasets(Input/Output):
        the_item = 'datasets_fold'
        if the_item in config:
            self.datasets_fold = config[the_item]
        the_item = 'train_file'
        if the_item in config:
            self.train_file = config[the_item]
        the_item = 'dev_file'
        if the_item in config:
            self.dev_file = self.str2none(config[the_item])
        else:
            self.dev_file = None
        the_item = 'test_file'
        if the_item in config:
            self.test_file = self.str2none(config[the_item])
        else:
            self.test_file = None

        the_item = 'delimiter'
        if the_item in config:
            self.delimiter = config[the_item]

        the_item = 'use_pretrained_model'
        if the_item in config:
            self.use_pretrained_model = self.str2bool(config[the_item])

        the_item = 'pretrained_model'
        if the_item in config:
            self.pretrained_model = config[the_item]

        the_item = 'use_middle_model'
        if the_item in config:
            self.use_middle_model = self.str2bool(config[the_item])

        the_item = 'middle_model'
        if the_item in config:
            self.middle_model = config[the_item]

        the_item = 'finetune'
        if the_item in config:
            self.finetune = self.str2bool(config[the_item])

        the_item = 'vocabs_dir'
        if the_item in config:
            self.vocabs_dir = config[the_item]

        the_item = 'checkpoints_dir'
        if the_item in config:
            self.checkpoints_dir = config[the_item]

        the_item = 'log_dir'
        if the_item in config:
            self.log_dir = config[the_item]

        # Labeling Scheme
        the_item = 'label_scheme'
        if the_item in config:
            self.label_scheme = config[the_item]

        the_item = 'label_level'
        if the_item in config:
            self.label_level = int(config[the_item])

        the_item = 'hyphen'
        if the_item in config:
            self.hyphen = config[the_item]

        the_item = 'suffix'
        if the_item in config:
            self.suffix = config[the_item]

        the_item = 'measuring_metrics'
        if the_item in config:
            self.measuring_metrics = config[the_item]

        the_item = 'embedding_dim'
        if the_item in config:
            self.embedding_dim = int(config[the_item])

        the_item = 'max_sequence_length'
        if the_item in config:
            self.max_sequence_length = int(config[the_item])

        the_item = 'hidden_dim'
        if the_item in config:
            self.hidden_dim = int(config[the_item])

        the_item = 'filter_nums'
        if the_item in config:
            self.filter_nums = int(config[the_item])

        the_item = 'idcnn_nums'
        if the_item in config:
            self.idcnn_nums = int(config[the_item])
            if self.idcnn_nums <= 0:
                raise Exception('idcnn numbers must over 0')

        the_item = 'CUDA_VISIBLE_DEVICES'
        if the_item in config:
            self.CUDA_VISIBLE_DEVICES = config[the_item]

        the_item = 'seed'
        if the_item in config:
            self.seed = int(config[the_item])

        # Training Settings:
        the_item = 'is_early_stop'
        if the_item in config:
            self.is_early_stop = self.str2bool(config[the_item])
        the_item = 'patient'
        if the_item in config:
            self.patient = int(config[the_item])

        the_item = 'use_gan'
        if the_item in config:
            self.use_gan = self.str2bool(config[the_item])
        the_item = 'gan_method'
        if the_item in config:
            self.gan_method = config[the_item]

        the_item = 'epoch'
        if the_item in config:
            self.epoch = int(config[the_item])
        the_item = 'batch_size'
        if the_item in config:
            self.batch_size = int(config[the_item])

        the_item = 'dropout'
        if the_item in config:
            self.dropout = float(config[the_item])
        the_item = 'learning_rate'
        if the_item in config:
            self.learning_rate = float(config[the_item])

        the_item = 'optimizer'
        if the_item in config:
            self.optimizer = config[the_item]

        the_item = 'checkpoint_name'
        if the_item in config:
            self.checkpoint_name = config[the_item]

        the_item = 'checkpoints_max_to_keep'
        if the_item in config:
            self.checkpoints_max_to_keep = int(config[the_item])

        the_item = 'print_per_batch'
        if the_item in config:
            self.print_per_batch = int(config[the_item])

    @staticmethod
    def config_file_to_dict(input_file):
        config = {}
        fins = open(input_file, 'r', encoding='utf-8').readlines()
        for line in fins:
            if len(line) > 0 and line[0] == '#':
                continue
            if '=' in line:
                pair = line.strip().split('#', 1)[0].split('=', 1)
                item = pair[0]
                value = pair[1]
                # noinspection PyBroadException
                try:
                    if item in config:
                        print('Warning: duplicated config item found: {}, updated.'.format((pair[0])))
                    if value[0] == '[' and value[-1] == ']':
                        value_items = list(value[1:-1].split(','))
                        config[item] = value_items
                    else:
                        config[item] = value
                except Exception:
                    print('configuration parsing error, please check correctness of the config file.')
                    exit(1)
        return config

    @staticmethod
    def str2bool(string):
        if string == 'True' or string == 'true' or string == 'TRUE':
            return True
        else:
            return False

    @staticmethod
    def str2none(string):
        if string == 'None' or string == 'none' or string == 'NONE':
            return None
        else:
            return string

    def show_data_summary(self, logger):
        logger.info('++' * 20 + 'CONFIGURATION SUMMARY' + '++' * 20)
        logger.info(' Status:')
        logger.info('     mode                 : {}'.format(self.mode))
        logger.info(' ' + '++' * 20)
        logger.info(' Datasets:')
        logger.info('     datasets         fold: {}'.format(self.datasets_fold))
        logger.info('     train            file: {}'.format(self.train_file))
        logger.info('     validation       file: {}'.format(self.dev_file))
        logger.info('     vocab             dir: {}'.format(self.vocabs_dir))
        logger.info('     delimiter            : {}'.format(self.delimiter))
        logger.info('     use  pretrained model: {}'.format(self.use_pretrained_model))
        logger.info('     pretrained      model: {}'.format(self.pretrained_model))
        logger.info('     finetune             : {}'.format(self.finetune))
        logger.info('     use    middle   model: {}'.format(self.use_middle_model))
        logger.info('     middle          model: {}'.format(self.middle_model))
        logger.info('     checkpoints       dir: {}'.format(self.checkpoints_dir))
        logger.info('     log               dir: {}'.format(self.log_dir))
        logger.info(' ' + '++' * 20)
        logger.info('Labeling Scheme:')
        logger.info('     label          scheme: {}'.format(self.label_scheme))
        logger.info('     label           level: {}'.format(self.label_level))
        logger.info('     suffixes             : {}'.format(self.suffix))
        logger.info('     measuring     metrics: {}'.format(self.measuring_metrics))
        logger.info(' ' + '++' * 20)
        logger.info('Model Configuration:')
        logger.info('     embedding         dim: {}'.format(self.embedding_dim))
        logger.info('     max  sequence  length: {}'.format(self.max_sequence_length))
        logger.info('     hidden            dim: {}'.format(self.hidden_dim))
        logger.info('     filter           nums: {}'.format(self.filter_nums))
        logger.info('     idcnn            nums: {}'.format(self.idcnn_nums))
        logger.info('     CUDA  VISIBLE  DEVICE: {}'.format(self.CUDA_VISIBLE_DEVICES))
        logger.info('     seed                 : {}'.format(self.seed))
        logger.info(' ' + '++' * 20)
        logger.info(' Training Settings:')
        logger.info('     epoch                : {}'.format(self.epoch))
        logger.info('     batch            size: {}'.format(self.batch_size))
        logger.info('     dropout              : {}'.format(self.dropout))
        logger.info('     learning         rate: {}'.format(self.learning_rate))
        logger.info('     optimizer            : {}'.format(self.optimizer))
        logger.info('     use               gan: {}'.format(self.use_gan))
        logger.info('     gan            method: {}'.format(self.gan_method))
        logger.info('     checkpoint       name: {}'.format(self.checkpoint_name))
        logger.info('     max       checkpoints: {}'.format(self.checkpoints_max_to_keep))
        logger.info('     print       per_batch: {}'.format(self.print_per_batch))
        logger.info('     is     early     stop: {}'.format(self.is_early_stop))
        logger.info('     patient              : {}'.format(self.patient))
        logger.info('++' * 20 + 'CONFIGURATION SUMMARY END' + '++' * 20)
        sys.stdout.flush()
