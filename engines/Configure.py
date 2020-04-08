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
            self.dev_file = config[the_item]
        the_item = 'delimiter'
        if the_item in config:
            self.delimiter = config[the_item]

        the_item = 'use_pre_trained_embedding'
        if the_item in config:
            self.use_pre_trained_embedding = self.str2bool(config[the_item])
        the_item = 'token_emb_dir'
        if the_item in config:
            self.token_emb_dir = config[the_item]

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

        the_item = 'labeling_level'
        if the_item in config:
            self.labeling_level = config[the_item]

        the_item = 'measuring_metrics'
        if the_item in config:
            self.measuring_metrics = config[the_item]

        # ModelConfiguration
        the_item = 'cell_type'
        if the_item in config:
            self.cell_type = config[the_item]
        the_item = 'bidirectional'
        if the_item in config:
            self.bidirectional = self.str2bool(config[the_item])
        the_item = 'encoder_layers'
        if the_item in config:
            self.encoder_layers = int(config[the_item])

        the_item = 'embedding_dim'
        if the_item in config:
            self.embedding_dim = int(config[the_item])

        the_item = 'max_sequence_length'
        if the_item in config:
            self.max_sequence_length = int(config[the_item])

        the_item = 'hidden_dim'
        if the_item in config:
            self.hidden_dim = int(config[the_item])

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

        # Testing Settings
        the_item = 'output_test_file'
        if the_item in config:
            self.output_test_file = config[the_item]
        the_item = 'is_output_sentence_entity'
        if the_item in config:
            self.is_output_sentence_entity = self.str2bool(config[the_item])
        the_item = 'output_sentence_entity_file'
        if the_item in config:
            self.output_sentence_entity_file = config[the_item]

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

    def show_data_summary(self, logger):
        logger.info('++' * 20 + 'CONFIGURATION SUMMARY' + '++' * 20)
        logger.info(' Status:')
        logger.info('     mode                 : {}'.format(self.mode))
        logger.info(' ' + '++' * 20)
        logger.info(' Datasets:')
        logger.info('     datasets         fold: {}'.format(self.datasets_fold))
        logger.info('     train            file: {}'.format(self.train_file))
        logger.info('     validation       file: {}'.format(self.dev_file))
        logger.info('     pre-trained embedding: {}'.format(self.use_pre_trained_embedding))
        logger.info('     embedding        file: {}'.format(self.token_emb_dir))
        logger.info('     vocab             dir: {}'.format(self.vocabs_dir))
        logger.info('     delimiter            : {}'.format(self.delimiter))
        logger.info('     checkpoints       dir: {}'.format(self.checkpoints_dir))
        logger.info('     log               dir: {}'.format(self.log_dir))
        logger.info(' ' + '++' * 20)
        logger.info('Labeling Scheme:')
        logger.info('     label          scheme: {}'.format(self.label_scheme))
        logger.info('     label           level: {}'.format(self.label_level))
        logger.info('     suffixes             : {}'.format(self.suffix))
        logger.info('     labeling_level       : {}'.format(self.labeling_level))
        logger.info('     measuring     metrics: {}'.format(self.measuring_metrics))
        logger.info(' ' + '++' * 20)
        logger.info('Model Configuration:')
        logger.info('     cell             type: {}'.format(self.cell_type))
        logger.info('     bidirectional        : {}'.format(self.bidirectional))
        logger.info('     encoder        layers: {}'.format(self.encoder_layers))
        logger.info('     embedding         dim: {}'.format(self.embedding_dim))
        logger.info('     max  sequence  length: {}'.format(self.max_sequence_length))
        logger.info('     hidden            dim: {}'.format(self.hidden_dim))
        logger.info('     CUDA  VISIBLE  DEVICE: {}'.format(self.CUDA_VISIBLE_DEVICES))
        logger.info('     seed                 : {}'.format(self.seed))
        logger.info(' ' + '++' * 20)
        logger.info(' Training Settings:')
        logger.info('     epoch                : {}'.format(self.epoch))
        logger.info('     batch            size: {}'.format(self.batch_size))
        logger.info('     dropout              : {}'.format(self.dropout))
        logger.info('     learning         rate: {}'.format(self.learning_rate))
        logger.info('     optimizer            : {}'.format(self.optimizer))
        logger.info('     checkpoint       name: {}'.format(self.checkpoint_name))
        logger.info('     max       checkpoints: {}'.format(self.checkpoints_max_to_keep))
        logger.info('     print       per_batch: {}'.format(self.print_per_batch))
        logger.info('     is     early     stop: {}'.format(self.is_early_stop))
        logger.info('     patient              : {}'.format(self.patient))
        logger.info('++' * 20 + 'CONFIGURATION SUMMARY END' + '++' * 20)
        sys.stdout.flush()
