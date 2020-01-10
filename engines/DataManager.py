import logging
import os
import jieba
import re
import numpy as np
from engines.utils.IOFunctions import read_csv

jieba.setLogLevel(logging.INFO)


class DataManager:
    def __init__(self, configs, logger):
        self.configs = configs
        self.train_file = configs.train_file
        self.logger = logger
        self.hyphen = configs.hyphen

        self.UNKNOWN = '<UNK>'
        self.PADDING = '<PAD>'

        self.train_file = configs.datasets_fold + '/' + configs.train_file
        self.dev_file = configs.datasets_fold + '/' + configs.dev_file
        self.test_file = configs.datasets_fold + '/' + configs.test_file

        self.output_test_file = configs.datasets_fold + '/' + configs.output_test_file
        self.is_output_sentence_entity = configs.is_output_sentence_entity
        self.output_sentence_entity_file = configs.datasets_fold + '/' + configs.output_sentence_entity_file

        self.label_scheme = configs.label_scheme
        self.label_level = configs.label_level
        self.suffix = configs.suffix
        self.labeling_level = configs.labeling_level

        self.batch_size = configs.batch_size
        self.max_sequence_length = configs.max_sequence_length
        self.embedding_dim = configs.embedding_dim

        self.vocabs_dir = configs.vocabs_dir
        self.token2id_file = self.vocabs_dir + '/token2id'
        self.label2id_file = self.vocabs_dir + '/label2id'

        self.token2id, self.id2token, self.label2id, self.id2label = self.load_vocab()

        self.max_token_number = len(self.token2id)
        self.max_label_number = len(self.label2id)

        jieba.load_userdict(self.token2id.keys())

        self.logger.info('dataManager initialed...')

    def load_vocab(self):
        if not os.path.isfile(self.token2id_file):
            self.logger.info('vocab files not exist, building vocab...')
            return self.build_vocab(self.train_file)

        self.logger.info('loading vocab...')
        token2id = {}
        id2token = {}
        with open(self.token2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.rstrip()
                token = row.split('\t')[0]
                token_id = int(row.split('\t')[1])
                token2id[token] = token_id
                id2token[token_id] = token

        label2id = {}
        id2label = {}
        with open(self.label2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.rstrip()
                label = row.split('\t')[0]
                label_id = int(row.split('\t')[1])
                label2id[label] = label_id
                id2label[label_id] = label

        return token2id, id2token, label2id, id2label

    def build_vocab(self, train_path):
        df_train = read_csv(train_path, names=['token', 'label'], delimiter=self.configs.delimiter)
        tokens = list(set(df_train['token'][df_train['token'].notnull()]))
        labels = list(set(df_train['label'][df_train['label'].notnull()]))
        token2id = dict(zip(tokens, range(1, len(tokens) + 1)))
        label2id = dict(zip(labels, range(1, len(labels) + 1)))
        id2token = dict(zip(range(1, len(tokens) + 1), tokens))
        id2label = dict(zip(range(1, len(labels) + 1), labels))
        id2token[0] = self.PADDING
        id2label[0] = self.PADDING
        token2id[self.PADDING] = 0
        label2id[self.PADDING] = 0
        id2token[len(tokens) + 1] = self.UNKNOWN
        token2id[self.UNKNOWN] = len(tokens) + 1

        self.save_vocab(id2token, id2label)

        return token2id, id2token, label2id, id2label

    def save_vocab(self, id2token, id2label):
        with open(self.token2id_file, 'w', encoding='utf-8') as outfile:
            for idx in id2token:
                outfile.write(id2token[idx] + '\t' + str(idx) + '\n')
        with open(self.label2id_file, 'w', encoding='utf-8') as outfile:
            for idx in id2label:
                outfile.write(id2label[idx] + '\t' + str(idx) + '\n')

    def get_embedding(self, embed_file):
        emb_matrix = np.random.normal(loc=0.0, scale=0.08, size=(len(self.token2id.keys()), self.embedding_dim))
        emb_matrix[self.token2id[self.PADDING], :] = np.zeros(shape=self.embedding_dim)

        with open(embed_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.rstrip()
                items = row.split()
                token = items[0]
                assert self.embedding_dim == len(
                    items[1:]), 'embedding dim must be consistent with the one in `token_emb_dir`.'
                emb_vec = np.array([float(val) for val in items[1:]])
                if token in self.token2id.keys():
                    emb_matrix[self.token2id[token], :] = emb_vec

        return emb_matrix

    def next_batch(self, X, y, start_index):
        last_index = start_index + self.batch_size
        X_batch = list(X[start_index:min(last_index, len(X))])
        y_batch = list(y[start_index:min(last_index, len(X))])
        if last_index > len(X):
            left_size = last_index - (len(X))
            for i in range(left_size):
                index = np.random.randint(len(X))
                X_batch.append(X[index])
                y_batch.append(y[index])
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        return X_batch, y_batch

    def next_random_batch(self, X, y):
        X_batch = []
        y_batch = []
        for i in range(self.batch_size):
            index = np.random.randint(len(X))
            X_batch.append(X[index])
            y_batch.append(y[index])
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        return X_batch, y_batch

    def padding(self, sample):
        for i in range(len(sample)):
            if len(sample[i]) < self.max_sequence_length:
                sample[i] += [self.token2id[self.PADDING] for _ in range(self.max_sequence_length - len(sample[i]))]
        return sample

    def prepare(self, tokens, labels, is_padding=True, return_psyduo_label=False):
        X = []
        y = []
        y_psyduo = []
        tmp_x = []
        tmp_y = []
        tmp_y_psyduo = []

        for record in zip(tokens, labels):
            c = record[0]
            l = record[1]
            if c == -1:  # empty line
                if len(tmp_x) <= self.max_sequence_length:
                    X.append(tmp_x)
                    y.append(tmp_y)
                    if return_psyduo_label:
                        y_psyduo.append(tmp_y_psyduo)
                tmp_x = []
                tmp_y = []
                if return_psyduo_label:
                    tmp_y_psyduo = []
            else:
                tmp_x.append(c)
                tmp_y.append(l)
                if return_psyduo_label:
                    tmp_y_psyduo.append(self.label2id['O'])
        if is_padding:
            X = np.array(self.padding(X))
        else:
            X = np.array(X)
        y = np.array(self.padding(y))
        if return_psyduo_label:
            y_psyduo = np.array(self.padding(y_psyduo))
            return X, y_psyduo
        return X, y

    def get_training_set(self, train_val_ratio=0.9):
        df_train = read_csv(self.train_file, names=['token', 'label'], delimiter=self.configs.delimiter)
        # map the token and label into id
        df_train['token_id'] = df_train.token.map(lambda x: -1 if str(x) == str(np.nan) else self.token2id[x])
        df_train['label_id'] = df_train.label.map(lambda x: -1 if str(x) == str(np.nan) else self.label2id[x])
        # convert the data in matrix
        X, y = self.prepare(df_train['token_id'], df_train['label_id'])
        # shuffle the samples
        num_samples = len(X)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        if self.dev_file is not None:
            X_train = X
            y_train = y
            X_val, y_val = self.get_valid_set()
        else:
            # split the data into train and validation set
            X_train = X[:int(num_samples * train_val_ratio)]
            y_train = y[:int(num_samples * train_val_ratio)]
            X_val = X[int(num_samples * train_val_ratio):]
            y_val = y[int(num_samples * train_val_ratio):]
        self.logger.info('training set size: {}, validating set size: {}'.format(len(X_train), len(y_val)))
        return X_train, y_train, X_val, y_val

    def get_valid_set(self):
        df_val = read_csv(self.dev_file, names=['token', 'label'], delimiter=self.configs.delimiter)
        df_val['token_id'] = df_val.token.map(lambda x: self.map_func(x, self.token2id))
        df_val['label_id'] = df_val.label.map(lambda x: -1 if str(x) == str(np.nan) else self.label2id[x])
        X_val, y_val = self.prepare(df_val['token_id'], df_val['label_id'])
        return X_val, y_val

    def get_testing_set(self):
        df_test = read_csv(self.test_file, names=None, delimiter=self.configs.delimiter)

        if len(list(df_test.columns)) == 2:
            df_test.columns = ['token', 'label']
            df_test = df_test[['token']]
        elif len(list(df_test.columns)) == 1:
            df_test.columns = ['token']

        df_test['token_id'] = df_test.token.map(lambda x: self.map_func(x, self.token2id))
        df_test['token'] = df_test.token.map(lambda x: -1 if str(x) == str(np.nan) else x)
        X_test_id, y_test_psyduo_label = self.prepare(df_test['token_id'], df_test['token_id'], return_psyduo_label=True)
        X_test_token, _ = self.prepare(df_test['token'], df_test['token'])
        self.logger.info('testing set size: {}'.format(len(X_test_id)))
        return X_test_id, y_test_psyduo_label, X_test_token

    def map_func(self, x, token2id):
        if str(x) == str(np.nan):
            return -1
        elif x not in token2id:
            return token2id[self.UNKNOWN]
        else:
            return token2id[x]

    def prepare_single_sentence(self, sentence):
        if self.labeling_level == 'word':
            if self.check_contain_chinese(sentence):
                sentence = list(jieba.cut(sentence))
            else:
                sentence = list(sentence.split())
        elif self.labeling_level == 'char':
            sentence = list(sentence)

        gap = self.batch_size - 1

        x = []
        y = []

        for token in sentence:
            # noinspection PyBroadException
            try:
                x.append(self.token2id[token])
            except Exception:
                x.append(self.token2id[self.UNKNOWN])
            y.append(self.label2id['O'])

        if len(x) < self.max_sequence_length:
            sentence += ['x' for _ in range(self.max_sequence_length - len(sentence))]
            x += [self.token2id[self.PADDING] for _ in range(self.max_sequence_length - len(x))]
            y += [self.label2id['O'] for _ in range(self.max_sequence_length - len(y))]
        elif len(x) > self.max_sequence_length:
            sentence = sentence[:self.max_sequence_length]
            x = x[:self.max_sequence_length]
            y = y[:self.max_sequence_length]

        X = [x]
        Sentence = [sentence]
        Y = [y]
        X += [[0 for j in range(self.max_sequence_length)] for i in range(gap)]
        Sentence += [['x' for j in range(self.max_sequence_length)] for i in range(gap)]
        Y += [[self.label2id['O'] for j in range(self.max_sequence_length)] for i in range(gap)]
        X = np.array(X)
        Sentence = np.array(Sentence)
        Y = np.array(Y)

        return X, Sentence, Y

    @staticmethod
    def check_contain_chinese(check_str):
        return True if re.search(r'[\u4e00-\u9fff]', check_str) else False
