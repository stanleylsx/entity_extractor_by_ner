# -*- coding: utf-8 -*-
# @Time : 2020/9/13 3:18 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : data.py
# @Software: PyCharm
import os
import numpy as np
import tensorflow as tf
from engines.utils.io_functions import read_csv
from tqdm import tqdm


class DataManager:
    """
    数据管理器
    """
    def __init__(self, configs, logger):
        self.configs = configs
        self.logger = logger
        self.hyphen = configs.hyphen

        self.UNKNOWN = '[UNK]'
        self.PADDING = '[PAD]'

        self.train_file = configs.datasets_fold + '/' + configs.train_file

        if configs.dev_file is not None:
            self.dev_file = configs.datasets_fold + '/' + configs.dev_file
        else:
            self.dev_file = None

        self.label_scheme = configs.label_scheme
        self.label_level = configs.label_level
        self.suffix = configs.suffix

        self.batch_size = configs.batch_size

        self.max_sequence_length = configs.max_sequence_length
        self.embedding_dim = configs.embedding_dim
        self.vocabs_dir = configs.vocabs_dir
        self.token2id_file = self.vocabs_dir + '/token2id'
        self.label2id_file = self.vocabs_dir + '/label2id'

        self.token2id, self.id2token, self.label2id, self.id2label = self.load_vocab()

        if configs.use_bert:
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            self.max_token_number = len(self.tokenizer.get_vocab())
        else:
            self.max_token_number = len(self.token2id)
        self.max_label_number = len(self.label2id)

        self.logger.info('dataManager initialed...')

    def load_vocab(self):
        """
        若不存在词表则生成，若已经存在则加载词表
        :return:
        """
        if not os.path.isfile(self.token2id_file):
            self.logger.info('label vocab files not exist, building label vocab...')
            return self.build_vocab(self.train_file)

        self.logger.info('loading vocab...')
        token2id, id2token = {}, {}
        # 不使用Bert做嵌入
        if not self.configs.use_bert:
            with open(self.token2id_file, 'r', encoding='utf-8') as infile:
                for row in infile:
                    row = row.strip()
                    token, token_id = row.split('\t')[0], int(row.split('\t')[1])
                    token2id[token] = token_id
                    id2token[token_id] = token

        label2id, id2label = {}, {}
        with open(self.label2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.strip()
                label, label_id = row.split('\t')[0], int(row.split('\t')[1])
                label2id[label] = label_id
                id2label[label_id] = label
        return token2id, id2token, label2id, id2label

    def build_vocab(self, train_path):
        """
        根据训练集生成词表
        :param train_path:
        :return:
        """
        df_train = read_csv(train_path, names=['token', 'label'], delimiter=self.configs.delimiter)
        token2id, id2token = {}, {}
        if not self.configs.use_bert:
            tokens = list(set(df_train['token'][df_train['token'].notnull()]))
            token2id = dict(zip(tokens, range(1, len(tokens) + 1)))
            id2token = dict(zip(range(1, len(tokens) + 1), tokens))
            id2token[0] = self.PADDING
            token2id[self.PADDING] = 0
            # 向生成的词表中加入[UNK]
            id2token[len(tokens) + 1] = self.UNKNOWN
            token2id[self.UNKNOWN] = len(tokens) + 1
            # 保存词表及标签表
            with open(self.token2id_file, 'w', encoding='utf-8') as outfile:
                for idx in id2token:
                    outfile.write(id2token[idx] + '\t' + str(idx) + '\n')

        labels = list(set(df_train['label'][df_train['label'].notnull()]))
        label2id = dict(zip(labels, range(1, len(labels) + 1)))
        id2label = dict(zip(range(1, len(labels) + 1), labels))
        id2label[0] = self.PADDING
        label2id[self.PADDING] = 0
        with open(self.label2id_file, 'w', encoding='utf-8') as outfile:
            for idx in id2label:
                outfile.write(id2label[idx] + '\t' + str(idx) + '\n')
        return token2id, id2token, label2id, id2label

    def padding(self, sample):
        """
        长度不足max_sequence_length则补齐
        :param sample:
        :return:
        """
        for i in range(len(sample)):
            if len(sample[i]) < self.max_sequence_length:
                sample[i] += [self.token2id[self.PADDING] for _ in range(self.max_sequence_length - len(sample[i]))]
        return sample

    def prepare(self, tokens, labels, is_padding=True):
        """
        输出X矩阵和y向量
        :param tokens:
        :param labels:
        :param is_padding:
        :return:
        """
        self.logger.info('loading data...')
        X = []
        y = []
        tmp_x = []
        tmp_y = []
        for record in tqdm(zip(tokens, labels)):
            token = record[0]
            label = record[1]
            if token == -1:  # empty line
                if len(tmp_x) <= self.max_sequence_length:
                    X.append(tmp_x)
                    y.append(tmp_y)
                else:
                    X.append(tmp_x[:self.max_sequence_length])
                    y.append(tmp_y[:self.max_sequence_length])
                tmp_x = []
                tmp_y = []
            else:
                tmp_x.append(token)
                tmp_y.append(label)
        if is_padding:
            X = np.array(self.padding(X))
        else:
            X = np.array(X)
        y = np.array(self.padding(y))
        return X, y

    def prepare_bert_embedding(self, df):
        """
        输出前接Bert做词嵌入时候X矩阵和y向量
        :param df:
        :return:
        """
        self.logger.info('loading data...')
        X = []
        y = []
        att_mask = []
        tmp_x = []
        tmp_y = []
        for index, record in tqdm(df.iterrows()):
            token = record.token
            label = record.label
            if str(token) == str(np.nan):
                if len(tmp_x) <= self.max_sequence_length - 2:
                    tmp_x = self.tokenizer.encode(tmp_x)
                    tmp_att_mask = [1] * len(tmp_x)
                    tmp_y = [self.label2id[y] for y in tmp_y]
                    tmp_y.insert(0, self.label2id['O'])
                    tmp_y.append(self.label2id['O'])
                    # padding
                    tmp_x += [0 for _ in range(self.max_sequence_length - len(tmp_x))]
                    tmp_y += [self.label2id[self.PADDING] for _ in range(self.max_sequence_length - len(tmp_y))]
                    tmp_att_mask += [0 for _ in range(self.max_sequence_length - len(tmp_att_mask))]
                    X.append(tmp_x)
                    y.append(tmp_y)
                    att_mask.append(tmp_att_mask)
                else:
                    # 此处的padding不能在self.max_sequence_length加2，否则不同维度情况下，numpy没办法转换成矩阵
                    tmp_x = tmp_x[:self.max_sequence_length-2]
                    tmp_x = self.tokenizer.encode(tmp_x)
                    X.append(tmp_x)
                    tmp_y = tmp_y[:self.max_sequence_length-2]
                    tmp_y = [self.label2id[y] for y in tmp_y]
                    tmp_y.insert(0, self.label2id['O'])
                    tmp_y.append(self.label2id['O'])
                    y.append(tmp_y)
                    tmp_att_mask = [1] * self.max_sequence_length
                    att_mask.append(tmp_att_mask)
                tmp_x = []
                tmp_y = []
            else:
                tmp_x.append(token)
                tmp_y.append(label)
        return np.array(X), np.array(y), np.array(att_mask)

    def get_training_set(self, train_val_ratio=0.9):
        """
        获取训练数据集、验证集
        :param train_val_ratio:
        :return:
        """
        df_train = read_csv(self.train_file, names=['token', 'label'], delimiter=self.configs.delimiter)
        if self.configs.use_bert:
            X, y, att_mask = self.prepare_bert_embedding(df_train)
            # shuffle the samples
            num_samples = len(X)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            att_mask = att_mask[indices]

            if self.dev_file is not None:
                X_train = X
                y_train = y
                att_mask_train = att_mask
                X_val, y_val, att_mask_val = self.get_valid_set()
            else:
                # split the data into train and validation set
                X_train = X[:int(num_samples * train_val_ratio)]
                y_train = y[:int(num_samples * train_val_ratio)]
                att_mask_train = att_mask[:int(num_samples * train_val_ratio)]
                X_val = X[int(num_samples * train_val_ratio):]
                y_val = y[int(num_samples * train_val_ratio):]
                att_mask_val = att_mask[int(num_samples * train_val_ratio):]
                self.logger.info('validating set is not exist, built...')
            self.logger.info('training set size: {}, validating set size: {}'.format(len(X_train), len(X_val)))
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, att_mask_train))
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val, att_mask_val))
        else:
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
                self.logger.info('validating set is not exist, built...')
            self.logger.info('training set size: {}, validating set size: {}'.format(len(X_train), len(X_val)))
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        return train_dataset, val_dataset

    def get_valid_set(self):
        """
        获取验证集
        :return:
        """
        df_val = read_csv(self.dev_file, names=['token', 'label'], delimiter=self.configs.delimiter)
        if self.configs.use_bert:
            X_val, y_val, att_mask_val = self.prepare_bert_embedding(df_val)
            return X_val, y_val, att_mask_val
        else:
            df_val['token_id'] = df_val.token.map(lambda x: self.map_func(x, self.token2id))
            df_val['label_id'] = df_val.label.map(lambda x: -1 if str(x) == str(np.nan) else self.label2id[x])
            X_val, y_val = self.prepare(df_val['token_id'], df_val['label_id'])
            return X_val, y_val

    def map_func(self, x, token2id):
        if str(x) == str(np.nan):
            return -1
        elif x not in token2id:
            return token2id[self.UNKNOWN]
        else:
            return token2id[x]

    def prepare_single_sentence(self, sentence):
        """
        把预测的句子转成矩阵和向量
        :param sentence:
        :return:
        """
        sentence = list(sentence)
        if self.configs.use_bert:
            if len(sentence) <= self.max_sequence_length - 2:
                x = self.tokenizer.encode(sentence)
                att_mask = [1] * len(x)
                x += [0 for _ in range(self.max_sequence_length - len(x))]
                att_mask += [0 for _ in range(self.max_sequence_length - len(att_mask))]
            else:
                sentence = sentence[:self.max_sequence_length-2]
                x = self.tokenizer.encode(sentence)
                att_mask = [1] * len(x)
            y = [self.label2id['O']] * self.max_sequence_length
            return np.array([x]), np.array([y]), np.array([att_mask]), np.array([sentence])
        else:
            x = []
            for token in sentence:
                # noinspection PyBroadException
                try:
                    x.append(self.token2id[token])
                except Exception:
                    x.append(self.token2id[self.UNKNOWN])

            if len(x) < self.max_sequence_length:
                sentence += ['[PAD]' for _ in range(self.max_sequence_length - len(sentence))]
                x += [self.token2id[self.PADDING] for _ in range(self.max_sequence_length - len(x))]
            elif len(x) > self.max_sequence_length:
                sentence = sentence[:self.max_sequence_length]
                x = x[:self.max_sequence_length]
            y = [self.label2id['O']] * self.max_sequence_length
            return np.array([x]), np.array([y]), np.array([sentence])
