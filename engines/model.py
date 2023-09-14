# -*- coding: utf-8 -*-
# @Time : 2020/9/9 6:14 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : model.py
# @Software: PyCharm
from abc import ABC

import tensorflow as tf
from tensorflow_addons.text.crf import crf_log_likelihood


class NerModel(tf.keras.Model, ABC):
    def __init__(self, configs, vocab_size, num_classes):
        super(NerModel, self).__init__()
        self.use_pretrained_model = configs.use_pretrained_model
        self.finetune = configs.finetune

        if self.use_pretrained_model and self.finetune:
            if configs.pretrained_model == 'Bert':
                from transformers import TFBertModel
                self.pretrained_model = TFBertModel.from_pretrained('bert-base-chinese')
            elif configs.pretrained_model == 'AlBert':
                from transformers import TFAlbertModel
                self.pretrained_model = TFAlbertModel.from_pretrained('uer/albert-base-chinese-cluecorpussmall')
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size, configs.embedding_dim, mask_zero=True)

        self.use_middle_model = configs.use_middle_model
        self.middle_model = configs.middle_model
        if self.use_middle_model:
            if self.middle_model == 'bilstm':
                self.hidden_dim = configs.hidden_dim
                self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True))
            if self.middle_model == 'idcnn':
                filter_nums = configs.filter_nums
                self.idcnn_nums = configs.idcnn_nums
                self.cnn = tf.keras.Sequential([
                    tf.keras.layers.Conv1D(filters=filter_nums, kernel_size=3, activation='relu',
                                           padding='same', dilation_rate=1),
                    tf.keras.layers.Conv1D(filters=filter_nums, kernel_size=3, activation='relu',
                                           padding='same', dilation_rate=1),
                    tf.keras.layers.Conv1D(filters=filter_nums, kernel_size=3, activation='relu',
                                           padding='same', dilation_rate=2)])
                self.idcnn = [self.cnn for _ in range(self.idcnn_nums)]

        self.dropout_rate = configs.dropout
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense = tf.keras.layers.Dense(num_classes)
        self.transition_params = tf.Variable(tf.random.uniform(shape=(num_classes, num_classes)))

    @tf.function
    def call(self, inputs, inputs_length, targets, training=None):
        if self.use_pretrained_model:
            if self.finetune:
                embedding_inputs = self.pretrained_model(inputs[0], attention_mask=inputs[1])[0]
            else:
                embedding_inputs = inputs
        else:
            embedding_inputs = self.embedding(inputs)

        outputs = self.dropout(embedding_inputs, training)

        if self.use_middle_model:
            if self.middle_model == 'bilstm':
                outputs = self.bilstm(outputs)
            if self.middle_model == 'idcnn':
                cnn_outputs = [idcnn(outputs) for idcnn in self.idcnn]
                if self.idcnn_nums == 1:
                    outputs = cnn_outputs[0]
                else:
                    outputs = tf.keras.layers.concatenate(cnn_outputs, axis=-1, name='concatenate')

        logits = self.dense(outputs)
        tensor_targets = tf.convert_to_tensor(targets, dtype=targets.dtype)
        log_likelihood, self.transition_params = crf_log_likelihood(
            logits, tensor_targets, inputs_length, transition_params=self.transition_params)
        return logits, log_likelihood, self.transition_params
