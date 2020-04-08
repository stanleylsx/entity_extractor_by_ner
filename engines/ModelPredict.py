import tensorflow as tf
import numpy as np
import pandas as pd
import math
from engines.Model import Model
from engines.utils.ExtractEntity import extract_entity
from engines.utils.IOFunctions import save_csv


class Predictor:
    def __init__(self, configs, logger, data_manager):
        self.graph = tf.Graph()
        self.dataManager = data_manager
        self.configs = configs
        self.output_test_file = configs.datasets_fold + '/' + configs.output_test_file
        self.is_output_sentence_entity = configs.is_output_sentence_entity
        self.output_sentence_entity_file = configs.datasets_fold + '/' + configs.output_sentence_entity_file
        self.logger = logger
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True))
        logger.info('loading model parameter')
        with self.sess.as_default():
            with self.graph.as_default():
                self.model = Model(configs, data_manager)
                tf.initialize_all_variables().run(session=self.sess)
                saver = tf.train.Saver()
                saver.restore(self.sess, tf.train.latest_checkpoint(configs.checkpoints_dir))
        logger.info('loading model successfully')

    def predict(self, sentence):
        """
        对输入的句子进行ner识别
        :param sentence:
        :return:
        """
        X, Sentence, Y = self.dataManager.prepare_single_sentence(sentence)
        predicts_labels, token, entities, entity_labels, labeled_indices = self.predict_batch(X, Y, Sentence)
        return entities, entity_labels, labeled_indices

    def predict_batch(self, X, y_prepare_label, X_test_token_batch):
        """
        将待预测的句子放到模型中预测
        :param X:
        :param y_prepare_label:
        :param X_test_token_batch:
        :return:
        """
        predicts_label_id, lengths = self.sess.run([self.model.batch_pred_sequence, self.model.length],
                                                   feed_dict={self.model.inputs: X,
                                                              self.model.targets: y_prepare_label})
        sentence_length = lengths[0]
        token = [val for val in X_test_token_batch[0][0:sentence_length]]
        predicts_labels = [str(self.dataManager.id2label[val]) for val in predicts_label_id[0][0:sentence_length]]
        entities, entity_labels, labeled_indices = extract_entity(token, predicts_labels, self.dataManager)
        return predicts_labels, token, entities, entity_labels, labeled_indices
