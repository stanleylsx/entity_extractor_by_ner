# -*- coding: utf-8 -*-
# @Time : 2020/9/10 7:15 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : predict.py
# @Software: PyCharm
import tensorflow as tf
from engines.model import BiLSTM_CRFModel
from engines.utils.extract_entity import extract_entity
from tensorflow_addons.text.crf import crf_decode


class Predictor:
    def __init__(self, configs, data_manager, logger):
        self.dataManager = data_manager
        vocab_size = data_manager.max_token_number
        num_classes = data_manager.max_label_number
        self.configs = configs
        self.output_test_file = configs.datasets_fold + '/' + configs.output_test_file
        self.is_output_sentence_entity = configs.is_output_sentence_entity
        self.output_sentence_entity_file = configs.datasets_fold + '/' + configs.output_sentence_entity_file
        self.logger = logger
        logger.info('loading model parameter')
        self.bilstm_crf_model = BiLSTM_CRFModel(configs, vocab_size, num_classes)
        # 实例化Checkpoint，设置恢复对象为新建立的模型
        checkpoint = tf.train.Checkpoint(model=self.bilstm_crf_model)
        checkpoint.restore(tf.train.latest_checkpoint(configs.checkpoints_dir))  # 从文件恢复模型参数
        logger.info('loading model successfully')

    def predict_one(self, sentence):
        """
        对输入的句子进行ner识别，取batch中的第一行结果
        :param sentence:
        :return:
        """
        X, y, Sentence = self.dataManager.prepare_single_sentence(sentence)
        logits, input_length, log_likelihood, transition_params = self.bilstm_crf_model.call(
            inputs=X, targets=y)
        label_predicts, _ = crf_decode(logits, transition_params, input_length)
        label_predicts = label_predicts.numpy()
        sentence = Sentence[0, 0:input_length[0]]
        y_pred = [str(self.dataManager.id2label[val]) for val in label_predicts[0][0:input_length[0]]]
        entities, suffixes, indices = extract_entity(sentence, y_pred, self.dataManager)
        return entities, suffixes, indices
