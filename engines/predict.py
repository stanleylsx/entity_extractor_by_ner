# -*- coding: utf-8 -*-
# @Time : 2020/9/13 3:18 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : predict.py
# @Software: PyCharm
import tensorflow as tf
from engines.model import NerModel
from engines.utils.extract_entity import extract_entity
from tensorflow_addons.text.crf import crf_decode
from transformers import TFBertModel


class Predictor:
    def __init__(self, configs, data_manager, logger):
        self.dataManager = data_manager
        vocab_size = data_manager.max_token_number
        num_classes = data_manager.max_label_number
        self.logger = logger
        self.configs = configs
        logger.info('loading model parameter')
        if self.configs.use_bert and not self.configs.finetune:
            self.bert_model = TFBertModel.from_pretrained('bert-base-chinese')
        self.ner_model = NerModel(configs, vocab_size, num_classes)
        # 实例化Checkpoint，设置恢复对象为新建立的模型
        checkpoint = tf.train.Checkpoint(ner_model=self.ner_model)
        checkpoint.restore(tf.train.latest_checkpoint(configs.checkpoints_dir))  # 从文件恢复模型参数
        logger.info('loading model successfully')

    def predict_one(self, sentence):
        """
        对输入的句子进行ner识别，取batch中的第一行结果
        :param sentence:
        :return:
        """
        if self.configs.use_bert:
            X, y, att_mask, Sentence = self.dataManager.prepare_single_sentence(sentence)
            if self.configs.finetune:
                model_inputs = (X, att_mask)
            else:
                model_inputs = self.bert_model(X, attention_mask=att_mask)[0]
        else:
            X, y, Sentence = self.dataManager.prepare_single_sentence(sentence)
            model_inputs = X
        inputs_length = tf.math.count_nonzero(X, 1)
        logits, log_likelihood, transition_params = self.ner_model(
                inputs=model_inputs, inputs_length=inputs_length, targets=y)
        label_predicts, _ = crf_decode(logits, transition_params, inputs_length)
        label_predicts = label_predicts.numpy()
        sentence = Sentence[0, 0:inputs_length[0]]
        y_pred = [str(self.dataManager.id2label[val]) for val in label_predicts[0][0:inputs_length[0]]]
        if self.configs.use_bert:
            # 去掉[CLS]和[SEP]对应的位置
            y_pred = y_pred[1:-1]
        entities, suffixes, indices = extract_entity(sentence, y_pred, self.dataManager)
        return entities, suffixes, indices
