# -*- coding: utf-8 -*-
# @Time : 2020/9/13 3:18 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : predict.py
# @Software: PyCharm
import tensorflow as tf
import time
import math
from engines.model import NerModel
from engines.utils.extract_entity import extract_entity
from tensorflow_addons.text.crf import crf_decode
from tqdm import tqdm
from engines.utils.metrics import metrics


class Predictor:
    def __init__(self, configs, data_manager, logger):
        self.dataManager = data_manager
        vocab_size = data_manager.max_token_number
        num_classes = data_manager.max_label_number
        self.logger = logger
        self.configs = configs
        logger.info('loading model parameter')

        if configs.use_pretrained_model and not configs.finetune:
            if configs.pretrained_model == 'Bert':
                from transformers import TFBertModel
                self.pretrained_model = TFBertModel.from_pretrained('bert-base-chinese')
            elif configs.pretrained_model == 'AlBert':
                from transformers import TFAlbertModel
                self.pretrained_model = TFAlbertModel.from_pretrained('uer/albert-base-chinese-cluecorpussmall')

        self.ner_model = NerModel(configs, vocab_size, num_classes)
        # 实例化Checkpoint，设置恢复对象为新建立的模型
        if configs.finetune:
            checkpoint = tf.train.Checkpoint(ner_model=self.ner_model)
        else:
            checkpoint = tf.train.Checkpoint(ner_model=self.ner_model)
        checkpoint.restore(tf.train.latest_checkpoint(configs.checkpoints_dir))  # 从文件恢复模型参数
        logger.info('loading model successfully')

    def predict_one(self, sentence):
        """
        对输入的句子进行ner识别，取batch中的第一行结果
        :param sentence:
        :return:
        """
        if self.configs.use_pretrained_model:
            X, y, att_mask, Sentence = self.dataManager.prepare_single_sentence(sentence)
            if self.configs.finetune:
                model_inputs = (X, att_mask)
            else:
                model_inputs = self.pretrained_model(X, attention_mask=att_mask)[0]
        else:
            X, y, Sentence = self.dataManager.prepare_single_sentence(sentence)
            model_inputs = X
        inputs_length = tf.math.count_nonzero(X, 1)
        logits, _, transition_params = self.ner_model(
            inputs=model_inputs, inputs_length=inputs_length, targets=y)
        label_predicts, _ = crf_decode(logits, transition_params, inputs_length)
        label_predicts = label_predicts.numpy()
        sentence = Sentence[0, 0:inputs_length[0]]
        y_pred = [str(self.dataManager.id2label[val]) for val in label_predicts[0][0:inputs_length[0]]]
        if self.configs.use_pretrained_model:
            # 去掉[CLS]和[SEP]对应的位置
            y_pred = y_pred[1:-1]
        entities, suffixes, indices = extract_entity(sentence, y_pred, self.dataManager)
        return entities, suffixes, indices

    def predict_test(self):
        loss_values = []
        test_results = {}
        test_labels_results = {}
        for label in self.dataManager.suffix:
            test_labels_results.setdefault(label, {})
        for measure in self.configs.measuring_metrics:
            test_results[measure] = 0
        for label, content in test_labels_results.items():
            for measure in self.configs.measuring_metrics:
                if measure != 'accuracy':
                    test_labels_results[label][measure] = 0
        test_dataset = self.dataManager.get_test_dataset()
        start_time = time.time()
        num_test_iterations = int(math.ceil(1.0 * len(test_dataset) / self.dataManager.batch_size))
        for test_batch in tqdm(test_dataset.batch(self.dataManager.batch_size)):
            if self.configs.use_pretrained_model:
                X_test_batch, y_test_batch, att_mask_batch = test_batch
                if self.configs.finetune:
                    model_inputs = (X_test_batch, att_mask_batch)
                else:
                    model_inputs = self.pretrained_model(X_test_batch, attention_mask=att_mask_batch)[0]
            else:
                X_test_batch, y_test_batch = test_batch
                model_inputs = X_test_batch
            inputs_length_test = tf.math.count_nonzero(X_test_batch, 1)
            logits_test, log_likelihood_test, transition_params_test = self.ner_model(
                inputs=model_inputs, inputs_length=inputs_length_test, targets=y_test_batch)
            test_loss = -tf.reduce_mean(log_likelihood_test)
            batch_pred_sequence_val, _ = crf_decode(logits_test, transition_params_test, inputs_length_test)
            measures, lab_measures = metrics(
                X_test_batch, y_test_batch, batch_pred_sequence_val, self.configs, self.dataManager)

            for k, v in measures.items():
                test_results[k] += v
            for lab in lab_measures:
                for k, v in lab_measures[lab].items():
                    test_labels_results[lab][k] += v
            loss_values.append(test_loss)

        time_span = (time.time() - start_time) / 60
        test_res_str = ''
        for k, v in test_results.items():
            test_results[k] /= num_test_iterations
            test_res_str += (k + ': %.3f ' % test_results[k])
        for label, content in test_labels_results.items():
            test_label_str = ''
            for k, v in content.items():
                test_labels_results[label][k] /= num_test_iterations
                test_label_str += (k + ': %.3f ' % test_labels_results[label][k])
            self.logger.info('label: %s, %s' % (label, test_label_str))
        self.logger.info('time consumption:%.2f(min), %s' % (time_span, test_res_str))

    def save_pb(self):
        if self.configs.use_pretrained_model:
            if self.configs.finetune:
                tf.saved_model.save(
                    self.ner_model, self.configs.checkpoints_dir, signatures=self.ner_model.call.get_concrete_function(
                        (tf.TensorSpec([None, self.configs.max_sequence_length], tf.int32, name='token_inputs'),
                         tf.TensorSpec([None, self.configs.max_sequence_length], tf.int32, name='mask_inputs')),
                        tf.TensorSpec([None], tf.int32, name='inputs_length'),
                        tf.TensorSpec([None, self.configs.max_sequence_length], tf.int32, name='targets')))
            else:
                tf.saved_model.save(
                    self.ner_model, self.configs.checkpoints_dir, signatures=self.ner_model.call.get_concrete_function(
                        tf.TensorSpec([None, self.configs.max_sequence_length, 768], tf.float32, name='token_inputs'),
                        tf.TensorSpec([None], tf.int32, name='inputs_length'),
                        tf.TensorSpec([None, self.configs.max_sequence_length], tf.int32, name='targets')))
        else:
            tf.saved_model.save(
                self.ner_model, self.configs.checkpoints_dir, signatures=self.ner_model.call.get_concrete_function(
                    tf.TensorSpec([None, self.configs.max_sequence_length], tf.int32, name='token_inputs'),
                    tf.TensorSpec([None], tf.int32, name='inputs_length'),
                    tf.TensorSpec([None, self.configs.max_sequence_length], tf.int32, name='targets')))
        self.logger.info('The model has been saved')
