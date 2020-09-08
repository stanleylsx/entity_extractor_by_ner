import tensorflow as tf
import math
import pandas as pd
import numpy as np
from engines.model import BiLSTM_CRFModel
from engines.utils.extract_entity import extract_entity
from engines.utils.io_functions import save_csv
from tensorflow_addons.text.crf import crf_decode
from tqdm import tqdm


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

    def predict(self, sentence):
        """
        对输入的句子进行ner识别，取batch中的第一行结果
        :param sentence:
        :return:
        """
        X, Sentence, Y = self.dataManager.prepare_single_sentence(sentence)
        _, _, entities_predicts_batch, suffixes_predicts_batch, indices_predicts_batch = self.predict_batch(X, Y,
                                                                                                            Sentence)
        return entities_predicts_batch[0], suffixes_predicts_batch[0], indices_predicts_batch[0]

    def predict_batch(self, X, y_prepare_label, X_test_token_batch):
        """
        将待预测的句子batch放到模型中预测
        :param X:
        :param y_prepare_label:
        :param X_test_token_batch:
        :return:
        """
        entities_predicts_batch = []
        tokens_batch = []
        suffixes_predicts_batch = []
        indices_predicts_batch = []
        labels_predicts_batch = []
        logits, lengths_batch, log_likelihood, transition_params = self.bilstm_crf_model.call(
            inputs=X, targets=y_prepare_label)
        label_id_predicts_batch, _ = crf_decode(logits, transition_params, lengths_batch)
        label_id_predicts_batch = label_id_predicts_batch.numpy()
        for i in range(len(lengths_batch)):
            x = [val for val in X_test_token_batch[i, 0:lengths_batch[i]]]
            tokens_batch.append(x)
            y_pred = [str(self.dataManager.id2label[val]) for val in label_id_predicts_batch[i, 0:lengths_batch[i]]]
            labels_predicts_batch.append(y_pred)
            entities, suffixes, indices = extract_entity(x, y_pred, self.dataManager)
            entities_predicts_batch.append(entities)
            suffixes_predicts_batch.append(suffixes)
            indices_predicts_batch.append(indices)
        return labels_predicts_batch, tokens_batch, entities_predicts_batch, suffixes_predicts_batch, indices_predicts_batch

    def save_test_out(self, tokens, labels):
        # transform format
        new_tokens, new_labels = [], []
        for to, la in zip(tokens, labels):
            new_tokens.extend(to)
            new_tokens.append('')
            new_labels.extend(la)
            new_labels.append('')
        # save labels_predicts_batch
        save_csv(pd.DataFrame({'tokens_batch': new_tokens, 'label': new_labels}), self.output_test_file,
                 ['tokens_batch', 'label'], delimiter=self.configs.delimiter)

    def test_batch_predict(self):
        """
        对测试集进行批量的预测
        :return:
        """
        X_test, y_test_prepare_label, X_test_token = self.dataManager.get_testing_set()
        num_iterations = int(math.ceil(1.0 * len(X_test) / self.configs.batch_size))
        self.logger.info('total number of testing iterations: ' + str(num_iterations))
        tokens = []
        labels = []
        entities = []
        suffixes = []
        self.logger.info(('+' * 20) + 'testing starting' + ('+' * 20))
        for i in tqdm(range(num_iterations)):
            self.logger.info('batch: ' + str(i + 1))
            X_test_batch = X_test[i * self.configs.batch_size: (i + 1) * self.configs.batch_size]
            X_test_token_batch = X_test_token[i * self.configs.batch_size: (i + 1) * self.configs.batch_size]
            y_test_prepare_label_batch = y_test_prepare_label[
                                         i * self.configs.batch_size: (i + 1) * self.configs.batch_size]

            if i == num_iterations - 1 and len(X_test_batch) < self.configs.batch_size:
                X_test_batch = list(X_test_batch)
                X_test_token_batch = list(X_test_token_batch)
                y_test_prepare_label_batch = list(y_test_prepare_label_batch)
                gap = self.configs.batch_size - len(X_test_batch)

                X_test_batch += [[0 for _ in range(self.configs.max_sequence_length)] for _ in range(gap)]
                X_test_token_batch += [['x' for _ in range(self.configs.max_sequence_length)] for _ in range(gap)]
                y_test_prepare_label_batch += [
                    [self.dataManager.label2id['O'] for _ in range(self.configs.max_sequence_length)] for _
                    in range(gap)]
                X_test_batch = np.array(X_test_batch)
                X_test_token_batch = np.array(X_test_token_batch)
                y_test_prepare_label_batch = np.array(y_test_prepare_label_batch)
                labels_predicts_batch, tokens_batch, entities_predicts_batch, suffixes_predicts_batch, _ = self.predict_batch(
                    X_test_batch, y_test_prepare_label_batch, X_test_token_batch)
                labels_predicts_batch = labels_predicts_batch[:len(X_test_batch)]
                tokens_batch = tokens_batch[:len(X_test_batch)]
                entities_predicts_batch = entities_predicts_batch[:len(X_test_batch)]
                suffixes_predicts_batch = suffixes_predicts_batch[:len(X_test_batch)]
            else:
                labels_predicts_batch, tokens_batch, entities_predicts_batch, suffixes_predicts_batch, _ = self.predict_batch(
                    X_test_batch, y_test_prepare_label_batch, X_test_token_batch)
            labels.extend(labels_predicts_batch)
            tokens.extend(tokens_batch)
            entities.extend(entities_predicts_batch)
            suffixes.extend(suffixes_predicts_batch)
        self.save_test_out(tokens, labels)
        self.logger.info('testing labels_predicts_batch saved.')
        if self.is_output_sentence_entity:
            with open(self.output_sentence_entity_file, 'w', encoding='utf-8') as outfile:
                for i in range(len(entities)):
                    if self.configs.label_level == 1:
                        outfile.write(' '.join(tokens[i]) + '\n' + '\n'.join(entities[i]) + '\n\n')
                    elif self.configs.label_level == 2:
                        outfile.write(' '.join(tokens[i]) + '\n' + '\n'.join(
                            [a + '\t({})'.format(b) for a, b in zip(entities[i], suffixes[i])]) + '\n\n')
            self.logger.info('testing labels_predicts_batch with sentences&entities saved.')