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
        X, Sentence, Y = self.dataManager.prepare_single_sentence(sentence)
        _, tokens, entities, predicts_labels_entity_level, indices = self.predict_batch(X, Y, Sentence)
        return entities[0], predicts_labels_entity_level[0], indices[0]

    def predict_batch(self, X, y_psydo_label, X_test_str_batch):
        entity_list = []
        tokens = []
        predicts_labels_entity_level = []
        indices = []
        predicts_labels_token_level = []
        predicts_label_id, lengths = self.sess.run([self.model.batch_pred_sequence, self.model.length],
                                                   feed_dict={self.model.inputs: X,
                                                              self.model.targets: y_psydo_label, })
        for i in range(len(lengths)):
            x_ = [val for val in X_test_str_batch[i, 0:lengths[i]]]
            tokens.append(x_)
            y_pred = [str(self.dataManager.id2label[val]) for val in predicts_label_id[i, 0:lengths[i]]]
            predicts_labels_token_level.append(y_pred)
            entities, entity_labels, labeled_indices = extract_entity(x_, y_pred, self.dataManager)
            entity_list.append(entities)
            predicts_labels_entity_level.append(entity_labels)
            indices.append(labeled_indices)
        return predicts_labels_token_level, tokens, entity_list, predicts_labels_entity_level, indices

    def batch_predict(self):
        X_test, y_test_psyduo_label, X_test_str = self.dataManager.get_testing_set()
        num_iterations = int(math.ceil(1.0 * len(X_test) / self.configs.batch_size))
        self.logger.info('total number of testing iterations: ' + str(num_iterations))
        tokens = []
        labels = []
        entities = []
        entities_types = []
        self.logger.info(('+' * 20) + 'testing starting' + ('+' * 20))
        for i in range(num_iterations):
            self.logger.info('batch: ' + str(i + 1))
            X_test_batch = X_test[i * self.configs.batch_size: (i + 1) * self.configs.batch_size]
            X_test_str_batch = X_test_str[i * self.configs.batch_size: (i + 1) * self.configs.batch_size]
            y_test_psyduo_label_batch = y_test_psyduo_label[i * self.configs.batch_size: (i + 1) * self.configs.batch_size]

            if i == num_iterations - 1 and len(X_test_batch) < self.configs.batch_size:
                X_test_batch = list(X_test_batch)
                X_test_str_batch = list(X_test_str_batch)
                y_test_psyduo_label_batch = list(y_test_psyduo_label_batch)
                gap = self.configs.batch_size - len(X_test_batch)

                X_test_batch += [[0 for j in range(self.configs.max_sequence_length)] for i in range(gap)]
                X_test_str_batch += [['x' for j in range(self.configs.max_sequence_length)] for i in range(gap)]
                y_test_psyduo_label_batch += [[self.dataManager.label2id['O'] for j in range(self.configs.max_sequence_length)] for i
                                              in range(gap)]
                X_test_batch = np.array(X_test_batch)
                X_test_str_batch = np.array(X_test_str_batch)
                y_test_psyduo_label_batch = np.array(y_test_psyduo_label_batch)
                results, token, entity, entities_type, _ = self.predict_batch(X_test_batch,
                                                                              y_test_psyduo_label_batch,
                                                                              X_test_str_batch)
                results = results[:len(X_test_batch)]
                token = token[:len(X_test_batch)]
                entity = entity[:len(X_test_batch)]
                entities_type = entities_type[:len(X_test_batch)]
            else:
                results, token, entity, entities_type, _ = self.predict_batch(X_test_batch,
                                                                              y_test_psyduo_label_batch,
                                                                              X_test_str_batch)
            labels.extend(results)
            tokens.extend(token)
            entities.extend(entity)
            entities_types.extend(entities_type)

        def save_test_out(tokens, labels):
            # transform format
            new_tokens, new_labels = [], []
            for to, la in zip(tokens, labels):
                new_tokens.extend(to)
                new_tokens.append('')
                new_labels.extend(la)
                new_labels.append('')
            # save results
            save_csv(pd.DataFrame({'token': new_tokens, 'label': new_labels}), self.output_test_file,
                     ['token', 'label'], delimiter=self.configs.delimiter)
        save_test_out(tokens, labels)
        self.logger.info('testing results saved.')
        if self.is_output_sentence_entity:
            with open(self.output_sentence_entity_file, 'w', encoding='utf-8') as outfile:
                for i in range(len(entities)):
                    if self.configs.label_level == 1:
                        outfile.write(' '.join(tokens[i]) + '\n' + '\n'.join(entities[i]) + '\n\n')
                    elif self.configs.label_level == 2:
                        outfile.write(' '.join(tokens[i]) + '\n' + '\n'.join(
                            [a + '\t({})'.format(b) for a, b in zip(entities[i], entities_types[i])]) + '\n\n')
            self.logger.info('testing results with sentences&entities saved.')
        self.sess.close()
