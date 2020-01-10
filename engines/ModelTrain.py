import tensorflow as tf
import numpy as np
import math
import time
from engines.Model import Model
from engines.utils.Metrics import metrics


class Train:
    def __init__(self, configs, logger, data_manager):
        self.graph = tf.Graph()
        self.dataManager = data_manager
        self.configs = configs
        self.logger = logger
        self.max_to_keep = configs.checkpoints_max_to_keep
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True))
        logger.info('loading model...')
        with self.sess.as_default():
            with self.graph.as_default():
                self.model = Model(configs, logger, data_manager)
                tf.initialize_all_variables().run(session=self.sess)
                self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
        logger.info('loading model successfully...')
        self.best_f1_val = 0
        self.inputs = self.model.inputs
        self.targets = self.model.targets

    def train(self):
        X_train, y_train, X_val, y_val = self.dataManager.get_training_set()
        tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.configs.log_dir + '/training_loss', self.sess.graph)
        dev_writer = tf.summary.FileWriter(self.configs.log_dir + '/validating_loss', self.sess.graph)
        num_iterations = int(math.ceil(1.0 * len(X_train) / self.configs.batch_size))
        num_val_iterations = int(math.ceil(1.0 * len(X_val) / self.configs.batch_size))

        cnt = 0
        cnt_dev = 0
        unprocessed = 0
        very_start_time = time.time()
        best_at_epoch = 0
        self.logger.info(('+' * 20) + 'training starting' + ('+' * 20))

        for epoch in range(self.configs.epoch):
            start_time = time.time()
            # shuffle train at each epoch
            sh_index = np.arange(len(X_train))
            np.random.shuffle(sh_index)
            X_train = X_train[sh_index]
            y_train = y_train[sh_index]

            self.logger.info('\ncurrent epoch: {}'.format(epoch))
            for iteration in range(num_iterations):
                X_train_batch, y_train_batch = self.dataManager.next_batch(X_train, y_train,
                                                                           start_index=iteration * self.configs.batch_size)
                _, loss_train, train_batch_viterbi_sequence, train_summary = self.sess.run([
                    self.model.opt_op,
                    self.model.loss,
                    self.model.batch_pred_sequence,
                    self.model.train_summary
                ],
                    feed_dict={
                        self.inputs: X_train_batch,
                        self.targets: y_train_batch,
                    })

                if iteration % self.configs.print_per_batch == 0:
                    cnt += 1
                    train_writer.add_summary(train_summary, cnt)
                    measures = metrics(X_train_batch, y_train_batch, train_batch_viterbi_sequence,
                                       self.configs.measuring_metrics, self.dataManager)
                    res_str = ''
                    for k, v in measures.items():
                        res_str += (k + ': %.3f ' % v)
                    self.logger.info('training batch: %5d, loss: %.5f, %s' % (iteration, loss_train, res_str))

            # validation
            loss_values = list()
            val_results = dict()
            for measure in self.configs.measuring_metrics:
                val_results[measure] = 0

            for iteration in range(num_val_iterations):
                cnt_dev += 1
                X_val_batch, y_val_batch = self.dataManager.next_batch(X_val, y_val,
                                                                       start_index=iteration * self.configs.batch_size)
                loss_val, val_batch_viterbi_sequence, dev_summary = self.sess.run([
                    self.model.loss,
                    self.model.batch_pred_sequence,
                    self.model.dev_summary
                ],
                    feed_dict={
                        self.inputs: X_val_batch,
                        self.targets: y_val_batch,
                    })

                measures = metrics(X_val_batch, y_val_batch, val_batch_viterbi_sequence,
                                   self.configs.measuring_metrics, self.dataManager)
                dev_writer.add_summary(dev_summary, cnt_dev)

                for k, v in measures.items():
                    val_results[k] += v
                loss_values.append(loss_val)

            time_span = (time.time() - start_time) / 60
            val_res_str = ''
            dev_f1_avg = 0
            for k, v in val_results.items():
                val_results[k] /= num_val_iterations
                val_res_str += (k + ': %.3f ' % val_results[k])
                if k == 'f1':
                    dev_f1_avg = val_results[k]

            self.logger.info('time consumption:%.2f(min),  validation loss: %.5f, %s' %
                             (time_span, np.array(loss_values).mean(), val_res_str))
            if np.array(dev_f1_avg).mean() > self.best_f1_val:
                unprocessed = 0
                self.best_f1_val = np.array(dev_f1_avg).mean()
                best_at_epoch = epoch
                self.saver.save(self.sess, self.configs.checkpoints_dir + '/' + self.configs.checkpoint_name,
                                global_step=self.model.global_step)
                self.logger.info('saved the new best model with f1: %.3f' % self.best_f1_val)
            else:
                unprocessed += 1

            if self.configs.is_early_stop:
                if unprocessed >= self.configs.patient:
                    self.logger.info('early stopped, no progress obtained within {} epochs'.format(self.configs.patient))
                    self.logger.info('overall best f1 is {} at {} epoch'.format(self.best_f1_val, best_at_epoch))
                    self.logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
                    self.sess.close()
                    return
        self.logger.info('overall best f1 is {} at {} epoch'.format(self.best_f1_val, best_at_epoch))
        self.logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
        self.sess.close()



