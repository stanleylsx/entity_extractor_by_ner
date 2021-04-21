# -*- coding: utf-8 -*-
# @Time : 2020/9/13 3:18 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : train.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np
import math
import time
from tqdm import tqdm
from engines.model import BiLSTM_CRFModel
from engines.utils.metrics import metrics
from tensorflow_addons.text.crf import crf_decode
from transformers import TFBertModel, BertTokenizer


def train(configs, data_manager, logger):
    vocab_size = data_manager.max_token_number
    num_classes = data_manager.max_label_number
    learning_rate = configs.learning_rate
    max_to_keep = configs.checkpoints_max_to_keep
    checkpoints_dir = configs.checkpoints_dir
    checkpoint_name = configs.checkpoint_name
    best_f1_val = 0.0
    best_at_epoch = 0
    unprocessed = 0
    very_start_time = time.time()
    epoch = configs.epoch
    batch_size = configs.batch_size

    # 优化器大致效果Adagrad>Adam>RMSprop>SGD
    if configs.optimizer == 'Adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif configs.optimizer == 'Adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    elif configs.optimizer == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif configs.optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if configs.use_bert:
        bert_model = TFBertModel.from_pretrained('bert-base-chinese')
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        X_train, y_train, att_mask_train, X_val, y_val, att_mask_val = data_manager.get_training_set()
    else:
        X_train, y_train, X_val, y_val = data_manager.get_training_set()
        att_mask_train, att_mask_val = np.array([]), np.array([])
        bert_model, tokenizer = None, None

    bilstm_crf_model = BiLSTM_CRFModel(configs, vocab_size, num_classes, configs.use_bert)
    checkpoint = tf.train.Checkpoint(model=bilstm_crf_model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpoints_dir, checkpoint_name=checkpoint_name, max_to_keep=max_to_keep)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    num_iterations = int(math.ceil(1.0 * len(X_train) / batch_size))
    num_val_iterations = int(math.ceil(1.0 * len(X_val) / batch_size))
    logger.info(('+' * 20) + 'training starting' + ('+' * 20))
    for i in range(epoch):
        start_time = time.time()
        # shuffle train at each epoch
        sh_index = np.arange(len(X_train))
        np.random.shuffle(sh_index)
        X_train = X_train[sh_index]
        y_train = y_train[sh_index]
        if configs.use_bert:
            att_mask_train = att_mask_train[sh_index]
        logger.info('epoch:{}/{}'.format(i + 1, epoch))
        for iteration in tqdm(range(num_iterations)):
            if configs.use_bert:
                X_train_batch, y_train_batch, att_mask_batch = data_manager.next_batch(
                    X_train, y_train, att_mask_train, start_index=iteration * batch_size)
                # 计算没有加入pad之前的句子的长度
                inputs_length = tf.math.count_nonzero(X_train_batch, 1)
                # 获得bert的模型输出
                model_inputs = bert_model(X_train_batch, attention_mask=att_mask_batch)[0]
            else:
                X_train_batch, y_train_batch = data_manager.next_batch(
                    X_train, y_train, start_index=iteration * batch_size)
                # 计算没有加入pad之前的句子的长度
                inputs_length = tf.math.count_nonzero(X_train_batch, 1)
                model_inputs = X_train_batch
            with tf.GradientTape() as tape:
                logits, log_likelihood, transition_params = bilstm_crf_model(
                    inputs=model_inputs, inputs_length=inputs_length, targets=y_train_batch, training=1)
                loss = -tf.reduce_mean(log_likelihood)
            # 定义好参加梯度的参数
            gradients = tape.gradient(loss, bilstm_crf_model.trainable_variables)
            # 反向传播，自动微分计算
            optimizer.apply_gradients(zip(gradients, bilstm_crf_model.trainable_variables))
            if iteration % configs.print_per_batch == 0 and iteration != 0:
                batch_pred_sequence, _ = crf_decode(logits, transition_params, inputs_length)
                measures, _ = metrics(
                    X_train_batch, y_train_batch, batch_pred_sequence, configs, data_manager, tokenizer)
                res_str = ''
                for k, v in measures.items():
                    res_str += (k + ': %.3f ' % v)
                logger.info('training batch: %5d, loss: %.5f, %s' % (iteration, loss, res_str))

        # validation
        logger.info('start evaluate engines...')
        loss_values = []
        val_results = {}
        val_labels_results = {}
        for label in data_manager.suffix:
            val_labels_results.setdefault(label, {})
        for measure in configs.measuring_metrics:
            val_results[measure] = 0
        for label, content in val_labels_results.items():
            for measure in configs.measuring_metrics:
                val_labels_results[label][measure] = 0

        for iteration in tqdm(range(num_val_iterations)):
            if configs.use_bert:
                X_val_batch, y_val_batch, att_mask_batch = data_manager.next_batch(
                    X_val, y_val, att_mask_val, iteration * batch_size)
                inputs_length_val = tf.math.count_nonzero(X_val_batch, 1)
                # 获得bert的模型输出
                model_inputs = bert_model(X_val_batch, attention_mask=att_mask_batch)[0]
            else:
                X_val_batch, y_val_batch = data_manager.next_batch(X_val, y_val, iteration * batch_size)
                inputs_length_val = tf.math.count_nonzero(X_val_batch, 1)
                model_inputs = X_val_batch
            logits_val, log_likelihood_val, transition_params_val = bilstm_crf_model(
                inputs=model_inputs, inputs_length=inputs_length_val, targets=y_val_batch)
            val_loss = -tf.reduce_mean(log_likelihood_val)
            batch_pred_sequence_val, _ = crf_decode(logits_val, transition_params_val, inputs_length_val)
            measures, lab_measures = metrics(
                X_val_batch, y_val_batch, batch_pred_sequence_val, configs, data_manager, tokenizer)

            for k, v in measures.items():
                val_results[k] += v
            for lab in lab_measures:
                for k, v in lab_measures[lab].items():
                    val_labels_results[lab][k] += v
            loss_values.append(val_loss)

        time_span = (time.time() - start_time) / 60
        val_res_str = ''
        dev_f1_avg = 0
        for k, v in val_results.items():
            val_results[k] /= num_val_iterations
            val_res_str += (k + ': %.3f ' % val_results[k])
            if k == 'f1':
                dev_f1_avg = val_results[k]
        for label, content in val_labels_results.items():
            val_label_str = ''
            for k, v in content.items():
                val_labels_results[label][k] /= num_val_iterations
                val_label_str += (k + ': %.3f ' % val_labels_results[label][k])
            logger.info('label: %s, %s' % (label, val_label_str))
        logger.info('time consumption:%.2f(min), %s' % (time_span, val_res_str))

        if np.array(dev_f1_avg).mean() > best_f1_val:
            unprocessed = 0
            best_f1_val = np.array(dev_f1_avg).mean()
            best_at_epoch = i + 1
            checkpoint_manager.save()
            logger.info('saved the new best model with f1: %.3f' % best_f1_val)
        else:
            unprocessed += 1

        if configs.is_early_stop:
            if unprocessed >= configs.patient:
                logger.info('early stopped, no progress obtained within {} epochs'.format(configs.patient))
                logger.info('overall best f1 is {} at {} epoch'.format(best_f1_val, best_at_epoch))
                logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
                return
    logger.info('overall best f1 is {} at {} epoch'.format(best_f1_val, best_at_epoch))
    logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
