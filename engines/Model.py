import os
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Model(object):
    def __init__(self, configs, data_manager):
        os.environ['CUDA_VISIBLE_DEVICES'] = configs.CUDA_VISIBLE_DEVICES
        if configs.mode == 'train':
            self.is_training = True
        else:
            self.is_training = False

        self.bidirectional = configs.bidirectional
        self.num_layers = configs.encoder_layers
        self.emb_dim = configs.embedding_dim
        self.hidden_dim = configs.hidden_dim
        self.is_crf = configs.use_crf
        if configs.cell_type == 'LSTM':
            if self.bidirectional:
                self.cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
            else:
                self.cell = tf.nn.rnn_cell.LSTMCell(2 * self.hidden_dim)
        else:
            if self.bidirectional:
                self.cell = tf.nn.rnn_cell.GRUCell(self.hidden_dim)
            else:
                self.cell = tf.nn.rnn_cell.GRUCell(2 * self.hidden_dim)
        self.batch_size = configs.batch_size
        self.max_sequence_length = configs.max_sequence_length
        self.num_tokens = data_manager.max_token_number
        self.num_classes = data_manager.max_label_number
        self.initializer = tf.contrib.layers.xavier_initializer()
        if configs.use_pre_trained_embedding:
            embedding_matrix = data_manager.get_embedding(configs.token_emb_dir)
            self.embedding = tf.Variable(embedding_matrix, trainable=False, name='emb', dtype=tf.float32)
        else:
            self.embedding = tf.get_variable('emb', [self.num_tokens, self.emb_dim], trainable=True,
                                             initializer=self.initializer)
        self.learning_rate = configs.learning_rate
        self.dropout_rate = configs.dropout
        if configs.optimizer == 'Adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif configs.optimizer == 'Adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        elif configs.optimizer == 'RMSprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif configs.optimizer == 'GD':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)
        self.build()

    def build(self):
        self.inputs = tf.placeholder(tf.int32, [None, self.max_sequence_length])
        self.targets = tf.placeholder(tf.int32, [None, self.max_sequence_length])

        self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)
        self.inputs_emb = tf.transpose(self.inputs_emb, [1, 0, 2])
        self.inputs_emb = tf.reshape(self.inputs_emb, [-1, self.emb_dim])
        self.inputs_emb = tf.split(self.inputs_emb, self.max_sequence_length, 0)

        # lstm cell
        if self.bidirectional:
            lstm_cell_fw = self.cell
            lstm_cell_bw = self.cell

            # dropout
            if self.is_training:
                lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - self.dropout_rate))
                lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - self.dropout_rate))

            lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * self.num_layers)
            lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * self.num_layers)

            # get the length of each sample
            self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
            self.length = tf.cast(self.length, tf.int32)

            # forward and backward
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                lstm_cell_fw,
                lstm_cell_bw,
                self.inputs_emb,
                dtype=tf.float32,
                sequence_length=self.length
            )
        else:
            lstm_cell = self.cell
            if self.is_training:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=(1 - self.dropout_rate))
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers)
            self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
            self.length = tf.cast(self.length, tf.int32)

            outputs, _ = tf.contrib.rnn.static_rnn(
                lstm_cell,
                self.inputs_emb,
                dtype=tf.float32,
                sequence_length=self.length
            )
        # outputs: list_steps[batch, 2*dim]
        outputs = tf.concat(outputs, 1)
        outputs = tf.reshape(outputs, [self.batch_size, self.max_sequence_length, self.hidden_dim * 2])

        # linear
        self.outputs = tf.reshape(outputs, [-1, self.hidden_dim * 2])
        self.softmax_w = tf.get_variable('softmax_w', [self.hidden_dim * 2, self.num_classes], initializer=self.initializer)
        self.softmax_b = tf.get_variable('softmax_b', [self.num_classes], initializer=self.initializer)
        self.logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b
        self.logits = tf.reshape(self.logits, [self.batch_size, self.max_sequence_length, self.num_classes])
        # print(self.logits.get_shape().as_list())
        if not self.is_crf:
            # softmax
            softmax_out = tf.nn.softmax(self.logits, axis=-1)
            self.batch_pred_sequence = tf.cast(tf.argmax(softmax_out, -1), tf.int32)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets)
            mask = tf.sequence_mask(self.length)
            self.losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)
        else:
            # crf
            self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.targets, self.length)
            self.batch_pred_sequence, self.batch_viterbi_score = tf.contrib.crf.crf_decode(self.logits,
                                                                                           self.transition_params,
                                                                                           self.length)
            self.loss = tf.reduce_mean(-self.log_likelihood)
        self.train_summary = tf.summary.scalar('loss', self.loss)
        self.dev_summary = tf.summary.scalar('loss', self.loss)
        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
