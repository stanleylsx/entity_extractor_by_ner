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

        self.bidirectional = configs.bidirectional  # True
        self.num_layers = configs.encoder_layers  # 1
        self.emb_dim = configs.embedding_dim  # 200
        self.hidden_dim = configs.hidden_dim  # 200
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
        self.batch_size = configs.batch_size  # 32
        self.max_sequence_length = configs.max_sequence_length  # 300
        self.num_tokens = data_manager.max_token_number  # 4314
        self.num_classes = data_manager.max_label_number  # 8
        self.initializer = tf.contrib.layers.xavier_initializer()
        if configs.use_pre_trained_embedding:
            embedding_matrix = data_manager.get_embedding(configs.token_emb_dir)
            self.embedding = tf.Variable(embedding_matrix, trainable=False, name='emb', dtype=tf.float32)
        else:
            self.embedding = tf.get_variable('emb', [self.num_tokens, self.emb_dim], trainable=True,
                                             initializer=self.initializer)  # (4314, 200)
        self.learning_rate = configs.learning_rate  # 0.001
        self.dropout_rate = configs.dropout  # 0.5
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
        self.inputs = tf.placeholder(shape=[None, self.max_sequence_length], dtype=tf.int32)  # (?, 300)
        self.targets = tf.placeholder(shape=[None, self.max_sequence_length], dtype=tf.int32)  # (?, 300)

        inputs_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)  # (?, 300, 200)
        inputs_emb = tf.transpose(inputs_emb, [1, 0, 2])   # (300, ?, 200)
        inputs_emb = tf.reshape(inputs_emb, [-1, self.emb_dim])  # (?, 200)
        inputs_emb = tf.split(inputs_emb, self.max_sequence_length, 0)

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
            self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)  # (?,)
            self.length = tf.cast(self.length, tf.int32)  # (?,)

            # forward and backward
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                lstm_cell_fw,
                lstm_cell_bw,
                inputs_emb,
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
                inputs_emb,
                dtype=tf.float32,
                sequence_length=self.length
            )
        # outputs: list_steps[batch, 2*dim]
        outputs = tf.concat(outputs, 1)  # (?, 120000)
        outputs = tf.reshape(outputs, [self.batch_size, self.max_sequence_length, self.hidden_dim * 2])  # (32, 300, 400)

        # linear
        outputs = tf.reshape(outputs, [-1, self.hidden_dim * 2])  # shape=(9600, 400)
        softmax_w = tf.get_variable('softmax_w', [self.hidden_dim * 2, self.num_classes], initializer=self.initializer)  # (400, 8)
        softmax_b = tf.get_variable('softmax_b', [self.num_classes], initializer=self.initializer)  # (8,)
        logits = tf.matmul(outputs, softmax_w) + softmax_b  # (9600, 8)
        logits = tf.reshape(logits, [self.batch_size, self.max_sequence_length, self.num_classes])  # (32, 300, 8)
        # add crf
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(logits, self.targets, self.length)
        self.batch_pred_sequence, self.batch_viterbi_score = tf.contrib.crf.crf_decode(logits, transition_params, self.length)
        self.loss = tf.reduce_mean(-log_likelihood)
        self.summary = tf.summary.scalar('loss', self.loss)
        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
