# encoding = utf8
import math
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

from tensorflow.contrib import rnn
from utils import result_to_json
from data_utils import create_input, iobes_iob
from bilm import BidirectionalLanguageModel, Batcher, weight_layers


class Model(object):
    def __init__(self, config, elmo_model):
        self.elmo = elmo_model

        self.config = config
        self.lr = config["lr"]
        self.lstm_dim = config["lstm_dim"]

        self.num_tags = config["num_tags"]
        #self.num_chars = config["num_chars"]
        self.num_segs = 4

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # add placeholders for the model
        self.char_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="ChatInputs")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Targets")
        self.dropout = tf.placeholder(dtype=tf.float32, name="Dropout")

        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]

        # for elmo
        self.ids = tf.placeholder('int32', shape=(None, None, 7))


        #self.token_ids = tf.placeholder('int64', shape=(None, None), name='token_ids')
        # embeddings for chinese character and segmentation representation
        embedding = self.embedding_layer(self.char_inputs, config)

        # apply dropout before feed to lstm layer
        lstm_inputs = tf.nn.dropout(embedding, self.dropout)

        # bi-directional lstm layer
        lstm_outputs = self.biLSTM_layer(lstm_inputs, self.lstm_dim, self.lengths)

        # logits for tags
        self.logits = self.project_layer(lstm_outputs)

        # loss of the model
        self.loss = self.loss_layer(self.logits, self.lengths)

        optimizer = self.config["optimizer"]
        if optimizer == "sgd":
            self.opt = tf.train.GradientDescentOptimizer(self.lr)
        elif optimizer == "adam":
            self.opt = tf.train.AdamOptimizer(self.lr)
        elif optimizer == "adgrad":
            self.opt = tf.train.AdagradOptimizer(self.lr)
        else:
            raise KeyError
        
            # apply grad clip to avoid gradient explosion
        grads_vars = self.opt.compute_gradients(self.loss)
        capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                            for g, v in grads_vars]
        self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, char_inputs, elmo_model, name=None):
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size],
        """
        # embedding = []
        # with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
        #     self.char_lookup = tf.get_variable(
        #             name="char_embedding",
        #             shape=[self.num_chars, self.char_dim],
        #             initializer=self.initializer)
        #     embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
        #     embed = tf.concat(embedding, axis=-1)

        # load bert embedding

        ops = self.elmo(self.ids)

        elmo_context_input = weight_layers('input', ops, l2_coef=0.0)
        elmo_embedding = elmo_context_input['weighted_op']

        return elmo_embedding

    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.LSTMCell(lstm_dim,
                                                        use_peepholes=True,
                                                        initializer=self.initializer,
                                                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
        return tf.concat(outputs, axis=2)

    def project_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"  if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        _, chars, tags, ids = batch
        feed_dict = {
            self.ids: ids,
            self.char_inputs: np.asarray(chars),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval(sess)
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-2]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results

    def predict_batch(self, sess, data, id_to_tag, batcher, batch_size=100):
        data_manager = BatchManagerNoSort(data, batch_size, batcher)

        results = []
        trans = self.trans.eval(sess)
        for batch in data_manager.iter_batch():
            strings = batch[0]
            #tags = batch[-1]
            lengths, scores = self.run_step(sess, is_train=False, batch=batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                #gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = [id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]]
                for char, pred in zip(string, pred):
                    result.append([char, pred])
                results.append(result)
        return results

    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval(sess)
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return result_to_json(inputs[0][0], tags)


class BatchManagerNoSort(object):
    def __init__(self, data,  batch_size, batcher):
        self.batcher = batcher
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) /batch_size))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(data[int(i*batch_size) : int((i+1)*batch_size)], self.batcher))
        return batch_data

    @staticmethod
    def pad_data(data, batcher):
        strings = []
        chars = []
        tags = []
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            string, idx = line
            padding = ['0'] * (max_length - len(string))
            strings.append(string + padding)

            padding = [0] * (max_length - len(string))
            chars.append(idx + padding)
            tags.append([])

        ids = (batcher.batch_sentences(strings))

        return [strings, chars, tags, ids]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]
