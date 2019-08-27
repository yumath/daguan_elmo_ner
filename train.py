# encoding=utf8
import os
import json
import pickle
import random
import itertools
from collections import OrderedDict

import tensorflow as tf
import numpy as np
from model import Model
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping, elmo_char_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, clean, create_model, save_model
from utils import print_config, save_config, load_config, test_ner
from data_utils import input_from_line, BatchManager

from bilm import BidirectionalLanguageModel, Batcher

flags = tf.app.flags
flags.DEFINE_boolean("clean",       False,      "clean train folder")
flags.DEFINE_boolean("train",       False,      "Wither train the model")
# configurations for the model
# flags.DEFINE_integer("char_dim",    1024,        "Embedding size for characters")
flags.DEFINE_integer("lstm_dim",    256,        "Num of hidden units in LSTM")
flags.DEFINE_string("tag_schema",   "iobes",    "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_integer("batch_size",  32,         "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")
flags.DEFINE_boolean("lower",       True,       "Wither lower case")

flags.DEFINE_integer("max_epoch",   50,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "result",       "Path for results")
flags.DEFINE_string("emb_file",     "data/wor2vec.txt", "Path for pre_trained embedding")
flags.DEFINE_string("train_file",   "data/train.txt",  "Path for train data")

# for elmo
flags.DEFINE_integer("max_chars",    7,        "Embedding size for characters")
flags.DEFINE_string("elmo_options", os.path.join('output', 'options.json'),
                    "Path for elmo LM options.json")
flags.DEFINE_string("elmo_weights", os.path.join('output', 'weights.hdf5'),
                    "Path for elmo LM weights.hdf5")
flags.DEFINE_string("elmo_vocab",   os.path.join('data', 'vocab.txt'),
                    "File for vocab")

FLAGS = tf.app.flags.FLAGS

assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


def result_to_file(results):

    def tag_to_line(answer):
        string = ''
        idx = 0
        for char, tag in answer:
            if tag[0] == "S":
                string = string + char + '/' + tag[2:] + '  '
            elif tag[0] == "B":
                string += char
                if idx == len(answer) -1:
                    string = string + '/' + tag[2:]
            elif tag[0] == "I":
                string += '_'
                string += char
                if idx == len(answer) -1:
                    string = string + '/' + tag[2:]
            elif tag[0] == "E":
                string += '_'
                string += char
                string = string + '/' + tag[2:] + '  '
            elif tag[0] == "O":
                if string == '' or string[-1] == ' ':
                    string += char
                else:
                    string += '_'
                    string += char

                if idx == len(answer)-1:
                    string += '/o  '
                elif answer[idx+1][1] != 'O':
                    string += '/o  '
            idx += 1
        return string.strip()

    ans = []
    for i, sentence in enumerate(results):
        ans.append(tag_to_line(sentence))

    with open('data/result.txt', 'w', encoding='utf-8')as f:
        for line in ans:
            f.write(line+'\n')
        print('process done!')


# config for the model
def config_model(tag_to_id):
    config = OrderedDict()
    #config["num_chars"] = len(char_to_id)
    config["num_tags"] = len(tag_to_id)
    #config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size

    #config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    #config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config


def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def get_batcher():
    with open(FLAGS.elmo_options, 'r') as fin:
       options = json.load(fin)

    max_word_length = options['char_cnn']['max_characters_per_token']

    elmo_batcher = Batcher(FLAGS.elmo_vocab, max_word_length)

    return elmo_batcher


def load_elmo():
    model = BidirectionalLanguageModel(options_file=FLAGS.elmo_options,
                                       weight_file=FLAGS.elmo_weights)
    return model


def train():
    # load data sets
    datasets = load_sentences(FLAGS.train_file, FLAGS.lower)
    random.shuffle(datasets)
    train_sentences = datasets[:14000]
    test_sentences = datasets[14000:]

    # Use selected tagging scheme (IOB / IOBES)
    update_tag_scheme(train_sentences, FLAGS.tag_schema)
    update_tag_scheme(test_sentences, FLAGS.tag_schema)

    # create maps if not exist
    if not os.path.isfile(FLAGS.map_file):
        # create dictionary for word
        char_to_id, _ = elmo_char_mapping(FLAGS.elmo_vocab)

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, tag_to_id, id_to_tag], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, tag_to_id, id_to_tag = pickle.load(f)

    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(train_sentences, char_to_id, tag_to_id, FLAGS.lower)
    test_data = prepare_dataset(test_sentences, char_to_id, tag_to_id, FLAGS.lower)
    print("%i / %i sentences in train / dev." % (len(train_data), len(test_data)))

    elmo_batcher = get_batcher()
    train_manager = BatchManager(train_data, FLAGS.batch_size, elmo_batcher)
    test_manager = BatchManager(test_data, FLAGS.batch_size, elmo_batcher)
    # make path for store log and model if not exist
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(tag_to_id)
        save_config(config, FLAGS.config_file)
    make_path(FLAGS)

    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        elmo_model = load_elmo()
        model = create_model(sess, Model, FLAGS.ckpt_path, elmo_model, config, logger)
        logger.info("start training")
        loss = []
        for i in range(FLAGS.max_epoch):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, NER loss:{:>9.6f}".format(
                        iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "test", test_manager, id_to_tag, logger)
            # evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            if best:
                save_model(sess, model, FLAGS.ckpt_path, logger)
            # evaluate(sess, model, "test", test_manager, id_to_tag, logger)


def main(_):
    if FLAGS.train:
        if FLAGS.clean:
            clean(FLAGS)
        train()
    # else:
    #     predict()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    FLAGS.train = True
    FLAGS.clean = True
    tf.app.run(main)