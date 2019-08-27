import os
import pickle
import tensorflow as tf

from model import Model
from utils import create_model
from train import FLAGS, get_batcher, get_logger, load_elmo, load_config, result_to_file


def predict():
    batcher = get_batcher()

    config = load_config(FLAGS.config_file)
    logger = get_logger(os.path.join('log', FLAGS.log_file))
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, tag_to_id, id_to_tag = pickle.load(f)

    def get_test_data(char2id):
        sentences = []
        with open('data/test.txt', 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split('_')
                ids = [char2id[char if char in char2id else '<UNK>'] for char in words]
                sentences.append([words, ids])
        return sentences

    test_data = get_test_data(char_to_id)
    with tf.Session(config=tf_config) as sess:
        elmo_model = load_elmo()
        model = create_model(sess, Model, FLAGS.ckpt_path, elmo_model, config, logger)
        results = model.predict_batch(sess,
                                      data=test_data,
                                      id_to_tag=id_to_tag,
                                      batcher=batcher,
                                      batch_size=FLAGS.batch_size)
        result_to_file(results)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    predict()