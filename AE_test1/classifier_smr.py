# coding: utf-8

# just smr classifier
# classifier0

import tensorflow as tf
import functions as F

ATTR = 121  # number of attribute

# LABEL_NUM = 6
LABEL_NUM = 5  # -----

INPUT = "data/kdd_train_1000.csv"
INPUT1 = "data/kdd_train_20Percent.csv"
INPUT2 = "data/kdd_train_39_class.csv"
INPUT3 = "data/kdd_test_39_class.csv"

PARAM_DIR = "train0/model_classifier_5class.ckpt"


def smr_layer(_x, _w, _b):
    _y = tf.nn.softmax(tf.matmul(_x, _w) + _b, name='SMR')
    return [_x, _y]

# define networks
# ------------------------------------------------------------------------------------------------------------------
x = tf.placeholder(tf.float32, [None, ATTR], name='x')
y_ = tf.placeholder(tf.float32, [None, LABEL_NUM], name='label')
# ----------------------------------------------------------------------------------------------------------------------

with tf.variable_scope('c0_network'):
    w = F.weight_variable((ATTR, LABEL_NUM), '_w0')
    b = F.bias_variable([LABEL_NUM], '_b0')
    smr = smr_layer(x, w, b)

with tf.name_scope('c0_classifier'):
    # _cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(smr[1], y_), name='c0_cross_entropy')
    _cross_entropy = -tf.reduce_sum(y_ * tf.log(smr[1]))
    train_op_classifier = F.training('c0_classifier', _cross_entropy)

    accuracy_ = F.calc_acc(smr[1], y_)

# ----------------------------------------------------------------------------------------------------------------------
    # Prepare Session
    saver = tf.train.Saver()
# ----------------------------------------------------------------------------------------------------------------------


def prediction(batch_x, batch_y):
    with tf.Session() as sess:
        # check the model.ckpt
        ckpt = tf.train.get_checkpoint_state('train0/')
        if ckpt:
            last_model = ckpt.model_checkpoint_path
            print("Load %s" % last_model)
            saver.restore(sess, PARAM_DIR)
            pre = sess.run(smr[1], feed_dict={x: batch_x, y_: batch_y})

            return pre
        else:
            print("There are not learned parameters!\nGoing to start to learn!")
            exit(0)
