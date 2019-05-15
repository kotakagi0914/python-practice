# coding: utf-8

# just smr classifier
# classifier0

import tensorflow as tf
import functions as F

ATTR = 121  # number of attribute

LABEL_NUM = 5

INPUT = "data/kdd_train_1000.csv"
INPUT1 = "data/kdd_train_20Percent.csv"
INPUT2 = "data/kdd_train_39_class.csv"
INPUT3 = "data/kdd_test_39_class.csv"

PARAM_DIR = "train0/model_classifier0_5class.ckpt"


def smr_layer(_x, _w, _b):
    _y = tf.nn.softmax(tf.matmul(_x, _w) + _b, name='SMR')
    return [_x, _y]

# define networks
# ------------------------------------------------------------------------------------------------------------------
x = tf.placeholder(tf.float32, [None, ATTR], name='x')
# y_ = tf.placeholder(tf.float32, [None, LABEL_NUM], name='label')
# ----------------------------------------------------------------------------------------------------------------------

with tf.variable_scope('c0_network'):
    w = F.weight_variable((ATTR, LABEL_NUM), '_w0')
    b = F.bias_variable([LABEL_NUM], '_b0')
    smr = smr_layer(x, w, b)

# ----------------------------------------------------------------------------------------------------------------------
# Prepare Saver
c0_param = []
for variable in tf.trainable_variables():
    variable_name = variable.name
    if variable_name.find('c0_') >= 0:
        c0_param.append(variable)

saver = tf.train.Saver(c0_param)
# ----------------------------------------------------------------------------------------------------------------------


def prediction(batch_x):
    with tf.Session() as sess:
        # check the model.ckpt
        ckpt = tf.train.get_checkpoint_state('train0/')
        if ckpt:
            last_model = ckpt.model_checkpoint_path
            print("Load %s" % last_model)
            saver.restore(sess, PARAM_DIR)
            pre = sess.run(smr[1], feed_dict={x: [batch_x]})
            return pre

        else:
            print("There are not learned parameters!\nGoing to start to learn!")
            exit(0)
