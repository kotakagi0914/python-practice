# coding: utf-8

# sparse auto encoder + sml classifier
# classifier1

import tensorflow as tf
import functions as F

ATTR = 121  # number of attribute
H = [242]  # number of hidden layer unit
# LABEL_NUM = 6
LABEL_NUM = 5  # -----

INPUT = "data/kdd_train_1000.csv"
INPUT1 = "data/kdd_train_20Percent.csv"
INPUT2 = "data/kdd_train_39_class.csv"
INPUT3 = "data/kdd_test_39_class.csv"

# PARAM_DIR = "train1/model_classifier1.ckpt"
# PARAM_DIR = "train1/model_classifier1___h_30.ckpt"
PARAM_DIR = "train1/model_classifier1_5class.ckpt"

# define networks
# ------------------------------------------------------------------------------------------------------------------
x = tf.placeholder(tf.float32, [None, ATTR], name='x')
W1_cp = tf.placeholder(tf.float32, (ATTR, H[0]), name='l1_W1')
b1_cp = tf.placeholder(tf.float32, [H[0]], name='l1_b1')
y_ = tf.placeholder(tf.float32, [None, LABEL_NUM], name='label')
# ----------------------------------------------------------------------------------------------------------------------
# define layer1
layer1 = F.create_a_layer('c1_layer_1', x, H[0], reuse=False)
loss1 = F.calc_loss('c1_loss1', layer1)

train_op1 = F.training('c1_layer_1', loss1)
layer_classifier_input = tf.nn.sigmoid(tf.matmul(x, W1_cp) + b1_cp, name='c1_input_to_lc')
# ----------------------------------------------------------------------------------------------------------------------
# define classifier layer
l1 = tf.nn.sigmoid(tf.matmul(x, layer1[1]) + layer1[2])
y = F.classifier_layer('c1_classifier', l1, reuse=False)

with tf.name_scope('c1_classifier_'):
    # _cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y[1], y_), name='c1_cross_entropy')
    _cross_entropy = -tf.reduce_sum(y_ * tf.log(y[1]))
    # train_op_classifier = F.training('c1_classifier', _cross_entropy)
    train_op_classifier = tf.train.AdamOptimizer().minimize(_cross_entropy)

    accuracy_ = F.calc_acc(y[1], y_)
# ----------------------------------------------------------------------------------------------------------------------
# Prepare Session
saver = tf.train.Saver()
# ----------------------------------------------------------------------------------------------------------------------


def prediction(batch_x, batch_y):
    with tf.Session() as sess:
        # check the model.ckpt
        ckpt = tf.train.get_checkpoint_state('train1/')
        if ckpt:
            last_model = ckpt.model_checkpoint_path
            print("Load %s" % last_model)
            saver.restore(sess, PARAM_DIR)
            pre = sess.run(y[1], feed_dict={x: batch_x, y_: batch_y})
            return pre
        else:
            print("There are not learned parameters!")
            return 0
