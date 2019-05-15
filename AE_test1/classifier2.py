# coding: utf-8

# stacked sparse auto encoder + sml classifier
# classifier2

import tensorflow as tf
import functions as F

ATTR = 121  # number of attribute
H = [242, 363]  # number of hidden layer unit
# LABEL_NUM = 6
LABEL_NUM = 5

INPUT = "data/kdd_train_1000.csv"
INPUT1 = "data/kdd_train_20Percent.csv"
INPUT2 = "data/kdd_train_39_class.csv"
INPUT3 = "data/kdd_test_39_class.csv"

# PARAM_DIR = "train2/model_classifier2.ckpt"
# PARAM_DIR = "train2/model_classifier2___new.ckpt"
PARAM_DIR = "train2/model_classifier2_5class.ckpt"


# define networks
# ------------------------------------------------------------------------------------------------------------------
x = tf.placeholder(tf.float32, [None, ATTR], name='x')
W1_cp = tf.placeholder(tf.float32, (ATTR, H[0]), name='l1_W1')
b1_cp = tf.placeholder(tf.float32, [H[0]], name='l1_b1')
W2_cp = tf.placeholder(tf.float32, (H[0], H[1]), name='l2_W1')
b2_cp = tf.placeholder(tf.float32, [H[1]], name='l2_b1')
y_ = tf.placeholder(tf.float32, [None, LABEL_NUM], name='label')
# ----------------------------------------------------------------------------------------------------------------------
# define layer1
layer1 = F.create_a_layer('c2_layer_1', x, H[0], reuse=False)
loss1 = F.calc_loss('c2_loss1', layer1)

train_op1 = F.training('c2_layer_1', loss1)
layer2_input = tf.nn.sigmoid(tf.matmul(x, W1_cp) + b1_cp, name='c2_input_to_l2')
# ----------------------------------------------------------------------------------------------------------------------
# define layer2
layer2 = F.create_a_layer('c2_layer_2', layer2_input, H[1], reuse=False)
loss2 = F.calc_loss('c2_loss2', layer2)

train_op2 = F.training('c2_layer_2', loss2)
layer_classifier_input = tf.nn.sigmoid(tf.matmul(layer2_input, W2_cp) + b2_cp, name='c2_input_to_lc')
# ----------------------------------------------------------------------------------------------------------------------
# define classifier layer
l1 = tf.nn.sigmoid(tf.matmul(x, layer1[1]) + layer1[2])
l2 = tf.nn.sigmoid(tf.matmul(l1, layer2[1]) + layer2[2])
y = F.classifier_layer('c2_classifier', l2, reuse=False)

with tf.name_scope('c2_classifier_'):
    # _cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y[1], y_), name='c2_cross_entropy')
    _cross_entropy = -tf.reduce_sum(y_ * tf.log(y[1]))

    train_op_classifier = F.training('c2_classifier', _cross_entropy)

    accuracy_ = F.calc_acc(y[1], y_)
# ----------------------------------------------------------------------------------------------------------------------
# Prepare Session
saver = tf.train.Saver()
# ----------------------------------------------------------------------------------------------------------------------


def prediction(batch_x, batch_y):
    with tf.Session() as sess:
        # check the model.ckpt
        ckpt = tf.train.get_checkpoint_state('train2/')
        if ckpt:
            last_model = ckpt.model_checkpoint_path
            print("Load %s" % last_model)
            saver.restore(sess, PARAM_DIR)
            pre = sess.run(y[1], feed_dict={x: batch_x, y_: batch_y})
            return pre
        else:
            print("There are not learned parameters!")
            return 0
