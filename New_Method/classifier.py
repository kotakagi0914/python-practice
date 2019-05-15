# coding: utf-8

# stacked sparse auto encoder + sml classifier

# this code has all classifier

import tensorflow as tf
import functions as F


# def smr_layer(_x, _w, _b):
#     _y = tf.nn.softmax(tf.matmul(_x, _w) + _b, name='SMR')
#     return [_x, _y]

ATTR = 121  # number of attribute
H1 = [30]  # number of hidden layer unit
H2 = [200, 300]  # number of hidden layer unit
H3 = [200, 300, 400]  # number of hidden layer unit
LABEL_NUM = 5

INPUT = "data/kdd_train_1000.csv"
INPUT1 = "data/kdd_train_20Percent.csv"
INPUT2 = "data/kdd_train_39_class.csv"
INPUT3 = "data/kdd_test_39_class.csv"

PARAM_DIR = "train0/model_classifier0_5class.ckpt"
PARAM_DIR1 = "train1/model_classifier1_5class.ckpt"
PARAM_DIR2 = "train2/model_classifier2_5class.ckpt"
PARAM_DIR3 = "train3/model_classifier3_5class.ckpt"

# define placeholder
# ------------------------------------------------------------------------------------------------------------------
x = tf.placeholder(tf.float32, [None, ATTR], name='x')
x1 = tf.placeholder(tf.float32, [None, ATTR], name='x1')
x2 = tf.placeholder(tf.float32, [None, ATTR], name='x2')
x3 = tf.placeholder(tf.float32, [None, ATTR], name='x3')

W1_cp = tf.placeholder(tf.float32, (ATTR, H3[0]), name='l1_W1')
b1_cp = tf.placeholder(tf.float32, [H3[0]], name='l1_b1')
W2_cp = tf.placeholder(tf.float32, (H3[0], H3[1]), name='l2_W1')
b2_cp = tf.placeholder(tf.float32, [H3[1]], name='l2_b1')
W3_cp = tf.placeholder(tf.float32, (H3[1], H3[2]), name='l3_W1')
b3_cp = tf.placeholder(tf.float32, [H3[2]], name='l3_b1')
y_ = tf.placeholder(tf.float32, [None, LABEL_NUM], name='label')
# ----------------------------------------------------------------------------------------------------------------------
# c0
with tf.variable_scope('c0_network'):
    w = F.weight_variable((ATTR, LABEL_NUM), '_w0')
    b = F.bias_variable([LABEL_NUM], '_b0')
    smr = tf.nn.softmax(tf.matmul(x, w) + b, name='SMR')
# ----------------------------------------------------------------------------------------------------------------------
# c1
# define layer1
layer1 = F.create_a_layer('c1_layer_1', x1, H1[0], reuse=False)

# define classifier layer
l1 = tf.nn.sigmoid(tf.matmul(x1, layer1[1]) + layer1[2])
y1 = F.classifier_layer('c1_classifier', l1, reuse=False)
# ----------------------------------------------------------------------------------------------------------------------
# c2
# define layer1
layer1 = F.create_a_layer('c2_layer_1', x2, H2[0], reuse=False)
layer2_inputc2 = tf.nn.sigmoid(tf.matmul(x2, W1_cp) + b1_cp, name='c2_input_to_l2')

# define layer2
layer2 = F.create_a_layer('c2_layer_2', layer2_inputc2, H2[1], reuse=False)

# define classifier layer
l1 = tf.nn.sigmoid(tf.matmul(x2, layer1[1]) + layer1[2])
l2 = tf.nn.sigmoid(tf.matmul(l1, layer2[1]) + layer2[2])
y2 = F.classifier_layer('c2_classifier', l2, reuse=False)

# ----------------------------------------------------------------------------------------------------------------------
# c3
# define layer1
layer1 = F.create_a_layer('c3_layer_1', x3, H3[0], reuse=False)
layer2_input = tf.nn.sigmoid(tf.matmul(x3, W1_cp) + b1_cp, name='c3_input_to_l2')

# define layer2
layer2 = F.create_a_layer('c3_layer_2', layer2_input, H3[1], reuse=False)
layer3_input = tf.nn.sigmoid(tf.matmul(layer2_input, W2_cp) + b2_cp, name='c3_input_to_l3')

# define layer3
layer3 = F.create_a_layer('c3_layer_3', layer3_input, H3[2], reuse=False)

# define classifier layer
l1 = tf.nn.sigmoid(tf.matmul(x3, layer1[1]) + layer1[2])
l2 = tf.nn.sigmoid(tf.matmul(l1, layer2[1]) + layer2[2])
l3 = tf.nn.sigmoid(tf.matmul(l2, layer3[1]) + layer3[2])
y3 = F.classifier_layer('c3_classifier', l3, reuse=False)
# ----------------------------------------------------------------------------------------------------------------------
# Prepare Session
# saver = tf.train.Saver()
# saver1 = tf.train.Saver()
# saver2 = tf.train.Saver()
# saver3 = tf.train.Saver()
# ----------------------------------------------------------------------------------------------------------------------


def prediction(batch_x):
    with tf.Session() as sess:
        saver = tf.train.Saver()

        # check the model.ckpt
        ckpt = tf.train.get_checkpoint_state('train0/')
        if ckpt:
            last_model = ckpt.model_checkpoint_path
            print("Load %s" % last_model)
            saver.restore(sess, PARAM_DIR)
            # pre = sess.run(smr[1], feed_dict={x: batch_x})
            # pre = sess.run(smr, feed_dict={x: [batch_x]]})
            pre = sess.run(smr, feed_dict={x: [batch_x]})

            return pre
        else:
            print("There are not learned parameters!\n")
            exit(0)
    sess.close()
    # return pre


def prediction1(batch_x):
    with tf.Session() as sess1:
        saver1 = tf.train.Saver()

        # check the model.ckpt
        ckpt1 = tf.train.get_checkpoint_state('train1/')
        if ckpt1:
            last_model = ckpt1.model_checkpoint_path
            print("Load %s" % last_model)
            saver1.restore(sess1, PARAM_DIR1)
            # pre1 = sess1.run(y[1], feed_dict={x: batch_x})
            print("1")
            # pre1 = sess1.run(y1[1], feed_dict={x: [batch_x]})
            pre1 = sess1.run(y1[1], feed_dict={x1: [batch_x]})

            return pre1
        else:
            print("There are not learned parameters!")
            exit(0)


def prediction2(batch_x):
    with tf.Session() as sess2:
        saver2 = tf.train.Saver()

        # check the model.ckpt
        ckpt2 = tf.train.get_checkpoint_state('train2/')
        if ckpt2:
            last_model = ckpt2.model_checkpoint_path
            print("Load %s" % last_model)
            saver2.restore(sess2, PARAM_DIR2)
            # pre2 = sess2.run(y[1], feed_dict={x: batch_x})
            # pre2 = sess2.run(y2[1], feed_dict={x: [batch_x]})
            pre2 = sess2.run(y2[1], feed_dict={x2: [batch_x]})

            return pre2
        else:
            print("There are not learned parameters!")
            exit(0)


def prediction3(batch_x):
    with tf.Session() as sess3:
        saver3 = tf.train.Saver()

        # check the model.ckpt
        ckpt3 = tf.train.get_checkpoint_state('train3/')
        if ckpt3:
            last_model = ckpt3.model_checkpoint_path
            print("Load %s" % last_model)
            saver3.restore(sess3, PARAM_DIR3)
            # pre3 = sess3.run(y[1], feed_dict={x: batch_x})
            # pre3 = sess3.run(y3[1], feed_dict={x: [batch_x]]})
            pre3 = sess3.run(y3[1], feed_dict={x3: [batch_x]})

            return pre3
        else:
            print("There are not learned parameters!")
            exit(0)
