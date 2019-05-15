# coding: utf-8

# stacked sparse auto encoder + sml classifier
# classifier2

import tensorflow as tf
import functions as F

ATTR = 121  # number of attribute
H = [200, 300]  # number of hidden layer unit
LABEL_NUM = 5

INPUT = "data/kdd_train_1000.csv"
INPUT1 = "data/kdd_train_20Percent.csv"
INPUT2 = "data/kdd_train_39_class.csv"
INPUT3 = "data/kdd_test_39_class.csv"

PARAM_DIR2 = "train2/model_classifier2_5class.ckpt"

# define networks
# ------------------------------------------------------------------------------------------------------------------
x = tf.placeholder(tf.float32, [None, ATTR], name='x')
W1_cp = tf.placeholder(tf.float32, (ATTR, H[0]), name='l1_W1')
b1_cp = tf.placeholder(tf.float32, [H[0]], name='l1_b1')
W2_cp = tf.placeholder(tf.float32, (H[0], H[1]), name='l2_W1')
b2_cp = tf.placeholder(tf.float32, [H[1]], name='l2_b1')
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

# ----------------------------------------------------------------------------------------------------------------------
# Prepare Saver
c2_param = []
for variable in tf.trainable_variables():
    variable_name = variable.name
    if variable_name.find('c2_') >= 0:
        c2_param.append(variable)

saver2 = tf.train.Saver(c2_param)
# ----------------------------------------------------------------------------------------------------------------------


def prediction(batch_x):
    with tf.Session() as sess2:
        # check the model.ckpt
        ckpt = tf.train.get_checkpoint_state('train2/')
        if ckpt:
            last_model = ckpt.model_checkpoint_path
            print("Load %s" % last_model)
            saver2.restore(sess2, PARAM_DIR2)
            pre = sess2.run(y[1], feed_dict={x: [batch_x]})

            return pre
        else:
            print("There are not learned parameters!")
            exit(0)
