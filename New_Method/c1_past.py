# coding: utf-8

# sparse auto encoder + sml classifier
# classifier1

import tensorflow as tf
import functions as F

ATTR = 121  # number of attribute
H = [30]  # number of hidden layer unit
LABEL_NUM = 5

INPUT = "data/kdd_train_1000.csv"
INPUT1 = "data/kdd_train_20Percent.csv"
INPUT2 = "data/kdd_train_39_class.csv"
INPUT3 = "data/kdd_test_39_class.csv"

PARAM_DIR1 = "train_past/model_classifier1_5class.ckpt"

# define networks
# ------------------------------------------------------------------------------------------------------------------
x = tf.placeholder(tf.float32, [None, ATTR], name='x')
W1_cp = tf.placeholder(tf.float32, (ATTR, H[0]), name='l1_W1')
b1_cp = tf.placeholder(tf.float32, [H[0]], name='l1_b1')
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

# ----------------------------------------------------------------------------------------------------------------------
# Prepare Saver
c1_param = []
for variable in tf.trainable_variables():
    variable_name = variable.name
    if variable_name.find('c1_') >= 0:
        c1_param.append(variable)

saver1 = tf.train.Saver(c1_param)
# ----------------------------------------------------------------------------------------------------------------------


def prediction(batch_x):
    with tf.Session() as sess1:
        # check the model.ckpt
        ckpt = tf.train.get_checkpoint_state('train1/')
        if ckpt:
            last_model = ckpt.model_checkpoint_path
            print("Load %s" % last_model)
            saver1.restore(sess1, PARAM_DIR1)
            pre = sess1.run(y[1], feed_dict={x: [batch_x]})

            return pre
        else:
            print("There are not learned parameters!")
            exit(0)
