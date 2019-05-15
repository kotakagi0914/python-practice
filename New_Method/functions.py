# coding: utf-8

import tensorflow as tf
import numpy as np

RHO = 0.05  # h layer active probability
BETA = 3  # sparsity penalty parameter
LAMBDA = 0.00001  # weight decay parameter
# LEARNING_RATE = 0.1
LEARNING_RATE = 0.0005
LABEL_NUM = 5


def data_set(bs_start, bs_last, ind, all_d, data_array):
    for count in range(bs_start, bs_last):
            data_array.append(all_d[ind[count]])


def weight_variable(shape, scope_name):
    with tf.variable_scope(scope_name):
        weight = tf.get_variable(scope_name, initializer=tf.truncated_normal(shape=shape, stddev=0.1))

    return weight


def bias_variable(shape, scope_name):
    with tf.variable_scope(scope_name):
        bias = tf.get_variable(scope_name, initializer=tf.constant(0.1, shape=shape))

    return bias


def hidden_layer_set(x, w, b, scope_name):
    with tf.name_scope(scope_name):
        h_ = tf.nn.sigmoid(tf.matmul(x, w) + b)

    return h_


def create_a_layer(scope_name, x, h_unit_sizes, reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        input_dim = int(x.get_shape()[1])
        w1 = weight_variable((input_dim, h_unit_sizes), '_W1')
        b1 = bias_variable([h_unit_sizes], '_b1')

        h = hidden_layer_set(x, w1, b1, '_h')

        v = weight_variable(tf.transpose(w1).get_shape(), '_W2')
        b2 = bias_variable([input_dim], '_b2')

        _y_ = tf.nn.sigmoid(tf.matmul(h, v) + b2)

    return [x, w1, b1, v, b2, h, _y_]


def calc_loss(scope_name, z):
    with tf.name_scope(scope_name):
        input_dim = int(z[0].get_shape()[1])

        l2 = tf.reduce_sum(tf.nn.l2_loss(z[6] - z[0])) / input_dim
        weight_decay = (tf.reduce_sum(tf.pow(z[1], 2)) + tf.reduce_sum(tf.pow(z[3], 2))) * LAMBDA * 0.5
        bias_decay = (tf.reduce_sum(tf.pow(z[2], 2)) + tf.reduce_sum(tf.pow(z[4], 2))) * LAMBDA * 0.5
        rho_cap = tf.reduce_sum(z[5], 0) / input_dim
        kl_divergence = BETA * tf.reduce_sum(
            RHO * tf.log(RHO / rho_cap) + (1 - RHO) * tf.log((1 - RHO) / (1 - rho_cap)))

        _loss_ = l2 + weight_decay + bias_decay + kl_divergence

    return _loss_
# ----------------------------------------------------------------------------------------------------------------------


def classifier_layer(scope_name, x, reuse=False):
    with tf.variable_scope(scope_name + '_layer') as scope:
        if reuse:
            scope.reuse_variables()

        input_dim = int(x.get_shape()[1])
        w_class_ = weight_variable((input_dim, LABEL_NUM), '_W100')
        y = tf.nn.softmax(tf.matmul(x, w_class_))

    return [x, y]


def calc_acc(pre, ans):
    with tf.name_scope('cal_acc'):
        index_y = tf.argmax(pre, 1, name='index_y')
        index_y_ = tf.argmax(ans, 1, name='index_y_')
        correct_prediction = tf.equal(index_y, index_y_, name='correct_prediction')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    return accuracy


def update_matrix(p, a, z):
    with tf.name_scope('update_matrix'):
        _index_y = list(np.argmax(p, 1))
        _index_y_ = list(np.argmax(a, 1))
        for i in range(len(_index_y)):
            z[_index_y[i], _index_y_[i]] += 1

    return z


def update_result(p, a, z):
    index_p = int(np.argmax(p))
    index_a = int(np.argmax(a))
    z[index_p, index_a] += 1

    return z
# ----------------------------------------------------------------------------------------------------------------------


def training(scope_name, _loss):
    with tf.name_scope('training_' + scope_name):
        # train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(_loss)
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(_loss)

    return train_step
# ----------------------------------------------------------------------------------------------------------------------
