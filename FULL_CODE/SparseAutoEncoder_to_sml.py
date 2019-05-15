# coding: utf-8

# sparse auto encoder
# in A Deep Learning Approach for Network Intrusion Detection System - 2015


import tensorflow as tf
import random
import time
import csv
import numpy as np
import setting

ATTR = 121  # number of attribute
H = 30
RHO = 0.05  # h layer active probability
BETA = 3  # sparsity penalty parameter
LAMBDA = 0.00001  # weight decay parameter
EPOCH = 30
LEARNING_RATE = 0.1
BATCH_SIZE = 25
RATE = 0.7  # train : validation = 7 : 3
KEEP_RATE = 1.0  # rate of dropout
LABEL_NUM = 6

OUTPUT1 = "result/sparseAE_5000_h30_1_mac.csv"
OUTPUT_C = "result/classifier_with_sparse_mac.csv"

INPUT = "data/kdd_train_1000.csv"
INPUT1 = "data/kdd_train_20Percent.csv"
INPUT2 = "data/kdd_train_39_class.csv"
INPUT3 = "data/kdd_test_39_class.csv"

PARAM_DIR = "train/model_sparseAE_h30.ckpt"


def weight_variable(shape, scope_name):
    with tf.variable_scope(scope_name):
        weight = tf.get_variable(scope_name, initializer=tf.truncated_normal(shape=shape, stddev=0.1))

    return weight


def bias_variable(shape, scope_name):
    with tf.variable_scope(scope_name):
        bias = tf.get_variable(scope_name, initializer=tf.constant(0.1, shape=shape))

    return bias


def data_set(bs_start, bs_last, ind, all_d, data_array):
    for count in range(bs_start, bs_last):
            data_array.append(all_d[ind[count]])


def hidden_layer_set(input_x, w, b, scope_name):
    with tf.name_scope(scope_name):
        h_ = tf.nn.sigmoid(tf.matmul(input_x, w) + b)

    return h_


def create_a_layer(scope_name, input_x, h_unit_sizes, reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        input_dim = int(input_x.get_shape()[1])
        w1 = weight_variable((input_dim, h_unit_sizes), '_W1')
        b1 = bias_variable([h_unit_sizes], '_b1')

        h = hidden_layer_set(input_x, w1, b1, "_h")
        h_drop = tf.nn.dropout(h, KEEP_RATE)

        v = weight_variable(tf.transpose(w1).get_shape(), "_W2")
        b2 = bias_variable([input_dim], '_b2')

        _y_ = tf.nn.sigmoid(tf.matmul(h_drop, v) + b2)

    return {'x': input_x, 'W1': w1, 'b1': b1, 'W2': v, 'b2': b2, 'h': h, 'y': _y_}


def calc_loss(scope_name, z):
    with tf.name_scope(scope_name):
        input_dim = int(z['x'].get_shape()[1])

        l2 = tf.reduce_sum(tf.nn.l2_loss(z['y'] - z['x'])) / input_dim
        weight_decay = (tf.reduce_sum(tf.pow(z['W1'], 2)) + tf.reduce_sum(tf.pow(z['W2'], 2))) * LAMBDA * 0.5
        bias_decay = (tf.reduce_sum(tf.pow(z['b1'], 2)) + tf.reduce_sum(tf.pow(z['b2'], 2))) * LAMBDA * 0.5
        rho_cap = tf.reduce_sum(z['h'], 0) / input_dim
        kl_divergence = BETA * tf.reduce_sum(
            RHO * tf.log(RHO / rho_cap) + (1 - RHO) * tf.log((1 - RHO) / (1 - rho_cap)))

        _loss_ = l2 + weight_decay * bias_decay + kl_divergence

    return _loss_

# ----------------------------------------------------------------------------------------------------------------------


def classifier_layer(scope_name, input_x, reuse=False):
    with tf.variable_scope(scope_name + 'layer') as scope:
        if reuse:
            scope.reuse_variables()

        input_dim = int(input_x.get_shape()[1])
        _W100 = weight_variable((input_dim, LABEL_NUM), '_W100')
        y = tf.nn.softmax(tf.matmul(x, _W100))

    return {'x': input_x, 'y': y}


def classifier_loss(p, a):
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(p, a))

    return cross_entropy


def update_matrix(p, a, z):
    with tf.name_scope('update_matrix'):
        _index_y = list(np.argmax(p, 1))
        _index_y_ = list(np.argmax(a, 1))
        for i in range(len(_index_y)):
            z[_index_y[i], _index_y_[i]] += 1

    return z

# ----------------------------------------------------------------------------------------------------------------------


def training(scope_name, _loss):
    with tf.name_scope('training_' + scope_name):
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(_loss)

    return train_step

# ----------------------------------------------------------------------------------------------------------------------

# data pre-process ver. 2
all_data, label_list = [], []
setting.data_read(INPUT1, all_data, label_list)
LEN_ALL_DATA = len(all_data)
N_train = int(LEN_ALL_DATA * RATE)
N_val = LEN_ALL_DATA - N_train

# randomize index from all_data
index = list(range(0, LEN_ALL_DATA))
index_train = list(range(0, N_train))
index_val = list(range(0, N_val))

# data division
train_list, val_list = [], []
data_set(0, N_train, index, all_data, train_list)
data_set(N_train, LEN_ALL_DATA, index, all_data, val_list)

train_label, val_label = [], []
data_set(0, N_train, index, label_list, train_label)
data_set(N_train, LEN_ALL_DATA, index, label_list, val_label)

# batch setting
ite_train = int(N_train / BATCH_SIZE)
ite_val = int(N_val / BATCH_SIZE)
if N_train % BATCH_SIZE == 0:
    tbatch_list = [BATCH_SIZE] * ite_train
else:
    tbatch_list = [BATCH_SIZE] * ite_train
    tbatch_list[ite_train:ite_train + 1] = [N_train - ite_train * BATCH_SIZE]
    ite_train += 1
if N_val % BATCH_SIZE == 0:
    vbatch_list = [BATCH_SIZE] * ite_val
else:
    vbatch_list = [BATCH_SIZE] * ite_val
    vbatch_list[ite_val:ite_val + 1] = [N_val - ite_val * BATCH_SIZE]
    ite_val += 1


# define all items
# ----------------------------------------------------------------------------------------------------------------------
x = tf.placeholder(tf.float32, [None, ATTR], name='x')
W1_cp = tf.placeholder(tf.float32, (ATTR, H), name='l1_W1')
b1_cp = tf.placeholder(tf.float32, [H], name='l1_b1')
y_ = tf.placeholder(tf.float32, [None, LABEL_NUM], name='label')

# ----------------------------------------------------------------------------------------------------------------------
# define layer1
layer1 = create_a_layer('layer_1', x, H, reuse=False)
loss1 = calc_loss('loss1', layer1)

train_op1 = training('layer_1', loss1)
layer_classifier_input = tf.nn.sigmoid(tf.matmul(x, W1_cp) + b1_cp, name='input_to_lc')

# ----------------------------------------------------------------------------------------------------------------------

y = classifier_layer('classifier', layer_classifier_input, reuse=False)
predict_histogram = np.zeros((LABEL_NUM, LABEL_NUM))

with tf.name_scope('classifier_'):
    _cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y['y'], y_), name='cross_entropy')

    train_op_classifier = training('classifier', _cross_entropy)

    index_y = tf.argmax(y['y'], 1, name='index_y')
    index_y_ = tf.argmax(y_, 1, name='index_y_')
    correct_prediction = tf.equal(index_y, index_y_, name='correct_prediction')
    accuracy_ = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# ----------------------------------------------------------------------------------------------------------------------

init = tf.initialize_all_variables()

# ----------------------------------------------------------------------------------------------------------------------

with tf.Session() as sess:
    sess.run(init)

    start = time.time()
    # output setting
    f = open(OUTPUT1, 'w')
    csv_writer = csv.writer(f)

# layer1
# Training--------------------------------------------------------------------------------------------------------------
    for epoch in range(0, EPOCH):
        print("-- Epoch Number: %s --" % (epoch + 1))

        # 1 train epoch cycle
        train_start, train_loss = 0, 0.0
        random.shuffle(index_train)
        for batch_num in tbatch_list:
            in_list = []
            data_set(train_start, train_start + batch_num, index_train, train_list, in_list)
            batch_xs = in_list
            train_start += batch_num

            _, layer1_var = sess.run([train_op1, layer1], feed_dict={x: batch_xs})
            train_loss += sess.run(loss1, feed_dict={x: batch_xs})

# Validation------------------------------------------------------------------------------------------------------------
        # 1 validation epoch cycle
        val_start, val_loss = 0, 0.0
        random.shuffle(index_val)
        for batch_num in vbatch_list:
            in_list = []
            data_set(val_start, val_start + batch_num, index_val, val_list, in_list)
            batch_xs = in_list
            val_start += batch_num

            val_loss += sess.run(loss1, feed_dict={x: batch_xs})

        if (epoch + 1) % 10 == 0:
            print("Training Loss    : %0.15f" % (train_loss / ite_train))
            print("Validation Loss  : %0.15f\n" % (val_loss / ite_val))

        if (epoch + 1) % 100 == 0:
            print("Epoch %s time    : %s [s]" % ((epoch + 1), (time.time() - start)))
            # saver.save(sess, PARAM_DIR)

        f.writelines("%d,%0.25f,%0.25f\n" % ((epoch + 1), (train_loss / ite_train), (val_loss / ite_val)))

    f.writelines("\n")
    f.close()

    print("Learning Time    : %s [s]\n" % (time.time() - start))
    print("Input:\n%s" % val_list[2:4])  # input
    print("\n-----\n")
    o = sess.run(layer1['y'].eval, feed_dict={x: val_list[2:4]})

    print("Learning Result:\n%s" % o)  # learning result
    print("\n-----\n")

    print("finish layer 1 operation.\ngo to classifier layer!")

# ------------ !! Layer 1 !! -------------------------------------------------------------------------------------------
    LEARNING_RATE /= 5

    f = open(OUTPUT_C, 'w')
    csv_writer_c = csv.writer(f)

# classifier layer
# ----------------------------------------------------------------------------------------------------------------------
    for epoch in range(0, EPOCH):
        print("--Epoch Number: %s--" % str(epoch + 1))

        # 1 train epoch cycle
        train_start, train_loss, acc = 0, 0.0, 0.0
        random.shuffle(index_train)
        for batch_num in tbatch_list:
            in_list, in_label = [], []
            data_set(train_start, train_start + batch_num, index_train, train_list, in_list)
            data_set(train_start, train_start + batch_num, index_train, train_label, in_label)
            batch_xs, batch_ys = in_list, in_label
            train_start += batch_num

            _, y_var = sess.run([train_op_classifier, y], feed_dict={x: batch_xs, y_: batch_ys,
                                                                     W1_cp: layer1_var['W1'], b1_cp: layer1_var['b1']})
            train_loss += sess.run(_cross_entropy, feed_dict={x: batch_xs, y_: batch_ys,
                                                              W1_cp: layer1_var['W1'], b1_cp: layer1_var['b1']})
            acc += sess.run(accuracy_, feed_dict={x: batch_xs, y_: batch_ys,
                                                  W1_cp: layer1_var['W1'], b1_cp: layer1_var['b1']})

        # print training result
        print("Training Loss    : %0.15f, Accuracy : %0.15f" % ((train_loss / ite_train), (acc / ite_train)))

# Validation------------------------------------------------------------------------------------------------------------
        # 1 validation epoch cycle
        val_start, val_loss, acc, t_p, f_p, f_n = 0, 0.0, 0.0, 0.0, 0.0, 0.0
        random.shuffle(index_val)
        for batch_num in vbatch_list:
            in_list, in_label = [], []
            data_set(val_start, val_start + batch_num, index_val, val_list, in_list)
            data_set(val_start, val_start + batch_num, index_val, val_label, in_label)
            batch_xs, batch_ys = in_list, in_label
            val_start += batch_num

            val_loss += sess.run(_cross_entropy, feed_dict={x: batch_xs, y_: batch_ys,
                                                            W1_cp: layer1_var['W1'], b1_cp: layer1_var['b1']})
            acc += sess.run(accuracy_, feed_dict={x: batch_xs, y_: batch_ys,
                                                  W1_cp: layer1_var['W1'], b1_cp: layer1_var['b1']})

            pre = sess.run(y['y'], feed_dict={x: batch_xs, y_: batch_ys,
                                              W1_cp: layer1_var['W1'], b1_cp: layer1_var['b1']})
            update_matrix(pre, batch_ys, predict_histogram)

            t_p = float(np.sum(np.diag(predict_histogram[1:6, 1:6])))
            f_p = float(np.sum(predict_histogram[1:6, 0]))
            f_n = float(np.sum(predict_histogram[0, 1:6]))

            print("True Positive    : %.1f" % t_p)
            print("False Positive   : %.1f" % f_p)
            print("False Negative   : %.1f\n" % f_n)

        if t_p != 0:
            Precision = float(t_p / (t_p + f_p))
            Recall = float(t_p / (t_p + f_n))
            F_Measure = float(2 * Precision * Recall / (Precision + Recall))
        else:
            Precision, Recall, F_Measure = 0.0, 0.0, 0.0

        # print validation result
        if (epoch + 1) % 10 == 0:
            print("Validation Loss  : %0.15f, Accuracy : %0.15f" % ((val_loss / ite_val), (acc / ite_val)))

            print("True Positive    : %f" % t_p)
            print("False Positive   : %f" % f_p)
            print("False Negative   : %f\n" % f_n)
            print("Precision        : %0.15f" % Precision)
            print("Recall           : %0.15f" % Recall)
            print("F Measure        : %0.15f" % F_Measure)

        if (epoch + 1) % 100 == 0:
            print("Epoch %s time    : %s [s]\n" % ((epoch + 1), (time.time() - start)))
            print("\n-----\n")
            p = sess.run(y['y'], feed_dict={x: val_list[2:4], y_: val_label[2:4],
                                            W1_cp: layer1_var['W1'], b1_cp: layer1_var['b1']})
            print(p)  # learning result
            print("\n-----\n")
            # saver.save(sess, PARAM_DIR)

        f.writelines("%d,%0.25f,%0.25f,%0.25f,%0.25f,%0.25f,%0.25f\n" %
                     (epoch, (train_loss / ite_train), (val_loss / ite_val),
                      (acc / ite_val), Precision, Recall, F_Measure))
        predict_histogram = np.zeros_like(predict_histogram)

    print("Learning Time    : %s [s]\n" % str((time.time() - start)))
    print("Input:\n%s" % val_label[2:4])
    print("\n-----\n")
    o = sess.run(y['y'], feed_dict={x: val_list[2:4], y_: val_label[2:4],
                                    W1_cp: layer1_var['W1'], b1_cp: layer1_var['b1']})
    print("Learning result:\n%s" % o)
    print("\n-----\n")
    f.close()
    sess.close()
