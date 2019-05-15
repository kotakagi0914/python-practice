# coding: utf-8

# classifier only sml

import tensorflow as tf
import numpy as np
import random
import time
import csv
import setting

ATTR = 121
H = 30
BATCH_SIZE = 50
EPOCH = 100
RATE = 0.7  # train : validation = 7 : 3
LEARNING_RATE = 0.1
LABEL_NUM = 6
KEEP_RATE = 1.0

OUTPUT = "result/classifierSML_100_h30_mac2.csv"
INPUT = "data/kdd_train_1000.csv"
INPUT1 = "data/kdd_train_20Percent.csv"
INPUT2 = "data/kdd_train_39_class.csv"
INPUT3 = "data/kdd_test_39_class.csv"

PARAM_DIR = "train/model_SoftMaxRegression.ckpt"


def weight_variable(shape, n):
    initial = tf.truncated_normal(shape, stddev=0.1, name="w%d" % n)
    return tf.Variable(initial)


def bias_variable(shape, n):
    initial = tf.constant(0.1, shape=shape, name="b%d" % n)
    return tf.Variable(initial)


def data_set(list_start, list_last, ind, all_d, data_array):
    for count in range(list_start, list_last):
        data_array.append(all_d[ind[count]])


def update_matrix(p, a, z):
    with tf.name_scope('update_matrix'):
        index_y = list(np.argmax(p, 1))
        index_y_ = list(np.argmax(a, 1))
        for i in range(len(_index_y)):
            z[index_y[i], index_y_[i]] += 1

    return z

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
    tbatch_list[ite_train:ite_train + 1] = [N_train - (ite_train * BATCH_SIZE)]
    ite_train += 1
if N_val % BATCH_SIZE == 0:
    vbatch_list = [BATCH_SIZE] * ite_val
else:
    vbatch_list = [BATCH_SIZE] * ite_val
    vbatch_list[ite_val:ite_val + 1] = [N_val - (ite_val * BATCH_SIZE)]
    ite_val += 1

# define network
with tf.name_scope('inference'):
    x = tf.placeholder(tf.float32, [None, ATTR])

    # Variable: W, b1
    W1 = weight_variable((ATTR, H), 1)
    b1 = bias_variable([H], 1)

    # Hidden Layer: h
    h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
    h_drop = tf.nn.dropout(h, KEEP_RATE)

    # Variable:
    W2 = weight_variable((H, LABEL_NUM), 2)
    y = tf.nn.softmax(tf.matmul(h_drop, W2))

    # Ground Truth
    y_ = tf.placeholder(tf.float32, [None, LABEL_NUM])
    predict_histogram = np.zeros((LABEL_NUM, LABEL_NUM))

    # Define Loss Function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    loss = cross_entropy

    # prepare other items
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Use SGD Optimizer
train_step = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
saver = tf.train.Saver()

# Prepare Session
init = tf.initialize_all_variables()

# output setting
f = open(OUTPUT, 'w')
csv_writer = csv.writer(f)

# Training--------------------------------------------------------------------------------------------------------------
with tf.Session() as sess:
    sess.run(init)

    start = time.time()
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

            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            train_loss += sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
            acc += sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})

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

            val_loss += sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
            acc += sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})

            pre = sess.run(y['y'], feed_dict={x: batch_xs, y_: batch_ys})
            update_matrix(pre, batch_ys, predict_histogram)

            t_p = float(np.sum(np.diag(predict_histogram[1:6, 1:6])))
            f_p = float(np.sum(predict_histogram[1:6, 0]))
            f_n = float(np.sum(predict_histogram[0, 1:6]))

            print("True Positive    : %f" % t_p)
            print("False Positive   : %f" % f_p)
            print("False Negative   : %f\n" % f_n)

        if t_p + f_p != 0:
            Precision = t_p / (t_p + f_p)
            Recall = t_p / (t_p + f_n)
            F_Measure = 2 * Precision * Recall / (Precision + Recall)
        else:
            Precision, Recall, F_Measure = 0.0, 0.0, 0.0

        # print validation result
        if (epoch + 1) % 10 == 0:
            print("Validation Loss  : %0.15f, Accuracy : %0.15f" % ((val_loss / ite_val), (acc / ite_val)))
            print("True Positive    : %0.1f" % t_p)
            print("False Positive   : %0.1f" % f_p)
            print("False Negative   : %0.1f\n" % f_n)
            print("Precision        : %0.15f" % Precision)
            print("Recall           : %0.15f" % Recall)
            print("F Measure        : %0.15f" % F_Measure)

        if (epoch + 1) % 100 == 0:
            print("Epoch %s time    : %s [s]\n" % ((epoch + 1), (time.time() - start)))
            saver.save(sess, PARAM_DIR)

        f.writelines("%d,%0.25f,%0.25f,%0.25f,%0.25f,%0.25f,%0.25f\n" %
                     (epoch, (train_loss / ite_train), (val_loss / ite_val),
                      (acc / ite_val), Precision, Recall, F_Measure))

    print("Learning Time    : %s [s]\n" % str((time.time() - start)))
    print("Input:\n%s" % val_label[2:4])
    print("\n-----\n")
    o = y.eval(session=sess, feed_dict={x: val_list[2:4], y_: val_label[2:4]})
    print("Learning result:\n%s" % o)
    print("\n-----\n")
    f.close()
