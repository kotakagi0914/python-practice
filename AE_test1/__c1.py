# coding: utf-8

# sparse auto encoder + sml classifier
# classifier1

# to check the cross entropy value

import tensorflow as tf
import numpy as np
import random
import time
import setting_classifier_5_class  # -----
import functions as F

ATTR = 121  # number of attribute
H = [30]  # number of hidden layer unit
LEARNING_RATE = 0.1
EPOCH = 10
EPOCH_CLASSIFIER = 10
BATCH_SIZE = 30
LABEL_NUM = 5  # -----

OUTPUT1 = "result/classifier1_sparse_h242_mac.csv"
OUTPUT_C = "result/classifier1_mac.csv"
OUTPUT_C_ = "result/classifier1_histogram_mac.csv"

INPUT = "data/kdd_train_1000.csv"
INPUT1 = "data/kdd_train_20Percent.csv"
INPUT2 = "data/kdd_train_39_class.csv"
INPUT3 = "data/kdd_test_39_class.csv"
INPUT_N = "data/kdd_train_normal.csv"
INPUT_D = "data/kdd_train_dos.csv"
INPUT_P = "data/kdd_train_pro.csv"


# PARAM_DIR = "train1/model_classifier1.ckpt"
# PARAM_DIR = "train1/model_classifier1___new.ckpt"
PARAM_DIR = "train_pra/model_classifier1_5class2.ckpt"  # -----
# PARAM_DIR = "train1/model_classifier1_5class.ckpt"  # -----


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
_layer1 = F.create_a_layer('c1_layer_1_1', x, H[0], reuse=False)
_loss1 = F.calc_loss('c1_loss1', layer1)

train_op1 = F.training('c1_layer_1', loss1)
train_op2 = F.training('c1_layer_1', _loss1)

layer_classifier_input = tf.nn.sigmoid(tf.matmul(x, W1_cp) + b1_cp, name='c1_input_to_lc')
# ----------------------------------------------------------------------------------------------------------------------
# define classifier layer
l1 = tf.nn.sigmoid(tf.matmul(x, layer1[1]) + layer1[2])
y = F.classifier_layer('c1_classifier', l1, reuse=False)

_l1 = tf.nn.sigmoid(tf.matmul(x, _layer1[1]) + _layer1[2])
_y = F.classifier_layer('a', _l1, reuse=False)

predict_histogram = np.zeros((LABEL_NUM, LABEL_NUM), dtype=np.int32)
tp_ = np.zeros([LABEL_NUM])
tn_ = np.zeros([LABEL_NUM])
fp_ = np.zeros([LABEL_NUM])
fn_ = np.zeros([LABEL_NUM])
P = np.zeros([LABEL_NUM])
R = np.zeros([LABEL_NUM])
_acc = np.zeros([LABEL_NUM])
# ------------------------------------------------------------------------------------------------------------
predict_histogram4 = np.zeros((LABEL_NUM, LABEL_NUM), dtype=np.int32)
tp_4 = np.zeros([LABEL_NUM])
tn_4 = np.zeros([LABEL_NUM])
fp_4 = np.zeros([LABEL_NUM])
fn_4 = np.zeros([LABEL_NUM])
P4 = np.zeros([LABEL_NUM])
R4 = np.zeros([LABEL_NUM])
_acc4 = np.zeros([LABEL_NUM])

with tf.name_scope('c1_classifier'):
    # my calculation
    _cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y[1])))
    # with tensorflow
    _cross_entropy4 = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(_y[1])))

    _loss = _cross_entropy
    _loss4 = _cross_entropy4

    train_op_classifier = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(_loss)
    train_op_classifier4 = tf.train.AdamOptimizer().minimize(_loss4)

    accuracy_ = F.calc_acc(y[1], y_)
    accuracy_4 = F.calc_acc(_y[1], y_)

# ----------------------------------------------------------------------------------------------------------------------
# Prepare Session
saver = tf.train.Saver()
# ----------------------------------------------------------------------------------------------------------------------

# data pre-process
train_data, train_label = [], []
test_data, test_label = [], []

setting_classifier_5_class.data_read(INPUT2, train_data, train_label)  # -----
setting_classifier_5_class.data_read(INPUT3, test_data, test_label)  # -----

N_train = int(len(train_data))

N_val = int(len(test_data))

# randomize index from all_data
index_train = list(range(0, N_train))
index_val = list(range(0, N_val))

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

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('train_pra')
    if ckpt:
        last_model = ckpt.model_checkpoint_path
        print("Load %s" % last_model)
        saver.restore(sess, PARAM_DIR)
        # pre = sess.run(y[1], feed_dict={x: batch_x, y_: batch_y})
        # return pre
    else:
        print("There is not learned parameters!\nGoing to start to learn!")
        init = tf.initialize_all_variables()
        sess.run(init)

    start = time.time()

    # layer1
    # Training--------------------------------------------------------------------------------------------------------------
    for epoch in range(0, EPOCH):
        print("-- Epoch Number: %s --" % (epoch + 1))

        train_start, train_loss = 0, 0.0
        train_loss2 = 0.0
        random.shuffle(index_train)
        for batch_num in tbatch_list:
            in_list = []
            F.data_set(train_start, train_start + batch_num, index_train, train_data, in_list)
            batch_xs = in_list
            train_start += batch_num

            _, _ = sess.run([train_op1, train_op2], feed_dict={x: batch_xs})
            train_loss += sess.run(loss1, feed_dict={x: batch_xs})
            train_loss2 += sess.run(_loss1, feed_dict={x: batch_xs})

        # Validation------------------------------------------------------------------------------------------------------------
        if (epoch + 1) % 10 == 0:

            val_start, val_loss = 0, 0.0
            val_loss2 = 0.0
            random.shuffle(index_val)
            for batch_num in vbatch_list:
                in_list = []
                F.data_set(val_start, val_start + batch_num, index_val, test_data, in_list)
                batch_xs = in_list
                val_start += batch_num

                val_loss += sess.run(loss1, feed_dict={x: batch_xs})
                val_loss2 += sess.run(_loss1, feed_dict={x: batch_xs})

            print("Layer1:")
            print("Training Loss    : %0.15f" % (train_loss / ite_train))
            print("Validation Loss  : %0.15f\n" % (val_loss / ite_val))
            print("Training Loss2   : %0.15f" % (train_loss2 / ite_train))
            print("Validation Loss2 : %0.15f\n" % (val_loss2 / ite_val))

            print("Epoch %s time    : %s [s]\n" % ((epoch + 1), (time.time() - start)))
            saver.save(sess, PARAM_DIR)

    print("Learning Time    : %s [s]\n" % (time.time() - start))
    print("Input:\n%s" % test_data[0:4])
    print("\n-----\n")

    p = sess.run(layer1[6], feed_dict={x: test_data[0:4]})
    print("Learning Result:\n%s" % p)
    print("\n-----\n")

    print("finish layer 1 operation.\ngo to classifier!\n")
# ------------ !! Layer 1 !! -------------------------------------------------------------------------------------------

    classifier_start = time.time()

# classifier layer
# ----------------------------------------------------------------------------------------------------------------------
    for epoch in range(0, EPOCH_CLASSIFIER):
        print("--Epoch Number: %s--" % str(epoch + 1))

        train_start, train_loss, acc = 0, 0.0, 0.0
        train_loss4, acc4 = 0.0, 0.0
        random.shuffle(index_train)
        for batch_num in tbatch_list:
            in_list, in_label = [], []
            F.data_set(train_start, train_start + batch_num, index_train, train_data, in_list)
            F.data_set(train_start, train_start + batch_num, index_train, train_label, in_label)
            batch_xs, batch_ys = in_list, in_label
            train_start += batch_num

            _, y_var = sess.run([train_op_classifier, y], feed_dict={x: batch_xs, y_: batch_ys})
            _ = sess.run([train_op_classifier4], feed_dict={x: batch_xs, y_: batch_ys})

            train_loss += sess.run(_cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
            acc += sess.run(accuracy_, feed_dict={x: batch_xs, y_: batch_ys})

            train_loss4 += sess.run(_cross_entropy4, feed_dict={x: batch_xs, y_: batch_ys})
            acc4 += sess.run(accuracy_4, feed_dict={x: batch_xs, y_: batch_ys})

        # print training result
        print("Training Loss    : %0.15f, Accuracy : %0.15f" % ((train_loss / ite_train), (acc / ite_train)))
        print("Training Loss4   : %0.15f, Accuracy4 : %0.15f" % ((train_loss4 / ite_train), (acc4 / ite_train)))

        # Validation------------------------------------------------------------------------------------------------------------
        # 1 validation epoch cycle
        val_start, val_loss, acc = 0, 0.0, 0.0
        val_loss4, acc4 = 0.0, 0.0

        if (epoch + 1) % 10 == 0:
            random.shuffle(index_val)
            for batch_num in vbatch_list:
                in_list, in_label = [], []
                F.data_set(val_start, val_start + batch_num, index_val, test_data, in_list)
                F.data_set(val_start, val_start + batch_num, index_val, test_label, in_label)
                batch_xs, batch_ys = in_list, in_label
                val_start += batch_num

                val_loss += sess.run(_cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
                acc += sess.run(accuracy_, feed_dict={x: batch_xs, y_: batch_ys})
                pre = sess.run(y[1], feed_dict={x: batch_xs, y_: batch_ys})
                F.update_matrix(pre, batch_ys, predict_histogram)
# ------------------------------------------------------------------------------------------------------------
                val_loss4 += sess.run(_cross_entropy4, feed_dict={x: batch_xs, y_: batch_ys})
                acc4 += sess.run(accuracy_4, feed_dict={x: batch_xs, y_: batch_ys})
                pre4 = sess.run(_y[1], feed_dict={x: batch_xs, y_: batch_ys})
                F.update_matrix(pre4, batch_ys, predict_histogram4)

            pre_sum = np.sum(predict_histogram, 1)
            gt_sum = np.sum(predict_histogram, 0)
            for j in range(0, LABEL_NUM):
                tp_[j] = np.diag(predict_histogram)[j]
                fp_[j] = pre_sum[j] - tp_[j]
                fn_[j] = gt_sum[j] - tp_[j]
                tn_[j] = np.ndarray.sum(predict_histogram) - (tp_[j] + fp_[j] + fn_[j])

                print(predict_histogram[j])
                P[j] = tp_[j] / (tp_[j] + fp_[j] + 1)
                R[j] = tp_[j] / (tp_[j] + fn_[j] + 1)
                _acc[j] = (tp_[j] + tn_[j]) / (tp_[j] + tn_[j] + fp_[j] + fn_[j])

            precision = float(np.sum(P) / LABEL_NUM)
            recall = float(np.sum(R) / LABEL_NUM)
            f_measure = 2 * precision * recall / (precision + recall)
            accuracy2 = float(np.sum(_acc) / LABEL_NUM)

            tp = float(np.sum(tp_))
            tn = float(np.sum(tn_))
            fp = float(np.sum(fp_))
            fn = float(np.sum(fn_))
            accuracy3 = (tp + tn) / (tp + tn + fp + fn)
            print("\n-----\n")
# ------------------------------------------------------------------------------------------------------------
            pre_sum4 = np.sum(predict_histogram4, 1)
            gt_sum4 = np.sum(predict_histogram4, 0)
            for j in range(0, LABEL_NUM):
                tp_4[j] = np.diag(predict_histogram4)[j]
                fp_4[j] = pre_sum4[j] - tp_4[j]
                fn_4[j] = gt_sum4[j] - tp_4[j]
                tn_4[j] = np.ndarray.sum(predict_histogram4) - (tp_4[j] + fp_4[j] + fn_4[j])

                print(predict_histogram4[j])
                P4[j] = tp_4[j] / (tp_4[j] + fp_4[j] + 1)
                R4[j] = tp_4[j] / (tp_4[j] + fn_4[j] + 1)
                _acc4[j] = (tp_4[j] + tn_4[j]) / (tp_4[j] + tn_4[j] + fp_4[j] + fn_4[j] + 1)

            precision4 = float(np.sum(P4) / LABEL_NUM)
            recall4 = float(np.sum(R4) / LABEL_NUM)
            f_measure4 = 2 * precision4 * recall4 / (precision4 + recall4)
            accuracy2_4 = float(np.sum(_acc4) / LABEL_NUM)

            tp4 = float(np.sum(tp_4))
            tn4 = float(np.sum(tn_4))
            fp4 = float(np.sum(fp_4))
            fn4 = float(np.sum(fn_4))
            accuracy3_4 = (tp4 + tn4) / (tp4 + tn4 + fp4 + fn4)
            print("\n-----\n")

            print("Validation Loss  : %0.15f, Accuracy : %0.15f" % ((val_loss / ite_val), (acc / ite_val)))
            print("Accuracy 2       : %0.15f" % accuracy2)
            print("Accuracy 3       : %0.15f" % accuracy3)
            print("Precision        : %0.15f" % precision)
            print("Recall           : %0.15f" % recall)
            print("F Measure        : %0.15f" % f_measure)
            print("\n-----\n")
# ------------------------------------------------------------------------------------------------------------
            print("Validation Loss4 : %0.15f, Accuracy2 : %0.15f" % ((val_loss4 / ite_val), (acc4 / ite_val)))
            print("Accuracy 2_4     : %0.15f" % accuracy2_4)
            print("Accuracy 3_4     : %0.15f" % accuracy3_4)
            print("Precision4       : %0.15f" % precision4)
            print("Recall4          : %0.15f" % recall4)
            print("F Measure4       : %0.15f" % f_measure4)
            print("\n-----\n")

            print("Epoch %s time    : %s [s]\n" % ((epoch + 1), (time.time() - start)))

            saver.save(sess, PARAM_DIR)
            predict_histogram = np.zeros_like(predict_histogram)
            predict_histogram4 = np.zeros_like(predict_histogram4)

    print("Classifier Time  : %s [s]" % str((time.time() - classifier_start)))
    print("Total Time       : %s [s]\n" % str((time.time() - start)))

    print("Input:\n%s" % test_label[0:8])
    print("\n-----\n")
    p = sess.run(y[1], feed_dict={x: test_data[0:8], y_: test_label[0:8]})
    print("Classification Result\n%s" % p)
    print("\n-----\n")
# ------------------------------------------------------------------------------------------------------------
    p = sess.run(_y[1], feed_dict={x: test_data[0:8], y_: test_label[0:8]})
    print("Classification Result2\n%s" % p)
    print("\n-----\n")

print("finished learning classifier1!")
