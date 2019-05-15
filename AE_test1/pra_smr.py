# coding: utf-8

# sml classifier
# classifier0

import tensorflow as tf
import numpy as np
import random
import time
# import csv
# import setting_classifier
import setting_classifier_5_class  # -----
import functions as F

ATTR = 121  # number of attribute
EPOCH_CLASSIFIER = 50
BATCH_SIZE = 25
LEARNING_RATE = 0.1
# LABEL_NUM = 6
LABEL_NUM = 5

OUTPUT_C = "result/classifier0_.csv"
OUTPUT_C_ = "result/classifier0_histogram.csv"

INPUT = "data/kdd_train_1000.csv"
INPUT1 = "data/kdd_train_20Percent.csv"
INPUT2 = "data/kdd_train_39_class.csv"
INPUT3 = "data/kdd_test_39_class.csv"
INPUT_N = "data/kdd_train_normal.csv"


# PARAM_DIR = "train_pra/model_classifier_smr.ckpt"
PARAM_DIR = "train0/model_classifier0_5class.ckpt"  # -----


def smr_layer(_x, _w, _b):
    _y = tf.nn.softmax(tf.matmul(_x, _w) + _b, name='SMR')

    return [_x, _y]

# data pre-process
train_data, train_label = [], []
test_data, test_label = [], []
# setting_classifier.data_read(INPUT2, train_data, train_label)
# setting_classifier.data_read(INPUT3, test_data, test_label)

# setting_classifier_5_class.data_read(INPUT2, train_data, train_label)
# setting_classifier_5_class.data_read(INPUT_N, train_data, train_label)
setting_classifier_5_class.data_read(INPUT3, test_data, test_label)

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

# define networks
# ------------------------------------------------------------------------------------------------------------------
x = tf.placeholder(tf.float32, [None, ATTR], name='x')
y_ = tf.placeholder(tf.float32, [None, LABEL_NUM], name='label')

# ----------------------------------------------------------------------------------------------------------------------
with tf.variable_scope('c0_network'):
    w = F.weight_variable((ATTR, LABEL_NUM), '_w0')
    b = F.bias_variable([LABEL_NUM], '_b0')
    smr = smr_layer(x, w, b)

# predict_histogram = np.zeros((LABEL_NUM, LABEL_NUM), dtype=np.int32)
tp_ = np.zeros([LABEL_NUM])
tn_ = np.zeros([LABEL_NUM])
fp_ = np.zeros([LABEL_NUM])
fn_ = np.zeros([LABEL_NUM])
P = np.zeros([LABEL_NUM])
R = np.zeros([LABEL_NUM])
_acc = np.zeros([LABEL_NUM])

with tf.name_scope('c0_classifier'):
    # _cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(smr[1], y_), name='c0_cross_entropy')
    # _cross_entropy = -tf.reduce_sum(y_ * tf.log(smr[1]))
    _cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(smr[1] + pow(10, -8))))

    _loss = tf.cast(_cross_entropy, dtype=tf.float64)
    # train_op_classifier = F.training('c0_classifier', _cross_entropy)
    # train_op_classifier = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(_loss)
    train_op_classifier = tf.train.AdamOptimizer().minimize(_loss)

    # tensorflow
    index_y = tf.argmax(smr[1], 1, name='index_y')
    index_y_ = tf.argmax(y_, 1, name='index_y_')
    correct_prediction = tf.equal(index_y, index_y_, name='correct_prediction')

    # accuracy_ = np.mean(float(correct_prediction))
    accuracy_ = F.calc_acc(smr[1], y_)

# ----------------------------------------------------------------------------------------------------------------------
# Prepare Session
saver = tf.train.Saver()
# ----------------------------------------------------------------------------------------------------------------------


def learn_parameters(batch_data, batch_label):
    batch_xs, batch_ys = batch_data, batch_label
    p = []
    predict_histogram = np.zeros((LABEL_NUM, LABEL_NUM), dtype=np.int32)

    with tf.Session() as sess:
        # check the model.ckpt
        ckpt = tf.train.get_checkpoint_state('train0')
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

        classifier_start = time.time()
        start = time.time()
    # classifier layer
    # ----------------------------------------------------------------------------------------------------------------------
        for epoch in range(0, EPOCH_CLASSIFIER):
            print("--Epoch Number: %s--" % str(epoch + 1))

            train_start, train_loss, acc = 0, 0.0, 0.0
            # random.shuffle(index_train)
            # for batch_num in tbatch_list:
            #     in_list, in_label = [], []
            #     F.data_set(train_start, train_start + batch_num, index_train, train_data, in_list)
            #     F.data_set(train_start, train_start + batch_num, index_train, train_label, in_label)
            #     batch_xs, batch_ys = in_list, in_label
            #     train_start += batch_num

            _, y_var = sess.run([train_op_classifier, smr], feed_dict={x: batch_xs, y_: batch_ys})
            train_loss = sess.run(_cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
            acc = sess.run(accuracy_, feed_dict={x: batch_xs, y_: batch_ys})

            print("C0 :")
            # print("Training Loss    : %0.15f, Accuracy : %0.15f" % ((train_loss / ite_train), (acc / ite_train)))
            print("Training Loss    : %0.15f, Accuracy : %0.15f" % (train_loss, acc))

    # Validation------------------------------------------------------------------------------------------------------------
    #     val_start, val_loss, acc = 0, 0.0, 0.0
        if (epoch + 1) % 10 == 0:
            random.shuffle(index_val)
                # for batch_num in vbatch_list:
                #     in_list, in_label = [], []
                #     F.data_set(val_start, val_start + batch_num, index_val, test_data, in_list)
                #     F.data_set(val_start, val_start + batch_num, index_val, test_label, in_label)
                #     batch_xs, batch_ys = in_list, in_label
                #     val_start += batch_num

            val_loss = sess.run(_cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
            acc = sess.run(accuracy_, feed_dict={x: batch_xs, y_: batch_ys})

            pre = sess.run(smr[1], feed_dict={x: batch_xs, y_: batch_ys})
            F.update_matrix(pre, batch_ys, predict_histogram)

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
            accuracy = float(np.sum(_acc) / LABEL_NUM)

            tp = float(np.sum(tp_))
            tn = float(np.sum(tn_))
            fp = float(np.sum(fp_))
            fn = float(np.sum(fn_))
            accuracy3 = (tp + tn) / (tp + tn + fp + fn)

            # print("Validation Loss  : %0.15f, Accuracy : %0.15f" % ((val_loss / ite_val), (acc / ite_val)))
            print("Validation Loss  : %0.15f, Accuracy : %0.15f" % (val_loss, acc))
            print("Accuracy 2       : %0.15f" % accuracy)
            print("Accuracy 3       : %0.15f" % accuracy3)
            print("Precision        : %0.15f" % precision)
            print("Recall           : %0.15f" % recall)
            print("F Measure        : %0.15f" % f_measure)

            print("Epoch %s time    : %s [s]\n" % ((epoch + 1), (time.time() - start)))
            saver.save(sess, PARAM_DIR)
            predict_histogram = np.zeros_like(predict_histogram)

        print("Classifier Time  : %s [s]\n" % str((time.time() - classifier_start)))
        print("Total Time       : %s [s]\n" % str((time.time() - start)))

        print("Input:\n%s" % test_label[0:8])
        print("\n-----\n")
        p = sess.run(smr[1], feed_dict={x: test_data[0:8], y_: test_label[0:8]})
        print("Classification Result\n%s" % p)
        print("\n-----\n")
        print("finished learning classifier0!")
        return p
