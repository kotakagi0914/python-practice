# coding: utf-8

# stacked sparse auto encoder + sml classifier
# classifier3

import tensorflow as tf
import numpy as np
import random
import time
# import csv
# import setting_classifier
import setting_classifier_5_class  # -----
import functions as F

ATTR = 121  # number of attribute
H = [200, 300, 400]  # number of hidden layer unit
# EPOCH = 1000
# EPOCH_CLASSIFIER = 1000
EPOCH = 50
EPOCH_CLASSIFIER = 50
BATCH_SIZE = 25
# LABEL_NUM = 6
LABEL_NUM = 5  # -----

OUTPUT1 = "result/classifier3_h242_1_mac.csv"
OUTPUT2 = "result/classifier3_h363_2_mac.csv"
OUTPUT3 = "result/classifier3_h484_3_mac.csv"
OUTPUT_C = "result/classifier3_.csv"
OUTPUT_C_ = "result/classifier3_histogram.csv"

INPUT = "data/kdd_train_1000.csv"
INPUT1 = "data/kdd_train_20Percent.csv"
INPUT2 = "data/kdd_train_39_class.csv"
INPUT3 = "data/kdd_test_39_class.csv"
INPUT_N = "data/kdd_train_normal.csv"

PARAM_DIR = "train3/model_classifier3_5class.ckpt"  # -----

# define networks
# ------------------------------------------------------------------------------------------------------------------
x = tf.placeholder(tf.float32, [None, ATTR], name='x')
W1_cp = tf.placeholder(tf.float32, (ATTR, H[0]), name='l1_W1')
b1_cp = tf.placeholder(tf.float32, [H[0]], name='l1_b1')
W2_cp = tf.placeholder(tf.float32, (H[0], H[1]), name='l2_W1')
b2_cp = tf.placeholder(tf.float32, [H[1]], name='l2_b1')
W3_cp = tf.placeholder(tf.float32, (H[1], H[2]), name='l3_W1')
b3_cp = tf.placeholder(tf.float32, [H[2]], name='l3_b1')
y_ = tf.placeholder(tf.float32, [None, LABEL_NUM], name='label')
# ----------------------------------------------------------------------------------------------------------------------
# define layer1
layer1 = F.create_a_layer('c3_layer_1', x, H[0], reuse=False)
loss1 = F.calc_loss('c3_loss1', layer1)

train_op1 = F.training('c3_layer_1', loss1)
layer2_input = tf.nn.sigmoid(tf.matmul(x, W1_cp) + b1_cp, name='c3_input_to_l2')
# ----------------------------------------------------------------------------------------------------------------------
# define layer2
layer2 = F.create_a_layer('c3_layer_2', layer2_input, H[1], reuse=False)
loss2 = F.calc_loss('c3_loss2', layer2)

train_op2 = F.training('c3_layer_2', loss2)
layer3_input = tf.nn.sigmoid(tf.matmul(layer2_input, W2_cp) + b2_cp, name='c3_input_to_l3')
# ----------------------------------------------------------------------------------------------------------------------
# define layer3
layer3 = F.create_a_layer('c3_layer_3', layer3_input, H[2], reuse=False)
loss3 = F.calc_loss('c3_loss3', layer3)

train_op3 = F.training('c3_layer_3', loss3)
# ----------------------------------------------------------------------------------------------------------------------
# define classifier layer
l1 = tf.nn.sigmoid(tf.matmul(x, layer1[1]) + layer1[2])
l2 = tf.nn.sigmoid(tf.matmul(l1, layer2[1]) + layer2[2])
l3 = tf.nn.sigmoid(tf.matmul(l2, layer3[1]) + layer3[2])
y = F.classifier_layer('c3_classifier', l3, reuse=False)
predict_histogram = np.zeros((LABEL_NUM, LABEL_NUM), dtype=np.int32)
tp_ = np.zeros([LABEL_NUM])
tn_ = np.zeros([LABEL_NUM])
fp_ = np.zeros([LABEL_NUM])
fn_ = np.zeros([LABEL_NUM])
P = np.zeros([LABEL_NUM])
R = np.zeros([LABEL_NUM])
_acc = np.zeros([LABEL_NUM])

with tf.name_scope('c3_classifier'):
    # _cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y[1], y_), name='c3_cross_entropy')
    # _cross_entropy = -tf.reduce_sum(y_ * tf.log(y[1]))
    _cross_entropy = -tf.reduce_sum(y_ * tf.log(y[1] + pow(10, -6)))
    loss = tf.cast(_cross_entropy, dtype=tf.float64)

    # train_op_classifier = F.training('c3_classifier', _cross_entropy)
    train_op_classifier = tf.train.AdamOptimizer().minimize(loss)

    accuracy_ = F.calc_acc(y[1], y_)
# ----------------------------------------------------------------------------------------------------------------------
# Prepare Session
saver = tf.train.Saver()
# ----------------------------------------------------------------------------------------------------------------------

# data pre-process
train_data, train_label = [], []
test_data, test_label = [], []
# setting_classifier.data_read(INPUT2, train_data, train_label)
# setting_classifier.data_read(INPUT3, test_data, test_label)
# setting_classifier_5_class.data_read(INPUT2, train_data, train_label)  # -----
setting_classifier_5_class.data_read(INPUT3, test_data, test_label)  # -----

N_train = int(len(train_data))
N_val = int(len(test_data))

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


def learn_parameters(batch_data, batch_label):
    batch_xs, batch_ys = batch_data, batch_label
    with tf.Session() as sess:
        # check the model.ckpt
        ckpt = tf.train.get_checkpoint_state('train3')
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
            random.shuffle(index_train)
            for batch_num in tbatch_list:
                in_list = []
                F.data_set(train_start, train_start + batch_num, index_train, train_data, in_list)
                batch_xs = in_list
                train_start += batch_num

                _, layer1_var = sess.run([train_op1, layer1], feed_dict={x: batch_xs})
                train_loss += sess.run(loss1, feed_dict={x: batch_xs})

# Validation------------------------------------------------------------------------------------------------------------
            if (epoch + 1) % 10 == 0:
                val_start, val_loss = 0, 0.0
                random.shuffle(index_val)
                for batch_num in vbatch_list:
                    in_list = []
                    F.data_set(val_start, val_start + batch_num, index_val, test_data, in_list)
                    batch_xs = in_list
                    val_start += batch_num

                    val_loss += sess.run(loss1, feed_dict={x: batch_xs})

                print("C3 Layer1:")
                print("Training Loss    : %0.15f" % (train_loss / ite_train))
                print("Validation Loss  : %0.15f\n" % (val_loss / ite_val))

                print("Epoch %s time    : %s [s]" % ((epoch + 1), (time.time() - start)))
                saver.save(sess, PARAM_DIR)

        print("Learning Time    : %s [s]\n" % (time.time() - start))
        print("Input:\n%s" % test_data[2:4])
        print("\n-----\n")

        p = sess.run(layer1[6], feed_dict={x: test_data[2:4]})
        print("Learning Result:\n%s" % p)
        print("\n-----\n")

        print("finish C3 layer 1 operation.\ngo to layer 2!")
# ------------ !! Layer 1 !! -------------------------------------------------------------------------------------------

# layer2
# Training--------------------------------------------------------------------------------------------------------------
        for epoch in range(0, EPOCH):
            print("-- Epoch Number: %s --" % (epoch + 1))

            train_start, train_loss = 0, 0.0
            random.shuffle(index_train)
            for batch_num in tbatch_list:
                in_list = []
                F.data_set(train_start, train_start + batch_num, index_train, train_data, in_list)
                batch_xs = in_list
                train_start += batch_num

                _, layer2_var = sess.run([train_op2, layer2],
                                         feed_dict={x: batch_xs, W1_cp: layer1_var[1], b1_cp: layer1_var[2]})
                train_loss += sess.run(loss2, feed_dict={x: batch_xs, W1_cp: layer1_var[1], b1_cp: layer1_var[2]})

# Validation------------------------------------------------------------------------------------------------------------
            if (epoch + 1) % 10 == 0:
                val_start, val_loss = 0, 0.0
                random.shuffle(index_val)
                for batch_num in vbatch_list:
                    in_list = []
                    F.data_set(val_start, val_start + batch_num, index_val, test_data, in_list)
                    batch_xs = in_list
                    val_start += batch_num

                    val_loss += sess.run(loss2, feed_dict={x: batch_xs, W1_cp: layer1_var[1], b1_cp: layer1_var[2]})

                print("C3 Layer2:")
                print("Training Loss    : %0.15f" % (train_loss / ite_train))
                print("Validation Loss  : %0.15f\n" % (val_loss / ite_val))

                print("Epoch %s time    : %s [s]" % ((epoch + 1), (time.time() - start)))
                saver.save(sess, PARAM_DIR)

        print("Learning Time    : %s [s]\n" % (time.time() - start))
        print("Input:\n%s" % layer1[5].eval(feed_dict={x: test_data[2:4]}))
        print("\n-----\n")
        p = sess.run(layer2[6], feed_dict={x: test_data[2:4], W1_cp: layer1_var[1], b1_cp: layer1_var[2]})
        print("Learning Result:\n%s" % p)
        print("\n-----\n")

        print("finish C3 layer 2 operation.\ngo to C3 layer 3!\n")
# ------------ !! Layer 2 !! -------------------------------------------------------------------------------------------

# layer3
# Training--------------------------------------------------------------------------------------------------------------
        for epoch in range(0, EPOCH):
            print("-- Epoch Number: %s --" % (epoch + 1))

            train_start, train_loss = 0, 0.0
            random.shuffle(index_train)
            for batch_num in tbatch_list:
                in_list = []
                F.data_set(train_start, train_start + batch_num, index_train, train_data, in_list)
                batch_xs = in_list
                train_start += batch_num

                _, layer3_var = sess.run([train_op3, layer3], feed_dict={x: batch_xs,
                                                                         W1_cp: layer1_var[1], b1_cp: layer1_var[2],
                                                                         W2_cp: layer2_var[1], b2_cp: layer2_var[2]})

                train_loss += sess.run(loss3, feed_dict={x: batch_xs, W1_cp: layer1_var[1], b1_cp: layer1_var[2],
                                                         W2_cp: layer2_var[1], b2_cp: layer2_var[2]})
# Validation------------------------------------------------------------------------------------------------------------
            if (epoch + 1) % 10 == 0:

                val_start, val_loss = 0, 0.0
                random.shuffle(index_val)
                for batch_num in vbatch_list:
                    in_list = []
                    F.data_set(val_start, val_start + batch_num, index_val, test_data, in_list)
                    batch_xs = in_list
                    val_start += batch_num

                    val_loss += sess.run(loss3, feed_dict={x: batch_xs,
                                                           W1_cp: layer1_var[1], b1_cp: layer1_var[2],
                                                           W2_cp: layer2_var[1], b2_cp: layer2_var[2]})

                print("C3 Layer3:")
                print("Training Loss    : %0.15f" % (train_loss / ite_train))
                print("Validation Loss  : %0.15f\n" % (val_loss / ite_val))

                print("Epoch %s time    : %s [s]" % ((epoch + 1), (time.time() - start)))
                saver.save(sess, PARAM_DIR)

        print("Learning Time    : %s [s]\n" % (time.time() - start))
        print("Input:\n%s" % layer2[5].eval(feed_dict={x: test_data[2:4], W1_cp: layer1_var[1], b1_cp: layer1_var[2]}))
        print("\n-----\n")
        p = sess.run(layer3[6], feed_dict={x: test_data[2:4],
                                           W1_cp: layer1_var[1], b1_cp: layer1_var[2],
                                           W2_cp: layer2_var[1], b2_cp: layer2_var[2]})
        print("Learning Result:\n%s" % p)
        print("\n-----\n")

        print("finish C3 layer 3 operation.\ngo to C3 classifier!\n")
# ------------ !! Layer 3 !! -------------------------------------------------------------------------------------------

        classifier_start = time.time()

# classifier layer
# ----------------------------------------------------------------------------------------------------------------------
        for epoch in range(0, EPOCH_CLASSIFIER):
            print("--Epoch Number: %s--" % str(epoch + 1))

            train_start, train_loss, acc = 0, 0.0, 0.0
            random.shuffle(index_train)
            for batch_num in tbatch_list:
                in_list, in_label = [], []
                F.data_set(train_start, train_start + batch_num, index_train, train_data, in_list)
                F.data_set(train_start, train_start + batch_num, index_train, train_label, in_label)
                batch_xs, batch_ys = in_list, in_label
                train_start += batch_num

                _, y_var = sess.run([train_op_classifier, y], feed_dict={x: batch_xs, y_: batch_ys})
                train_loss += sess.run(_cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
                acc += sess.run(accuracy_, feed_dict={x: batch_xs, y_: batch_ys})

            print("Training Loss    : %0.15f, Accuracy : %0.15f" % ((train_loss / ite_train), (acc / ite_train)))

# Validation------------------------------------------------------------------------------------------------------------
            val_start, val_loss, acc = 0, 0.0, 0.0
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
                accuracy__ = (tp + tn) / (tp + tn + fp + fn)

                # if tp + fp > 0:
                #     precision = tp / (tp + fp)
                #     recall = tp / (tp + fn)
                #     f_measure = 2 * precision * recall / (precision + recall)
                #     accuracy = (tp + tn) / (tp + tn + fp + fn)
                # else:
                #     precision, recall, f_measure = 0.0, 0.0, 0.0

                print("Validation Loss  : %0.15f, Accuracy : %0.15f" % ((val_loss / ite_val), (acc / ite_val)))
                print("Accuracy 2       : %0.15f" % accuracy)
                print("Accuracy 3       : %0.15f" % accuracy__)
                print("Precision        : %0.15f" % precision)
                print("Recall           : %0.15f" % recall)
                print("F Measure        : %0.15f\n" % f_measure)

                print("Epoch %s time    : %s [s]\n" % ((epoch + 1), (time.time() - start)))
                saver.save(sess, PARAM_DIR)
                predict_histogram = np.zeros_like(predict_histogram)

        print("Classifier Time  : %s [s]" % str((time.time() - classifier_start)))
        print("Total Time       : %s [s]\n" % str((time.time() - start)))

        print("Input:\n%s" % test_label[0:8])
        print("\n-----\n")
        p = sess.run(y[1], feed_dict={x: test_data[0:8], y_: test_label[0:8]})
        print("Classification Result\n%s" % p)
        print("\n-----\n")

    print("finished learning classifier3!")
    return p
