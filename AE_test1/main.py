# coding: utf-8

# this is the main file of my new method.

# classify traffic in each stages if there is any risk in it.
# there are 3 alert level.

# this is for the learning parameters

import numpy as np
import time
# import random
# import setting_classifier
import setting_classifier_5_class

import functions as F
# import classifier_smr as c0
# import classifier1 as c1
# import classifier2 as c2
# import classifier3 as c3

import pra_smr as c0
import c1_set_para as c1
import c2_set_para as c2
import c3_set_para as c3

G_EPOCH = 10
BATCH_SIZE = 50
LABEL_NUM = 5
PREDICT_THRESHOLD = 0.99
AVAILABLE_CLASSIFIER = 2

INPUT = "data/kdd_train_1000.csv"
INPUT1 = "data/kdd_train_20Percent.csv"
INPUT2 = "data/kdd_train_39_class.csv"
INPUT3 = "data/kdd_test_39_class.csv"

# data pre-process
train_data, train_label = [], []
test_data, test_label = [], []
# setting_classifier.data_read(INPUT3, test_data, test_label)
setting_classifier_5_class.data_read(INPUT2, train_data, train_label)  # -----
# setting_classifier_5_class.data_read(INPUT3, test_data, test_label)

N_train = int(len(train_data))
N_val = int(len(test_data))

# randomize index from all_data
index_train = list(range(0, N_train))
index_val = list(range(0, N_val))
# random.shuffle(index_train)
# random.shuffle(index_val)

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

predict_histogram = np.zeros((LABEL_NUM, LABEL_NUM))
temporary_histogram = np.zeros((LABEL_NUM, LABEL_NUM))

tp_ = np.zeros([LABEL_NUM])
tn_ = np.zeros([LABEL_NUM])
fp_ = np.zeros([LABEL_NUM])
fn_ = np.zeros([LABEL_NUM])

_acc = np.zeros([LABEL_NUM])
P = np.zeros([LABEL_NUM])
R = np.zeros([LABEL_NUM])
# p = np.zeros([AVAILABLE_CLASSIFIER])

# classifier = [c0, c1, c2, c3]
classifier = [c0, c1]

print("Learning start")
start = time.time()
train_start = 0
for epoch in range(0, G_EPOCH):
    for i, batch_num in enumerate(tbatch_list):
        in_list, in_label = [], []
        p = []
        next_batch = train_start + batch_num
        F.data_set(train_start, next_batch, index_train, train_data, in_list)
        F.data_set(train_start, next_batch, index_train, train_label, in_label)
        batch_xs, batch_ys = in_list, in_label
        train_start = next_batch

        c = 0
        while c < AVAILABLE_CLASSIFIER:
            print("learning in classifier %d" % c)
            z = classifier[c].learn_parameters(batch_xs, batch_ys)
            p.append(z)
            c += 1

        print(p)
        print("\n-----")
        print("go to next batch")
        print("-----\n")

print("learned all classifier!")

# classify block
# ----------------------------------------------------------------------------------------------------------------------
# test_start = 0
# print("classifying start!\n")
# start = time.time()
# for i, batch_num in enumerate(vbatch_list):
#     in_list, in_label, batch_predict = [], [], []
#     next_batch = test_start + batch_num
#     F.data_set(test_start, next_batch, index_val, test_data, in_list)
#     F.data_set(test_start, next_batch, index_val, test_label, in_label)
#     batch_xs, batch_ys = in_list, in_label
#     test_start = next_batch
#
#     c, acc = 0, 0.0
#     print(i)
#     if i > 100:
#         break
#
#     # classification
#     while -1 < c < AVAILABLE_CLASSIFIER:
#         print("operating in classifier %d" % c)
#         temporary_histogram = np.zeros_like(temporary_histogram)
#         batch_predict = classifier[c].prediction(batch_xs, batch_ys)
#         # batch_predict = classifier[c].prediction(test_data, test_label)  # use all data
#
#         # acc = F.calc_acc_(batch_predict, batch_ys)
#         batch_predict_max_list = np.max(batch_predict, 1)
#         pre_ave = float(np.sum(batch_predict_max_list) / batch_num)
#
#         if pre_ave < PREDICT_THRESHOLD:
#             print("Predict Average per a Batch in C%d: %0.20f" % (c, pre_ave))
#             c += 1
#             # c -= 10
#             # print(c)
#         else:
#             c -= 10
#
#         F.update_matrix(batch_predict, batch_ys, temporary_histogram)
#         # F.update_matrix(batch_predict, test_label, temporary_matrix)  # use all data
#
#     # add tmp matrix to predict matrix
#     predict_histogram = np.add(predict_histogram, temporary_histogram)
#     temporary_matrix = np.zeros_like(temporary_histogram)
#
#     # alert the degree of risk
#     if c > 0:
#         print("!!---------------------------------------!!")
#         print("The degree of risk in batch %d is level %d!" % (i, (c + 1)))
#         print("!!---------------------------------------!!")
#
#     if (i + 1) % 10 == 0:
#         pre_sum = np.sum(predict_histogram, 1)
#         gt_sum = np.sum(predict_histogram, 0)
#         for j in range(0, LABEL_NUM):
#             tp_[j] = np.diag(predict_histogram)[j]
#             fp_[j] = pre_sum[j] - tp_[j]
#             fn_[j] = gt_sum[j] - tp_[j]
#             tn_[j] = np.ndarray.sum(predict_histogram) - (tp_[j] + fp_[j] + fn_[j])
#
#             print(predict_histogram[j])
#             P[j] = tp_[j] / (tp_[j] + fp_[j] + 1)
#             R[j] = tp_[j] / (tp_[j] + fn_[j] + 1)
#             _acc = (tp_[j] + tn_[j]) / (tp_[j] + tn_[j] + fp_[j] + fn_[j])
#
#         precision = float(np.sum(P) / LABEL_NUM)
#         recall = float(np.sum(R) / LABEL_NUM)
#         f_measure = 2 * precision * recall / (precision + recall)
#         accuracy = float(np.sum(_acc) / LABEL_NUM)
#
#         print("--- Middle Result (classified %d batches) ---" % (i + 1))
#         print("Precision        : %0.15f" % precision)
#         print("Recall           : %0.15f" % recall)
#         print("F Measure        : %0.15f" % f_measure)
#         print("Accuracy         : %0.15f\n" % accuracy)
#
#
# # finish classifying all data
# tp_ = np.zeros([LABEL_NUM])
# tn_ = np.zeros([LABEL_NUM])
# fp_ = np.zeros([LABEL_NUM])
# fn_ = np.zeros([LABEL_NUM])
#
# # count the number of each status per 1 class
# pre_sum = np.sum(predict_histogram, 1)
# gt_sum = np.sum(predict_histogram, 0)
# for i in range(0, LABEL_NUM):
#     tp_[j] = np.diag(predict_histogram)[j]
#     fp_[j] = pre_sum[j] - tp_[j]
#     fn_[j] = gt_sum[j] - tp_[j]
#     tn_[j] = np.ndarray.sum(predict_histogram) - (tp_[j] + fp_[j] + fn_[j])
#
#     print(predict_histogram[j])
#     P[j] = tp_[j] / (tp_[j] + fp_[j] + 1)
#     R[j] = tp_[j] / (tp_[j] + fn_[j] + 1)
#     _acc[j] = (tp_[j] + tn_[j]) / (tp_[j] + tn_[j] + fp_[j] + fn_[j])
#
# precision = float(np.sum(P) / LABEL_NUM)
# recall = float(np.sum(R) / LABEL_NUM)
# f_measure = 2 * precision * recall / (precision + recall)
# accuracy = float(np.sum(_acc) / LABEL_NUM)
#
# print("--- Total Result ---")
# print("Precision        : %0.15f" % precision)
# print("Recall           : %0.15f" % recall)
# print("F Measure        : %0.15f" % f_measure)
# print("Accuracy         : %0.15f\n" % accuracy)
#
# print("\nFinish All classification!!\n")
# ----------------------------------------------------------------------------------------------------------------------
