# coding: utf-8

# this is the main file of my new method.

# classify traffic in each stages if there is any risk in it.
# there are 3 alert level.

import numpy as np
import time
# import random
# import setting_classifier
import setting_classifier_5_class

import functions as F
import classifier_smr as c0
import classifier1 as c1
import classifier2 as c2
import classifier3 as c3

LABEL_NUM = 5
PREDICT_THRESHOLD = 0.999
AVAILABLE_CLASSIFIER = 1

INPUT = "data/kdd_train_1000.csv"
INPUT1 = "data/kdd_train_20Percent.csv"
INPUT2 = "data/kdd_train_39_class.csv"
INPUT3 = "data/kdd_test_39_class.csv"

# data pre-process
test_data, test_label = [], []
# setting_classifier.data_read(INPUT3, test_data, test_label)
setting_classifier_5_class.data_read(INPUT3, test_data, test_label)

N_val = int(len(test_data))

# randomize index from all_data
index_val = list(range(0, N_val))
# random.shuffle(index_val)

predict_histogram = np.zeros((LABEL_NUM, LABEL_NUM))
temporary_histogram = np.zeros((LABEL_NUM, LABEL_NUM))

tp_ = np.zeros([LABEL_NUM])
tn_ = np.zeros([LABEL_NUM])
fp_ = np.zeros([LABEL_NUM])
fn_ = np.zeros([LABEL_NUM])

_acc = np.zeros([LABEL_NUM])
P = np.zeros([LABEL_NUM])
R = np.zeros([LABEL_NUM])

classifier = [c0, c1, c2, c3]
# classifier = [c1, c2, c3]

test_start = 0
print("classifying start!\n")
start = time.time()
for i, line_data in enumerate(test_data):
    c, acc = 0, 0.0
    line_label = test_label[i]
    print(i)
    if i > 100:
        break

    # classification
    while -1 < c < AVAILABLE_CLASSIFIER:
        print("operating in classifier %d" % c)
        temporary_histogram = np.zeros_like(temporary_histogram)
        line_predict = classifier[c].prediction(line_data, line_label)

        pre_rate = float(np.argmax(line_predict))

        if pre_rate < PREDICT_THRESHOLD:
            print("Predict Probability in C%d: %0.20f" % (c, pre_rate))
            c += 1
            # c -= 10
        else:
            c -= 10

        F.update_matrix(line_predict, line_label, temporary_histogram)

    # add tmp matrix to predict matrix
    predict_histogram = np.add(predict_histogram, temporary_histogram)
    temporary_matrix = np.zeros_like(temporary_histogram)

    # alert the degree of risk
    if c > 0:
        print("!!---------------------------------------!!")
        print("The degree of risk in batch %d is level %d!" % (i, (c + 1)))
        print("!!---------------------------------------!!")

    if (i + 1) % 10 == 0:
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
            _acc = (tp_[j] + tn_[j]) / (tp_[j] + tn_[j] + fp_[j] + fn_[j])

        precision = float(np.sum(P) / LABEL_NUM)
        recall = float(np.sum(R) / LABEL_NUM)
        f_measure = 2 * precision * recall / (precision + recall)
        accuracy = float(np.sum(_acc) / LABEL_NUM)

        print("--- Middle Result (classified %d batches) ---" % (i + 1))
        print("Precision        : %0.15f" % precision)
        print("Recall           : %0.15f" % recall)
        print("F Measure        : %0.15f" % f_measure)
        print("Accuracy         : %0.15f\n" % accuracy)


# finish classifying all data
tp_ = np.zeros([LABEL_NUM])
tn_ = np.zeros([LABEL_NUM])
fp_ = np.zeros([LABEL_NUM])
fn_ = np.zeros([LABEL_NUM])

# count the number of each status per 1 class
pre_sum = np.sum(predict_histogram, 1)
gt_sum = np.sum(predict_histogram, 0)
for i in range(0, LABEL_NUM):
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

print("--- Total Result ---")
print("Precision        : %0.15f" % precision)
print("Recall           : %0.15f" % recall)
print("F Measure        : %0.15f" % f_measure)
print("Accuracy         : %0.15f\n" % accuracy)

print("\nFinish All classification!!\n")
