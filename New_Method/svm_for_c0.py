# coding:utf-8

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn import metrics
import setting_for_svm as setting
import numpy as np

INPUT = "data/kdd_train_1000.csv"
INPUT1 = "data/kdd_train_20Percent.csv"
INPUT2 = "data/kdd_train_39_class.csv"
INPUT3 = "data/kdd_test_39_class.csv"


def main():
    # digits = datasets.load_digits()
    # X = digits.data
    # y = digits.target

    # data pre-process
    train_data, train_label = [], []
    test_data, test_label = [], []

    setting.data_read(INPUT2, train_data, train_label)
    setting.data_read(INPUT3, test_data, test_label)

    X = np.array(train_data)
    y = np.array(train_label)
    X_ = np.array(test_data)
    y_ = np.array(test_label)

    scores, scores1 = [], []
    train = np.array([], np.int32)
    test = np.array([], np.int32)
    # # K-fold 交差検証でアルゴリズムの汎化性能を調べる
    # kfold = cross_validation.KFold(len(X), n_folds=10)
    # for train, test in kfold:
    #     # デフォルトのカーネルは rbf になっている
    #     clf = svm.SVC(C=2**4, gamma=2**-6)
    #     # 訓練データで学習する
    #     clf.fit(X[train], y.ravel()[train])
    #     # テストデータの正答率を調べる
    #     score = metrics.accuracy_score(clf.predict(X_[test]), y_[test])
    #     scores.append(score)

    # K-fold 交差検証でアルゴリズムの汎化性能を調べる
    # デフォルトのカーネルは rbf になっている
    clf = svm.SVC(C=2 ** 4, gamma=2 ** -6)
    classifier = OneVsRestClassifier(clf)
    # 訓練データで学習する
    for i in range(len(train_data)):
        train = np.append(train, i)
    train.astype(np.int32)
    # for i in range(10):
    #     print(i)
    clf.fit(X[train], y.ravel()[train])
    classifier.fit(X[train], y.ravel()[train])
    # テストデータの正答率を調べる
    for i in range(len(test_data)):
        test = np.append(test, i)
    score = metrics.accuracy_score(clf.predict(X_[test]), y_[test])
    score1 = metrics.accuracy_score(classifier.predict(X_[test]), y_[test])
    predict_ = clf.predict(X_[test])
    print(predict_)
    scores.append(score)
    scores1.append(score1)

    # 最終的な正答率を出す
    accuracy = (sum(scores) / len(scores)) * 100
    msg = '正答率: {accuracy:.2f}%'.format(accuracy=accuracy)
    print(msg)

    accuracy_ = (sum(scores1) / len(scores1)) * 100
    msg_ = '正答率: {accuracy:.2f}%'.format(accuracy=accuracy_)
    print(msg_)


if __name__ == '__main__':
    main()
