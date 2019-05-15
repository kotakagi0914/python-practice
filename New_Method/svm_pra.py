# coding:utf-8

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import itertools
import operator

from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics
import setting_for_svm as setting
import numpy as np

INPUT = "data/kdd_train_1000.csv"
INPUT1 = "data/kdd_train_20Percent.csv"


def _print_result(percentage, C_n, gamma_n):
    """ 正答率とそれに使われたパラメータを出力する """
    msg = '正答率 {percentage:.2f}% C=2^{C} gamma=2^{gamma}'.format(
        percentage=percentage,
        C=C_n,
        gamma=gamma_n,
    )
    print(msg)


def main():
    # 数値画像のデータを読み込む
    # digits = datasets.load_digits()
    # X = digits.data
    # y = digits.target

    # data pre-process
    train_data, train_label = [], []
    # test_data, test_label = [], []

    setting.data_read(INPUT, train_data, train_label)
    # setting.data_read(INPUT1, test_data, test_label)
    # X = list(train_data)
    # y = list(test_data)

    X = np.array(train_data)
    y = np.array(train_label)
    # パラメータ C の候補 (2^-5 ~ 2^5)
    Cs = [(2 ** i, i) for i in range(-5, 5)]
    # パラメータ gamma の候補 (2^-12 ~ 2^-5)
    gammas = [(2 ** i, i) for i in range(-12, -5)]  # 2^-12 ~ 2^-5
    # 上記のパラメータが取りうる組み合わせ (デカルト積) を作る
    parameters = itertools.product(Cs, gammas)

    results = []
    # 各組み合わせで正答率にどういった変化があるかを調べていく
    for (C, C_n), (gamma, gamma_n) in parameters:
        scores = []
        # 正答率は K-fold 交差検定 (10 分割) で計算する
        kfold = cross_validation.KFold(len(X), n_folds=5)
        # 教師信号を学習用と検証用に分割する
        for train, test in kfold:
            # 前述したパラメータを使って SVM (RBF カーネル) を初期化する
            clf = svm.SVC(C=C, gamma=gamma)
            # 学習する
            clf.fit(X[train], y.ravel()[train])
            # 検証する
            score = metrics.accuracy_score(clf.predict(X[test]), y[test])
            scores.append(score)
        # 正答率をパーセンテージにしてパラメータと共に表示する
        percentage = (sum(scores) / len(scores)) * 100
        results.append((percentage, C_n, gamma_n))
        _print_result(*results[-1])

    # 正答率の最も高かったパラメータを出力する
    sorted_result = sorted(results, key=operator.itemgetter(0), reverse=True)
    print('--- 最適なパラメータ ---')
    _print_result(*sorted_result[0])

if __name__ == '__main__':
    main()
