# coding:utf-8

# K-meansアルゴリズム

import numpy as np
import sys
import setting_classifier_5_class as setting
from pylab import *

K = 2  # クラスターの数（固定）


def scale(X):
    """データ行列Xを属性ごとに標準化したデータを返す"""
    # 属性の数（=列の数）
    col = X.shape[1]

    # 属性ごとに平均値と標準偏差を計算
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    # 属性ごとデータを標準化
    for i in range(col):
        X[:, i] = (X[:, i] - mu[i]) / sigma[i]

    return X


def J(X, mean, r):
    """K-meansの目的関数（最小化を目指す）"""
    sum = 0.0
    for n in range(len(X)):
        temp = 0.0
        for k in range(K):
            temp += r[n, k] * np.linalg.norm(X[n] - mean[k]) ** 2
        sum += temp
    return sum


if __name__ == "__main__":
    # 訓練データをロード
    # data = np.genfromtxt("faithful.txt")
    data = np.genfromtxt("data/kdd_train_1000.csv", delimiter=",")
    # data = []
    X = []
    label = []
    setting.data_read("data/kdd_train_1000.csv", X, label)
    ATTR = len(X[0])

    # X = data[:, 0:2]
    # X = []
    # label = []
    # setting.data_read("data/kdd_train_1000.csv", X, label)
    # X = scale(X)
    N = len(X)

    # 平均を初期化
    # mean = np.random.rand(K, 2)
    mean = np.random.rand(K, ATTR)

    # クラスタ割当変数の配列を用意
    r = np.zeros((N, K))
    r[:, 0] = 1

    # 目的関数の初期値を計算
    target = J(X, mean, r)

    turn = 0
    while True:
        print(turn, target)

        # E-step : 現在のパラメータmeanを使って、クラスタ割当変数rを計算 (9.2)
        for n in range(N):
            idx = -1
            min = sys.maxsize
            for k in range(K):
                temp = np.linalg.norm(X[n] - mean[k]) ** 2
                if temp < min:
                    idx = k
                    min = temp
            for k in range(K):
                r[n, k] = 1 if k == idx else 0

        # M-step : 現在のクラスタ割当変数rを用いてパラメータmeanを更新 (9.4)
        for k in range(K):
            numerator = 0.0
            # denominator = 0.0
            numerator = []
            denominator = []
            for n in range(N):
                # numerator += r[n, k] * X[n]
                # denominator += r[n, k]
                numerator.append(r[n, k] * X[n])
                denominator.append(r[n, k])
            for n in range(ATTR):
                mean[k] = numerator[n, k] / float(denominator[n, k])

        # 収束判定
        new_target = J(X, mean, r)
        diff = target - new_target
        if diff < 0.01:
            break
        target = new_target
        turn += 1

    # 訓練データを描画
    color = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w']  # K < 8と想定
    for n in range(N):
        for k in range(K):
            if r[n, k] == 1:
                c = color[k]
        scatter(X[n, 0], X[n, 1], c=c, marker='o')

    # クラスタの平均を描画
    for k in range(K):
        scatter(mean[k, 0], mean[k, 1], s=120, c='y', marker='s')

    xlim(-2.5, 2.5)
    ylim(-2.5, 2.5)
    show()
