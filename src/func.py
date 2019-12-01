# -*- coding: utf-8 -*-

import numpy as np


def softmax(Z):
    if Z.ndim == 1:
        const = np.nanmax(Z)
        return np.exp(Z-const)/np.sum(np.exp(Z-const))
    elif Z.ndim == 2:
        Z = (Z.T-np.max(Z,axis=1)).T
        Y = np.exp(Z)/np.sum(np.exp(Z), axis=1).reshape(Z.shape[0],-1)
        return Y
    else:
        raise ValueError
        print("Dimension Error !")


def cross_entropy_error(Y, T):
    if Y.ndim == 1:
        T = T.reshape(1, T.size)
        Y = Y.reshape(1, Y.size)
    
    batch_size = Y.shape[0]
    delta = 1e-7
    
    return -np.sum(T*np.log(Y+delta)) / batch_size


# データの読み込み
def load_mnist(noise_per=0):
    TRAIN_LABEL_DIR = "../data/arr/train/train_T.npy"
    TEST_LABEL_DIR = "../data/arr/test/test_T.npy"
    TRAIN_IMG_DIR = "../data/arr/train/train_X_noise"+str(noise_per)+"%.npy"
    TEST_IMG_DIR = "../data/arr/test/test_X_noise"+str(noise_per)+"%.npy"

    # 教師データの読み込み
    train_X = np.load(TRAIN_IMG_DIR)
    train_T = np.load(TRAIN_LABEL_DIR)

    # 評価データの読み込み
    test_X = np.load(TEST_IMG_DIR)
    test_T = np.load(TEST_LABEL_DIR)
    
    return train_X, train_T, test_X, test_T


# 混同行列の実装
def confusion_matrix(true, pred, label_size=10):
    confusion_matrix = np.zeros((label_size, label_size)).astype(np.int)

    for t, p in zip(true, pred):
            confusion_matrix[t][p] += 1
    
    return confusion_matrix


if __name__ == "__main__":
    pass
