#!/Users/YujiNarita/.pyenv/shims/python3
# -*- coding: utf-8 -*-

import numpy as np
import gzip
from os import chdir


def load_label(one_hot=True, label_num=10):
    filename_lst = ['train-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

    for filename in filename_lst:
        filepath = "../data/ubyte/"+filename

        with gzip.open(filepath, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)

        if one_hot:
            labels = np.eye(label_num)[labels]

        if "train" in filename: savepath = "../data/arr/train/train_T.npy"
        else: savepath = "../data/arr/test/test_T.npy"

        np.save(savepath, labels)


def load_image():
    filename_lst = ['train-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    for filename in filename_lst:
        filepath = "../data/ubyte/"+filename

        with gzip.open(filepath, 'rb') as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16)

        images = images.reshape(-1, 28*28)

        images = images/255

        if "train" in filename: savepath = "../data/arr/train/train_X_noise0%.npy"
        else: savepath = "../data/arr/test/test_X_noise0%.npy"

        np.save(savepath, images)

if __name__ == "__main__":
    load_label()
    load_image()
