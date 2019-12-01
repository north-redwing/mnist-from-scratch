# -*- coding: utf-8 -*-

from func import load_mnist
import numpy as np


def make_noise(width=28, height=28):
    train_X, train_T, test_X, test_T = load_mnist()
    data_size = width*height

    noise_per_list = [i for i in range(1, 26)]
    for SAVE_DIR in ["../data/arr/train/train_X_noise", "../data/arr/test/test_X_noise"]:

        for noise_per in noise_per_list:
            noise_num = int(0.01*noise_per*data_size)

            if "train" in SAVE_DIR: noise_X = train_X.copy()
            elif "test" in SAVE_DIR: noise_X = test_X.copy()

            for each_X in noise_X:
                noise_mask = np.random.randint(0, data_size, noise_num)

                for noise_idx in noise_mask:
                    each_X[noise_idx] = np.random.random()

            save_dir = SAVE_DIR+str(noise_per)+"%.npy"

            np.save(save_dir, noise_X)
            print("noise {}% is written.".format(noise_per))

if __name__ == "__main__":
    make_noise()
