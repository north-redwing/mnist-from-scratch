# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from layers import Neuralnet
import time
import sys
from func import load_mnist
from collections import OrderedDict


def compare(FILE_NAME, debug=True, save=True):
    noise_per_list = [i for i in range(26)]
    load_params_dict_list = []

# ----- 各パラメータの読み込み -----
    for i in range(6):
        params_dict = OrderedDict()
        LOAD_W1 = "../data/arr/params/"+FILE_NAME+"_weight1_noise"+str(5*i)+"%.npy"
        LOAD_B1 = "../data/arr/params/"+FILE_NAME+"_bias1_noise"+str(5*i)+"%.npy"
        LOAD_W2 = "../data/arr/params/"+FILE_NAME+"_weight2_noise"+str(5*i)+"%.npy"
        LOAD_B2 = "../data/arr/params/"+FILE_NAME+"_bias2_noise"+str(5*i)+"%.npy"

        params_dict['W1'] = np.load(LOAD_W1)
        params_dict['B1'] = np.load(LOAD_B1)
        params_dict['W2'] = np.load(LOAD_W2)
        params_dict['B2'] = np.load(LOAD_B2)

        load_params_dict_list.append(params_dict)

    if debug: print("Finish loading parameters")

# ----- test_Xのノイズ耐性の調査 ----- 
    test_acc_list = []

    for i, load_params_dict in enumerate(load_params_dict_list):
        model = Neuralnet(load_params_dict=load_params_dict)
        test_acc_list_each = []
            
        for noise_per in noise_per_list:
            _, _, test_X, test_T = load_mnist(noise_per)
            test_acc = model.acc(test_X, test_T)
            test_acc_list_each.append(test_acc)

        test_acc_list.append(test_acc_list_each)
        if debug: print("{}% noise is written".format(i*5))

    # 描画
    plt.figure()
    graph = []
    graph_name = []
    for i, test_acc_list_each in enumerate(test_acc_list):
        p, = plt.plot(test_acc_list_each)
        graph.append(p)
        graph_name.append("noise "+str(5*i)+"%")
    plt.legend(graph, graph_name)
    plt.grid()
    plt.xlabel("test noise per")
    plt.ylabel("accuracy")
    plt.title("variation of accuracy with noise per")
    SAVE_DIR = "../data/output/"+FILE_NAME+"_compare.jpg"
    plt.savefig(SAVE_DIR)


if __name__ == "__main__":
    argc = len(sys.argv)
    FILE_NAME = sys.argv[1]
    option = sys.argv[2:argc]
    
    debug = ("-debug" in option)
    save = ("-save" in option)

    compare(FILE_NAME, debug=debug, save=save)
