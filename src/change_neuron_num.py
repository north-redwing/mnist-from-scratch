# -*- coding: utf-8 -*-

from deep_learning import learning
import matplotlib.pyplot as plt
import sys


def change_neuron_num(debug, save):
    train_acc_list = []
    test_acc_list = []
    neuron_num_list = [10, 100, 400, 784, 1000]

    for neuron_num in neuron_num_list:
        FILE_NAME = "neuron_num"+str(neuron_num)
        train_acc, test_acc = learning(4000, 0.1, FILE_NAME, neuron_num, 0, debug, save)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

    if save:
        plt.figure()
        plt.xscale("log")
        train_graph, = plt.plot(neuron_num_list, train_acc_list)
        test_graph, = plt.plot(neuron_num_list, test_acc_list)
        plt.title("variation of accuracy with neuron num")
        plt.xlabel("neuron num")
        plt.ylabel("accuracy")
        plt.grid()
        plt.legend([train_graph, test_graph], ["train acc", "test acc"])
        SAVE_NAME = "../data/output/change_neuron_num.jpg"
        plt.savefig(SAVE_NAME)

if __name__ == "__main__":
    option = sys.argv[1:]

    debug = ("-debug" in option) 
    save = ("-save" in option)

    change_neuron_num(debug, save)
