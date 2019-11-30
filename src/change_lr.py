#!/Users/YujiNarita/.pyenv/shims/python3
# -*- coding: utf-8 -*-

from deep_learning import learning
import sys
import matplotlib.pyplot as plt


def change_lr(debug, save):
    train_acc_list = []
    test_accs_list = []
    lr_list = [i*0.01 for i in range(5, 105, 5)]

    for lr in lr_list:
        FILE_NAME = "lr"+str(lr)
        train_acc, test_acc = learning(4000, lr, FILE_NAME, 100, 0, debug, save)
        train_acc_list.append(train_acc)
        test_accs_list.append(test_acc)

    if save:
        plt.figure()
        train_graph, = plt.plot(lr_list, train_acc_list)
        test_graph, = plt.plot(lr_list, test_accs_list)
        plt.legend([train_graph, test_graph], ["train acc", "test acc"])
        plt.xlabel("lr")
        plt.ylabel("accuracy")
        plt.title("variation of accuracy with lr")
        plt.grid()
        SAVE_NAME = "../data/output/change_lr_acc.jpg"
        plt.savefig(SAVE_NAME)

if __name__ == "__main__":
    options = sys.argv[1:]

    debug = ("-debug" in options)
    save = ("-save" in options)

    change_lr(True, True)
