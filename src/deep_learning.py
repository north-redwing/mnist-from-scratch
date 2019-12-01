# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import pprint
from layers import Neuralnet
from func import load_mnist, confusion_matrix


# 学習
def learning(iters_num, lr, FILE_NAME, neuron_num, noise_per, debug, save):
    if debug: start = time.time()
    
# ----- データセットの読み込み -----
    train_X, train_T, test_X, test_T = load_mnist(noise_per)
    
    model = Neuralnet(neuron_num=neuron_num)
    train_size = train_X.shape[0]
    batch_size = 100
    
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    tmp_train_loss_list = []
    tmp_train_acc_list = []
    tmp_test_acc_list = []
    
    for num in range(iters_num):
    # ----- batch処理 -----
        batch_mask = np.random.choice(train_size, batch_size)
        X_batch = train_X[batch_mask]
        T_batch = train_T[batch_mask]
        
    # ----- 勾配の算出 -----
        grads = model.grad(X_batch, T_batch)
        
    # ----- loss, accの算出 ------
        train_loss = model.loss(X_batch, T_batch)
        tmp_train_loss_list.append(train_loss)
        
        train_acc = model.acc(X_batch, T_batch)
        tmp_train_acc_list.append(train_acc)
        
        test_acc = model.acc(test_X, test_T)
        tmp_test_acc_list.append(test_acc)

    # ----- 10回の平均で計算する -----
        if num % 10 == 0:    
            avg_train_acc = np.mean(tmp_train_acc_list)
            avg_test_acc = np.mean(tmp_test_acc_list)
            avg_train_loss = np.mean(tmp_train_loss_list)

            train_acc_list.append(avg_train_acc)
            test_acc_list.append(avg_test_acc)
            train_loss_list.append(avg_train_loss)
                                                          
            tmp_train_loss_list = []
            tmp_train_acc_list = []
            tmp_test_acc_list = []

    # ----- 学習 -----
        for parameter in ('W1', 'B1', 'W2', 'B2'):
            model.parameters[parameter] -= lr * grads[parameter] 

    # ----- option -----
        if debug:
            percentage = int(100*(num+1)/iters_num)
            print("{}/{} : {}%".format(num+1, iters_num, percentage)) 
            print("train acc : {}".format(train_acc))
            print("test acc : {}".format(test_acc))
            print("train loss : {}".format(train_loss))
            print("\n", end="")

        if save:
            ACC_LOSS_FILE = "../data/output/"+FILE_NAME+"_noise"+str(noise_per)+"%_acc_loss.txt"
            with open(ACC_LOSS_FILE, mode='a') as f:
                f.write("{}/{} : {}%\n".format(num+1, iters_num, percentage))
                f.write("train acc : {}\n".format(train_acc))
                f.write("test acc : {}\n".format(test_acc))
                f.write("train loss : {}\n".format(train_loss))
                f.write("\n")
    
    if save:
    # ----- 正答率のグラフの保存 -----
        plt.figure()
        p1, = plt.plot(train_acc_list)
        p2, = plt.plot(test_acc_list)
        plt.legend([p1, p2], ["train acc noise"+str(noise_per)+"%", "test acc noise"+str(noise_per)+"%"])
        plt.title("accuracy graph")
        plt.xlabel("iterations/10")
        plt.ylabel("accuracy")
        plt.grid()
        SAVE_ACC_DIR = "../data/output/"+FILE_NAME+"_noise"+str(noise_per)+"%_acc_graph.jpg"
        plt.savefig(SAVE_ACC_DIR)

    # ----- 損失関数のグラフの保存 -----
        plt.figure()
        p, = plt.plot(train_loss_list)
        plt.legend([p], ["train loss noise "+str(noise_per)+"%"])
        plt.xlabel("iterations/10")
        plt.ylabel("loss")
        plt.grid()
        plt.title("loss graph")
        SAVE_LOSS_DIR = "../data/output/"+FILE_NAME+"_noise"+str(noise_per)+"%_loss_graph.jpg"
        plt.savefig(SAVE_LOSS_DIR)
        
    # ----- 混同行列で可視化 -----    
        np.set_printoptions(threshold=100)
        pred = model.pred_label(test_X)

        # one-hot から vectorへ変換
        true = np.argmax(test_T, axis=1)
        CM = confusion_matrix(true, pred)
        CM_FILE = "../data/output/"+FILE_NAME+"_noise"+str(noise_per)+"%_cm.txt"
        np.savetxt(CM_FILE, CM, fmt='%d', delimiter=',')
        pprint.pprint(CM)

        #正規化
        min = CM.min(axis=1, keepdims=True)
        max = CM.max(axis=1, keepdims=True)
        CM = (CM-min)/(max-min)
        CM = 255*CM

        # 描画
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("confusion matrix")
        img = ax.imshow(CM, cmap=plt.cm.jet)
        cbar = fig.colorbar(img, ax=ax)
        SAVE_CM_DIR = "../data/output/"+FILE_NAME+"_noise"+str(noise_per)+"%_cm_graph.jpg"
        plt.savefig(SAVE_CM_DIR)

    # ----- parameterの保存 -----
        WEIGHT_FILE1 = "../data/arr/params/"+FILE_NAME+"_weight1_noise"+str(noise_per)+"%.npy"
        np.save(WEIGHT_FILE1, model.parameters['W1'])
                                                                  
        BIAS_FILE1 = "../data/arr/params/"+FILE_NAME+"_bias1_noise"+str(noise_per)+"%.npy"
        np.save(BIAS_FILE1, model.parameters['B1'])
                                                                      
        WEIGHT_FILE2 = "../data/arr/params/"+FILE_NAME+"_weight2_noise"+str(noise_per)+"%.npy"
        np.save(WEIGHT_FILE2, model.parameters['W2'])
                                                                  
        BIAS_FILE2 = "../data/arr/params/"+FILE_NAME+"_bias2_noise"+str(noise_per)+"%.npy"
        np.save(BIAS_FILE2, model.parameters['B2'])

    if debug:
        elapsed_time = time.time()-start
        print("elapsed time : {}".format(elapsed_time))

    return train_acc_list[-1], test_acc_list[-1]


if __name__ == "__main__":
    FILE_NAME = sys.argv[1]
    iters_num = int(sys.argv[2])
    lr = float(sys.argv[3])
    neuron_num = int(sys.argv[4])
    noise_per = int(sys.argv[5])
    option = sys.argv[6:]

    debug = ("-debug" in option) 
    save = ("-save" in option)

    learning(iters_num, lr, FILE_NAME, neuron_num, noise_per, debug, save)
