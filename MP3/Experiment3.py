import sys
from gen import gen
import numpy as np
from Algorithms import *
import matplotlib.pyplot as plt


def random_10(input_data,input_label):
    N = len(input_label)
    index = np.arange(N)
    np.random.shuffle(index)
    train_idx = index[:int(N/10)]
    test_idx = index[int(N/10):int(N/5)]
    out_train_data = input_data[train_idx]
    out_train_label = input_label[train_idx]
    out_test_data = input_data[test_idx]
    out_test_label = input_label[test_idx]
    return (out_train_data, out_train_label, out_test_data, out_test_label)


def Get_Accuracy(true_label, est_label):
    N = len(true_label)
    Wrong = 0
    for i in range(N):
        if true_label[i] != est_label[i]:
            Wrong += 1
    Acc = 1 - float(Wrong) / N
    return Acc

    
if __name__ == "__main__":
    l = 10
    all_m = [100, 500, 1000]
    n = 1000
    num_iter = 20
    Perceptron1_Acc = []
    Perceptron2_Acc = []
    Winnow1_Acc = []
    Winnow2_Acc = []
    AdaGrad_Acc = []

    for m in all_m:
        print "m =",m
        (all_train_label, all_train_data) = gen(l, m, n, 50000, True)
        (all_test_label, all_test_data) = gen(l, m, n, 10000, False)
        (train_data, train_label, test_data, test_label) = random_10(all_train_data, all_train_label)

        ######### Tuning Parameters #########################################
        ######### Perceptron1: LR = 1, Margin = 0
        ######### Perceptron2: LR = [1.5, 0.25, 0.03, 0.005, 0.001], Margin = 1
        ######### Winnow1: Alpha = [1.1, 1.01, 1.005, 1.0005, 1.0001], Margin = 0
        ######### Winnow2: Alpha = [1.1, 1.01, 1.005, 1.0005, 1.0001], Margin = [2.0, 0.3, 0.04, 0.006, 0.001]
        ######### AdaGrad: LR = [1.5, 0.25, 0.03, 0.005, 0.001]

        Per2_LR = [1.5, 0.25, 0.03, 0.005, 0.001]
        Win_Alpha = [1.1, 1.01, 1.005, 1.0005, 1.0001]
        Win2_Margin = [2.0, 0.3, 0.04, 0.006, 0.001]
        Ada_LR = [1.5, 0.25, 0.03, 0.005, 0.001]

 
        ### 1: Perceptron1: No tuning Parameters #######################
        Perceptron1 = Perceptron(1, 0, all_train_data, n)
        i = 0
        while i < num_iter:
            Perceptron1.train(all_train_data, all_train_label)
            i+=1
        est_label = Perceptron1.predict(all_test_data)
        temp_Acc = Get_Accuracy(all_test_label, est_label)
        Perceptron1_Acc.append(temp_Acc)

        print "Perceptron1: No Margin"
        print "LR = 1, Margin = 0"
        print "m =", m, "Accuracy = ", temp_Acc

         
        ### 2: Perceptron2: Tuning LR #################################
        Acc = []

        for i in range(5):
            temp_LR = Per2_LR[i]
            Perceptron2 = Perceptron(temp_LR, 1, train_data, n)
            for j in range(20):
                #print "j =", j
                Perceptron2.train(train_data, train_label)
            est_label = Perceptron2.predict(test_data)
            temp_Acc = Get_Accuracy(test_label, est_label)
            Acc.append(temp_Acc)

        #print Acc_n1
        Best_LR_idx = np.argmax(Acc)
        Best_LR = Per2_LR[Best_LR_idx]
        Perceptron2 = Perceptron(Best_LR, 1, all_train_data, n)
        i = 0
        while i < num_iter:
            #print "i =",i
            Perceptron2.train(all_train_data, all_train_label)
            i += 1
        est_label = Perceptron2.predict(all_test_data)
        temp_Acc = Get_Accuracy(all_test_label, est_label)
        Perceptron2_Acc.append(temp_Acc)

        print "Percptron2: with Margin"
        print "m =", m ,"LR =", Best_LR, ", Margin = 1,", "Accuracy =", temp_Acc
        
        
        ### 3: Winnow1: Tuning Alpha #################################
        Acc = []

        for i in range(5):
            temp_alpha = Win_Alpha[i]
            Winnow1 = Winnow(temp_alpha, 0, train_data, n)
            for j in range(20):
                Winnow1.train(train_data, train_label)
            est_label = Winnow1.predict(test_data)
            temp_Acc = Get_Accuracy(test_label, est_label)
            Acc.append(temp_Acc)

        Best_alpha_idx = np.argmax(Acc)
        Best_alpha = Win_Alpha[Best_alpha_idx]
        Winnow1 = Winnow(Best_alpha, 1, all_train_data, n)
        i = 0
        while i < num_iter:
            Winnow1.train(all_train_data, all_train_label)
            i += 1
        est_label = Winnow1.predict(all_test_data)
        temp_Acc = Get_Accuracy(all_test_label, est_label)
        Winnow1_Acc.append(temp_Acc)

        print "Winnow1: w/o Margin"
        print "m =", m, "Alpha =", Best_alpha, ", Margin = 0,", "Accuracy =", temp_Acc


        
        ### 4: Winnow2: Tuning Alpha & Margin #################################
        Acc = []

        for i in range(5):      # select alpha
            temp_alpha = Win_Alpha[i]
            for k in range(5):      # select margin
                temp_margin = Win2_Margin[k]
                Winnow2 = Winnow(temp_alpha, temp_margin, train_data, n)
                for j in range(20):
                    Winnow2.train(train_data, train_label)
                est_label = Winnow2.predict(test_data)
                temp_Acc = Get_Accuracy(test_label, est_label)
                Acc.append(temp_Acc)

        Best_idx = np.argmax(Acc)
        Best_alpha_idx = Best_idx / 5
        Best_margin_idx = Best_idx % 5
        Best_alpha = Win_Alpha[Best_alpha_idx]
        Best_margin = Win2_Margin[Best_margin_idx]
        Winnow2 = Winnow(Best_alpha, Best_margin, all_train_data, n)
        i = 0
        while i < num_iter:
            Winnow2.train(all_train_data, all_train_label)
            i += 1
        est_label = Winnow2.predict(all_test_data)
        temp_Acc = Get_Accuracy(all_test_label, est_label)
        Winnow2_Acc.append(temp_Acc)

        print "Winnow2: with Margin"
        print "m =", m, ": Alpha =", Best_alpha, ", Margin =", Best_margin, "Accuracy =", temp_Acc


        ### 5: AdaGrad: Tuning LR #################################
        Acc = []

        for i in range(5):
            temp_LR = Ada_LR[i]
            AdaGrad1 = AdaGrad(temp_LR, train_data, n)
            AdaGrad1.w = np.zeros(n)
            AdaGrad1.theta = 0
            AdaGrad1.G = np.zeros(n+1)

            for j in range(20):
                print i, j
                AdaGrad1.train(train_data, train_label)
            est_label = AdaGrad1.predict(test_data)
            temp_Acc = Get_Accuracy(test_label, est_label)
            #print temp_Acc_n1
            Acc.append(temp_Acc)

        print Acc
        Best_LR_idx = np.argmax(Acc)
        Best_LR = Ada_LR[Best_LR_idx]
        AdaGrad2 = AdaGrad(Best_LR, all_train_data, n)
        AdaGrad2.w = np.zeros(n)
        AdaGrad2.theta = 0
        AdaGrad2.G = np.zeros(n+1)
        i = 0
        while i < num_iter:
            print i
            AdaGrad2.train(all_train_data, all_train_label)
            i += 1
        est_label = AdaGrad2.predict(all_test_data)
        temp_Acc = Get_Accuracy(all_test_label, est_label)
        AdaGrad_Acc.append(temp_Acc)

        print "AdaGrad:"
        print "m =", m, ": LR =", Best_LR, "Accuracy =", temp_Acc
    


    ####### Result ##############################################
    ####### m = 100: #############################################
    ####### Perceptron1: LR = 1 , Margin = 0, Accuracy =  0.9787
    ####### Perceptron2: LR = 0.03 , Margin = 1, Accuracy = 0.9738
    ####### Winnow1: Alpha = 1.1 , Margin = 0, Accuracy = 0.9705
    ####### Winnow2: Alpha = 1.1 , Margin = 0.3 Accuracy = 0.9655
    ####### AdaGrad: LR = 0.25 Accuracy = 0.9992
    ####### m = 500: #############################################
    ####### Perceptron1: LR = 1 , Margin = 0, Accuracy =  0.8655
    ####### Perceptron2: LR = 0.25 , Margin = 1, Accuracy = 0.9315
    ####### Winnow1: Alpha = 1.1 , Margin = 0, Accuracy = 0.8813
    ####### Winnow2: Alpha = 1.1 , Margin = 2.0 Accuracy = 0.9056
    ####### AdaGrad: LR = 1.5 Accuracy = 0.8985
    ####### m = 1000: #############################################
    ####### Perceptron1: LR = 1 , Margin = 0, Accuracy =  0.7545
    ####### Perceptron2: LR = 0.25 , Margin = 1, Accuracy = 0.7839
    ####### Winnow1: Alpha = 1.1 , Margin = 0, Accuracy = 0.7675
    ####### Winnow2: Alpha = 1.1 , Margin = 0.006 Accuracy = 0.7325
    ####### AdaGrad: LR = 1.5 Accuracy = 0.8488




