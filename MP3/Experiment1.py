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
    m = 100
    n1 = 500
    n2 = 1000
    N = 50000

    (all_label_n1, all_data_n1) = gen(l, m, n1, N, False)
    (all_label_n2, all_data_n2) = gen(l, m, n2, N, False)

    (train_data_n1, train_label_n1, test_data_n1, test_label_n1) = random_10(all_data_n1, all_label_n1)
    (train_data_n2, train_label_n2, test_data_n2, test_label_n2) = random_10(all_data_n2, all_label_n2)

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
    Perceptron1_n1 = Perceptron(1, 0, all_data_n1, 500)
    Perceptron1_n2 = Perceptron(1, 0, all_data_n2, 1000)

    Perceptron1_n1.train(all_data_n1, all_label_n1)
    Perceptron1_n2.train(all_data_n2, all_label_n2)

    W1_n1 = Perceptron1_n1.mistake[-1]
    W1_n2 = Perceptron1_n2.mistake[-1]

    print "Perceptron1: No Margin"
    print "LR = 1, Margin = 0"
    print "n=500, mistakes = ", W1_n1
    print "n=1000, mistakes = ", W1_n2


    
    ### 2: Perceptron2: Tuning LR #################################
    Acc_n1 = []
    Acc_n2 = []

    for i in range(5):
        print "2",i
        temp_LR = Per2_LR[i]
        Perceptron2_n1 = Perceptron(temp_LR, 1, train_data_n1, 500)
        Perceptron2_n2 = Perceptron(temp_LR, 1, train_data_n2, 1000)

        for j in range(20):
            Perceptron2_n1.train(train_data_n1, train_label_n1)
            Perceptron2_n2.train(train_data_n2, train_label_n2)

        est_label_n1 = Perceptron2_n1.predict(test_data_n1)
        est_label_n2 = Perceptron2_n2.predict(test_data_n2)
        temp_Acc_n1 = Get_Accuracy(test_label_n1, est_label_n1)
        temp_Acc_n2 = Get_Accuracy(test_label_n2, est_label_n2)
        #print temp_Acc_n1
        #print temp_Acc_n2
        Acc_n1.append(temp_Acc_n1)
        Acc_n2.append(temp_Acc_n2)

    #print Acc_n1
    #print Acc_n2
    Best_LR_n1_idx = np.argmax(Acc_n1)
    Best_LR_n2_idx = np.argmax(Acc_n2)
    Best_LR_n1 = Per2_LR[Best_LR_n1_idx]
    Best_LR_n2 = Per2_LR[Best_LR_n2_idx]
    Perceptron2_n1 = Perceptron(Best_LR_n1, 1, all_data_n1, 500)
    Perceptron2_n2 = Perceptron(Best_LR_n2, 1, all_data_n2, 1000)
    Perceptron2_n1.train(all_data_n1, all_label_n1)
    Perceptron2_n2.train(all_data_n2, all_label_n2)
    W2_n1 = Perceptron2_n1.mistake[-1]
    W2_n2 = Perceptron2_n2.mistake[-1]

    print "Percptron2: with Margin"
    print "n = 500: LR =", Best_LR_n1, ", Margin = 1,", "Mistakes =", W2_n1
    print "n = 1000: LR =", Best_LR_n2, ", Margin = 1,", "Mistakes =", W2_n2
    
    
    ### 3: Winnow1: Tuning Alpha #################################
    Acc_n1 = []
    Acc_n2 = []

    for i in range(5):
        print "3",i
        temp_alpha = Win_Alpha[i]
        Winnow1_n1 = Winnow(temp_alpha, 0, train_data_n1, 500)
        Winnow1_n2 = Winnow(temp_alpha, 0, train_data_n2, 1000)

        for j in range(20):
            Winnow1_n1.train(train_data_n1, train_label_n1)
            Winnow1_n2.train(train_data_n2, train_label_n2)

        est_label_n1 = Winnow1_n1.predict(test_data_n1)
        est_label_n2 =  Winnow1_n2.predict(test_data_n2)
        temp_Acc_n1 = Get_Accuracy(test_label_n1, est_label_n1)
        temp_Acc_n2 = Get_Accuracy(test_label_n2, est_label_n2)

        Acc_n1.append(temp_Acc_n1)
        Acc_n2.append(temp_Acc_n2)

    Best_alpha_n1_idx = np.argmax(Acc_n1)
    Best_alpha_n2_idx = np.argmax(Acc_n2)
    Best_alpha_n1 = Win_Alpha[Best_alpha_n1_idx]
    Best_alpha_n2 = Win_Alpha[Best_alpha_n2_idx]
    Winnow1_n1 = Winnow(Best_alpha_n1, 1, all_data_n1, 500)
    Winnow1_n2 = Winnow(Best_alpha_n2, 1, all_data_n2, 1000)
    Winnow1_n1.train(all_data_n1, all_label_n1)
    Winnow1_n2.train(all_data_n2, all_label_n2)
    W2_n1 = Winnow1_n1.mistake[-1]
    W2_n2 = Winnow1_n2.mistake[-1] 

    print "Winnow1: w/o Margin"
    print "n = 500: Alpha =", Best_alpha_n1, ", Margin = 0,", "Mistakes =", W2_n1
    print "n = 1000: Alpha =", Best_alpha_n2, ", Margin = 0,", "Mistakes =", W2_n2



    ### 4: Winnow2: Tuning Alpha & Margin #################################
    Acc_n1 = []
    Acc_n2 = []

    for i in range(5):      # select alpha
        print "4",i
        temp_alpha = Win_Alpha[i]
        for k in range(5):      # select margin
            temp_margin = Win2_Margin[k]
            Winnow2_n1 = Winnow(temp_alpha, temp_margin, train_data_n1, 500)
            Winnow2_n2 = Winnow(temp_alpha, temp_margin, train_data_n2, 1000)

            for j in range(20):
                Winnow2_n1.train(train_data_n1, train_label_n1)
                Winnow2_n2.train(train_data_n2, train_label_n2)

            est_label_n1 = Winnow2_n1.predict(test_data_n1)
            est_label_n2 =  Winnow2_n2.predict(test_data_n2)
            temp_Acc_n1 = Get_Accuracy(test_label_n1, est_label_n1)
            temp_Acc_n2 = Get_Accuracy(test_label_n2, est_label_n2)

            Acc_n1.append(temp_Acc_n1)
            Acc_n2.append(temp_Acc_n2)

    Best_n1_idx = np.argmax(Acc_n1)
    Best_n2_idx = np.argmax(Acc_n2)
    Best_alpha_n1_idx = Best_n1_idx / 5
    Best_alpha_n2_idx = Best_n2_idx / 5
    Best_margin_n1_idx = Best_n1_idx % 5
    Best_margin_n2_idx = Best_n2_idx % 5
    Best_alpha_n1 = Win_Alpha[Best_alpha_n1_idx]
    Best_alpha_n2 = Win_Alpha[Best_alpha_n2_idx]
    Best_margin_n1 = Win2_Margin[Best_margin_n1_idx]
    Best_margin_n2 = Win2_Margin[Best_margin_n2_idx]
    Winnow2_n1 = Winnow(Best_alpha_n1, Best_margin_n1, all_data_n1, 500)
    Winnow2_n2 = Winnow(Best_alpha_n2, Best_margin_n2, all_data_n2, 1000)
    Winnow2_n1.train(all_data_n1, all_label_n1)
    Winnow2_n2.train(all_data_n2, all_label_n2)
    W2_n1 = Winnow2_n1.mistake[-1]
    W2_n2 = Winnow2_n2.mistake[-1]  

    print "Winnow2: with Margin"
    print "n = 500: Alpha =", Best_alpha_n1, ", Margin =", Best_margin_n1, "Mistakes =", W2_n1
    print "n = 1000: Alpha =", Best_alpha_n2, ", Margin =", Best_margin_n2, "Mistakes =", W2_n2



    ### 5: AdaGrad: Tuning LR #################################
    Acc_n1 = []
    Acc_n2 = []

    for i in range(5):
        print "5", i
        temp_LR = Ada_LR[i]
        AdaGrad_n1 = AdaGrad(temp_LR, train_data_n1, 500)
        AdaGrad_n2 = AdaGrad(temp_LR, train_data_n2, 1000)

        for j in range(20):
            AdaGrad_n1.train(train_data_n1, train_label_n1)
            AdaGrad_n2.train(train_data_n2, train_label_n2)

        #print AdaGrad_n1.w
        #print AdaGrad_n2.w
        est_label_n1 = AdaGrad_n1.predict(test_data_n1)
        est_label_n2 = AdaGrad_n2.predict(test_data_n2)
        temp_Acc_n1 = Get_Accuracy(test_label_n1, est_label_n1)
        temp_Acc_n2 = Get_Accuracy(test_label_n2, est_label_n2)
        #print temp_Acc_n1
        #print temp_Acc_n2
        Acc_n1.append(temp_Acc_n1)
        Acc_n2.append(temp_Acc_n2)

    print Acc_n1
    print Acc_n2
    Best_LR_n1_idx = np.argmax(Acc_n1)
    Best_LR_n2_idx = np.argmax(Acc_n2)
    Best_LR_n1 = Ada_LR[Best_LR_n1_idx]
    Best_LR_n2 = Ada_LR[Best_LR_n2_idx]
    AdaGrad_n1 = AdaGrad(Best_LR_n1, all_data_n1, 500)
    AdaGrad_n2 = AdaGrad(Best_LR_n2, all_data_n2, 1000)
    AdaGrad_n1.train(all_data_n1, all_label_n1)
    AdaGrad_n2.train(all_data_n2, all_label_n2)
    W_n1 = AdaGrad_n1.mistake[-1]
    W_n2 = AdaGrad_n2.mistake[-1]   

    print "AdaGrad:"
    print "n = 500: LR =", Best_LR_n1, "Mistakes =", W_n1
    print "n = 1000: LR =", Best_LR_n2, "Mistakes =", W_n2


    ####### Plot Curves ########################################
    N = np.linspace(1,50000,50000)
    fig1 = plt.figure()
    plt.plot(N, Perceptron1_n1.mistake, 'b', label = 'Perceptron')
    plt.plot(N, Perceptron2_n1.mistake, 'g', label = 'Perceptron w/ Margin')
    plt.plot(N, Winnow1_n1.mistake, 'r', label = 'Winnow')
    plt.plot(N, Winnow2_n1.mistake, 'm', label = 'Winnow w/ Margin')
    plt.plot(N, AdaGrad_n1.mistake, 'y', label = 'AdaGrad')
    plt.xlabel('N')
    plt.ylabel('W')
    plt.title('W vs N for n=500')
    plt.legend(loc = 4, fontsize = 10)
    plt.show()


    fig2 = plt.figure()
    plt.plot(N, Perceptron1_n2.mistake, 'b', label = 'Perceptron')
    plt.plot(N, Perceptron2_n2.mistake, 'g', label = 'Perceptron w/ Margin')
    plt.plot(N, Winnow1_n2.mistake, 'r', label = 'Winnow')
    plt.plot(N, Winnow2_n2.mistake, 'm', label = 'Winnow w/ Margin')
    plt.plot(N, AdaGrad_n2.mistake, 'y', label = 'AdaGrad')
    plt.xlabel('N')
    plt.ylabel('W')
    plt.title('W vs N for n=1000')
    plt.legend(loc = 4, fontsize = 10)
    plt.show()


    ####### Result ##############################################
    ####### Perceptron1: n = 500, mistakes =  11535
    ####### Perceptron1: n = 1000, mistakes =  16338
    ####### Perceptron2: n = 500: LR = 0.25 , Margin = 1, Mistakes = 11464
    ####### Perceptron2: n = 1000: LR = 0.25 , Margin = 1, Mistakes = 16196
    ####### Winnow1: n = 500: Alpha = 1.1 , Margin = 0, Mistakes = 2987
    ####### Winnow1: n = 1000: Alpha = 1.1 , Margin = 0, Mistakes = 3943
    ####### Winnow2: n = 500: Alpha = 1.1 , Margin = 2.0 Mistakes = 2947
    ####### Winnow2: n = 1000: Alpha = 1.1 , Margin = 2.0 Mistakes = 3896
    ####### AdaGrad: n = 500: LR = 0.25 Mistakes = 4902
    ####### AdaGrad: n = 1000: LR = 0.25 Mistakes = 6796



