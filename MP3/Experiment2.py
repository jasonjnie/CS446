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
    m = 20
    all_n = np.linspace(40,200,5)
    N = 50000
    Perceptron1_Mistake = []
    Perceptron2_Mistake = []
    Winnow1_Mistake = []
    Winnow2_Mistake = []
    AdaGrad_Mistake = []


    for n in all_n:
        print "n =",n
        (all_label, all_data) = gen(l, m, n, N, False)
        (train_data, train_label, test_data, test_label) = random_10(all_data, all_label)

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
        Perceptron1 = Perceptron(1, 0, all_data, n)
        converge = Perceptron1.Converge(all_data, all_label)
        if converge == True:
            W1 = Perceptron1.mistake_converge
            Perceptron1_Mistake.append(W1)
            print "Perceptron1: No Margin"
            print "LR = 1, Margin = 0"
            print "n =", n ,"mistakes =", W1
        else:
            print "########## Warning: No Convergence ############"
        
        
        ### 2: Perceptron2: Tuning LR #################################
        Acc = []

        for i in range(5):
            temp_LR = Per2_LR[i]
            Perceptron2 = Perceptron(temp_LR, 1, train_data, n)
            for j in range(20):
                Perceptron2.train(train_data, train_label)
            est_label = Perceptron2.predict(test_data)
            temp_Acc = Get_Accuracy(test_label, est_label)
            #print temp_Acc
            Acc.append(temp_Acc)

        print Acc
        Best_LR_idx = np.argmax(Acc)
        Best_LR = Per2_LR[Best_LR_idx]
        Perceptron2 = Perceptron(Best_LR, 1, all_data, n)
        converge = Perceptron2.Converge(all_data, all_label)
        if converge == True:
            W2 = Perceptron2.mistake_converge
            Perceptron2_Mistake.append(W2)
            print "Percptron2: with Margin"
            print "n =", n, ": LR =", Best_LR, ", Margin = 1,", "Mistakes =", W2
        else:
            print "########## Warning: No Convergence ############"
        
        
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

        print Acc
        Best_alpha_idx = np.argmax(Acc)
        Best_alpha = Win_Alpha[Best_alpha_idx]
        Winnow1 = Winnow(Best_alpha, 1, all_data, n)
        converge = Winnow1.Converge(all_data, all_label)
        if converge == True:
            W3 = Winnow1.mistake_converge
            Winnow1_Mistake.append(W3)
            print "Winnow1: w/o Margin"
            print "n =", n, ": Alpha =", Best_alpha, ", Margin = 0,", "Mistakes =", W3
        else:
            print "########## Warning: No Convergence ############"

        
        
        ### 4: Winnow2: Tuning Alpha & Margin #################################
        Acc = []

        for i in range(5):      # select alpha
            temp_alpha = Win_Alpha[i]
            for k in range(5):      # select margin
                temp_margin = Win2_Margin[k]
                Winnow2= Winnow(temp_alpha, temp_margin, train_data, n)
                for j in range(20):
                    Winnow2.train(train_data, train_label)
                est_label = Winnow2.predict(test_data)
                temp_Acc = Get_Accuracy(test_label, est_label)
                Acc.append(temp_Acc)

        print Acc
        Best_idx = np.argmax(Acc)
        Best_alpha_idx = Best_idx / 5
        Best_margin_idx = Best_idx % 5
        Best_alpha = Win_Alpha[Best_alpha_idx]
        Best_margin = Win2_Margin[Best_margin_idx]
        Winnow2 = Winnow(Best_alpha, Best_margin, all_data, n)
        converge = Winnow2.Converge(all_data, all_label)
        if converge == True:
            W4 = Winnow2.mistake_converge
            Winnow2_Mistake.append(W4)
            print "Winnow2: with Margin"
            print "n =", n, ": Alpha =", Best_alpha, ", Margin =", Best_margin, "Mistakes =", W4
        else:
            print "########## Warning: No Convergence ############"

        
        ### 5: AdaGrad: Tuning LR #################################
        Acc = []

        for i in range(5):
            temp_LR = Ada_LR[i]
            AdaGrad1 = AdaGrad(temp_LR, train_data, n)
            for j in range(20):
                AdaGrad1.train(train_data, train_label)
            est_label = AdaGrad1.predict(test_data)
            temp_Acc = Get_Accuracy(test_label, est_label)
            #print temp_Acc_n1
            Acc.append(temp_Acc)

        print Acc
        Best_LR_idx = np.argmax(Acc)
        Best_LR = Ada_LR[Best_LR_idx]
        AdaGrad2 = AdaGrad(Best_LR, all_data, n)
        converge = AdaGrad2.Converge(all_data, all_label)
        if converge == True:
            W5 = AdaGrad2.mistake_converge
            AdaGrad_Mistake.append(W5)
            print "AdaGrad:"
            print "n =", n, ": LR =", Best_LR, "Mistakes =", W5
        else:
            print "########## Warning: No Convergence ############"
        


    ####### Plot Curves ########################################
    fig1 = plt.figure()
    plt.plot(all_n, Perceptron1_Mistake, 'b', label = 'Perceptron')
    plt.plot(all_n, Perceptron2_Mistake, 'g', label = 'Perceptron w/ Margin')
    plt.plot(all_n, Winnow1_Mistake, 'r', label = 'Winnow')
    plt.plot(all_n, Winnow2_Mistake, 'm', label = 'Winnow w/ Margin')
    plt.plot(all_n, AdaGrad_Mistake, 'y', label = 'AdaGrad')
    plt.xlabel('n')
    plt.ylabel('W')
    plt.title('W vs n')
    plt.legend(loc = 4, fontsize = 10)
    plt.show()


    ####### Result ##############################################
    ####### n = 40: #############################################
    ####### Perceptron1: LR = 1 , Margin = 0, mistakes = 895
    ####### Perceptron2: LR = 0.25 , Margin = 1, Mistakes = 822
    ####### Winnow1: Alpha = 1.1 , Margin = 0, Mistakes = 113
    ####### Winnow2: Alpha = 1.1 , Margin = 2.0 Mistakes = 108
    ####### AdaGrad: LR = 1.5 Mistakes = 679
    ####### n = 80: #############################################
    ####### Perceptron1: LR = 1 , Margin = 0, mistakes = 1868
    ####### Perceptron2: LR = 0.25 , Margin = 1, Mistakes = 1729
    ####### Winnow1: Alpha = 1.1 , Margin = 0, Mistakes = 230
    ####### Winnow2: Alpha = 1.1 , Margin = 2.0 Mistakes = 215
    ####### AdaGrad: LR = 1.5 Mistakes = 1557
    ####### n = 120: #############################################
    ####### Perceptron1: LR = 1 , Margin = 0, mistakes = 2822
    ####### Perceptron2: LR = 0.03 , Margin = 1, Mistakes = 2323
    ####### Winnow1: Alpha = 1.1 , Margin = 0, Mistakes = 306
    ####### Winnow2: Alpha = 1.1 , Margin = 2.0 Mistakes = 288
    ####### AdaGrad: LR = 1.5 Mistakes = 2655
    ####### n = 160: #############################################
    ####### Perceptron1: LR = 1 , Margin = 0, mistakes = 4019
    ####### Perceptron2: LR = 0.25 , Margin = 1, Mistakes = 3049
    ####### Winnow1: Alpha = 1.1 , Margin = 0, Mistakes = 369
    ####### Winnow2: Alpha = 1.1 , Margin = 2.0 Mistakes = 353
    ####### AdaGrad: LR = 1.5 Mistakes = 4232
    ####### n = 200: #############################################
    ####### Perceptron1: LR = 1 , Margin = 0, mistakes = 5128
    ####### Perceptron2: LR = 0.03 , Margin = 1, Mistakes = 3768
    ####### Winnow1: Alpha = 1.1 , Margin = 0, Mistakes = 416
    ####### Winnow2: Alpha = 1.1 , Margin = 0.3 Mistakes = 449
    ####### AdaGrad: LR = 1.5 Mistakes = 5377














