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
    n = 40
    N = 10000
    LR = 1.5
    num_round = 50
    misclf_rate = []
    hinge_loss = []

    (train_label, train_data) = gen(l, m, n, N, True)

    clf_AdaGrad = AdaGrad(LR, train_data, n)

    for i in range(num_round):
        print i
        clf_AdaGrad.train(train_data, train_label)
        #est_label = clf_AdaGrad.predict(train_data)
        temp_misclf_error = clf_AdaGrad.misclf_error(train_data,train_label)
        #print temp_misclf_error
        misclf_rate.append(temp_misclf_error)
        temp_hinge_loss = clf_AdaGrad.hinge_loss(train_data, train_label)
        #print temp_hinge_loss
        hinge_loss.append(temp_hinge_loss)

    N = np.linspace(1,50,50)
    fig1 = plt.figure()
    plt.plot(N, misclf_rate, 'b', label = 'Misclassification Error')
    plt.xlabel('Training Rounds')
    plt.ylabel('Misclassification Error')
    plt.title('Misclassification Error vs Training Rounds')
    plt.legend(loc = 4, fontsize = 10)
    plt.show()

    fig2 = plt.figure()
    plt.plot(N, hinge_loss, 'g', label = 'Hinge Loss')
    plt.xlabel('Training Rounds')
    plt.ylabel('Hinge Lodgfqafsss')
    plt.title('Hinge Loss vs Training Rounds')
    plt.legend(loc = 4, fontsize = 10)
    plt.show()



