import NN, data_loader, perceptron
import numpy as np
import matplotlib.pyplot as plt


#training_data, test_data = data_loader.load_circle_data()
training_data, test_data = data_loader.load_mnist_data()
num_train = len(training_data)
num_test = len(test_data)
train_fold_size = int(num_train / 5)
test_fold_size = int(num_test / 5)

#print "tainin:", len(training_data)
#print "test:", len(test_data)

#domain = 'circles'
domain = 'mnist'
all_batch_size = [10,50,100]
#all_batch_size = [10]
all_learning_rate = [0.1,0.01]
#all_learning_rate = [0.1]
all_activation_function = ['relu','tanh']
all_hidden_layer_width = [10,50]
#all_hidden_layer_width = [50]
data_dim = len(training_data[0][0])

all_params = []
for b_size in all_batch_size:
	for LR in all_learning_rate:
		for act_func in all_activation_function:
			for layer_width in all_hidden_layer_width:
				temp = [b_size,LR,act_func,layer_width]
				all_params.append(temp)


highest_acc = 0
count = 0
all_Acc = []
for param in all_params:
	print "Count =", count
	count += 1
	temp_Accuracy = []
	for i in range(5):
		if i != 4:	# first four folds
			train_fold = training_data[i*train_fold_size:(i+1)*train_fold_size]
			test_fold = test_data[i*test_fold_size:(i+1)*test_fold_size]
		else:		# last fold
			train_fold = training_data[i*train_fold_size:]
			test_fold = test_data[i*test_fold_size:]

		net = NN.create_NN(domain, param[0], param[1], param[2], param[3])
		net.train(train_fold)
		fold_acc = net.evaluate(test_fold)
		temp_Accuracy.append(fold_acc)		

	Accuracy = sum(temp_Accuracy) / 5.0
	all_Acc.append(Accuracy)
	if Accuracy > highest_acc:
		highest_acc = Accuracy
		best_param = param

print "Accuracy for each parameter setting:"
print "format: [batch size,learning rate,activation func,hidden layer width]"
for i in range(len(all_params)):
	print all_params[i], "accuracy:", all_Acc[i]
print "Highest Accuracy of Parameter Tuning:", highest_acc
print "Best Paramerters:", best_param


net_NN = NN.create_NN(domain, best_param[0], best_param[1], best_param[2], best_param[3])
NN_curve_data = net_NN.train_with_learning_curve(training_data)
NN_Acc = net_NN.evaluate(test_data)

perc = perceptron.Perceptron(data_dim)
perc_curve_data = perc.train_with_learning_curve(training_data)
perc_Acc = perc.evaluate(test_data)

NN_curve = []
perc_curve = []
for i in range(len(NN_curve_data)):
	NN_curve.append(NN_curve_data[i][1])
	perc_curve.append(perc_curve_data[i][1] * 100.0)

#print "NN_curve",NN_curve
#print "perc_curve",perc_curve

print "NN Accuracy:", NN_Acc
print "Perceptron Accuracy:", perc_Acc

#print len(Num_curve)
N = np.linspace(1,100,100)
fig = plt.figure()
ax = fig.gca()
ax.set_autoscale_on(False)
ax.plot(N, NN_curve, 'b', label = 'NN')
ax.plot(N, perc_curve, 'g', label = 'Perceptron')
ax.axis([0.0,100.0,0.0,120.0])
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Iteration on Circles Dataset')
plt.legend(loc = 4, fontsize = 10)
plt.show()





'''

print "A"
net = NN.create_NN(domain, batch_size, learning_rate, activation_function, hidden_layer_width)
print net.train(training_data)
print net.evaluate(test_data)

print "B"
net = NN.create_NN(domain, batch_size, learning_rate, activation_function, hidden_layer_width)
print net.train_with_learning_curve(training_data)
print net.evaluate(test_data)

print "C"
perc = perceptron.Perceptron(data_dim)
print perc.train(training_data)
print perc.evaluate(test_data)

print "D"
perc = perceptron.Perceptron(data_dim)
print perc.train_with_learning_curve(training_data)
print perc.evaluate(test_data)

'''

