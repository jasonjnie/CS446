import math
import random
import numpy as np
import os
import time
import arff  # Downloaded from https://pypi.python.org/pypi/arff/0.9


def Train_Classifier(w,theta,train_data,train_label,LR):
	w_temp = w
	for i in range(len(train_data)):
		example_data = train_data[i]		# example = (1,260)
		example_label = train_label[i]
		#Od = np.around(np.dot(w,example_data), decimals=10)

		temp_Od = np.dot(w_temp,example_data)
		if temp_Od >= 0:
			Od = 1;
		elif temp_Od < 0:
			Od = -1
		#print "example data =",example_data
		#print "example label =",example_label
		#print "Od =",Od
		#print "LR =",LR
		#Od = temp_Od
		#wi = [100]
		
		#print "example data shape:", len(example_data)
		#wi = LR * (example_label - Od) * example_data
		wi = []
		temp = LR * (example_label - Od)
		for j in range(len(train_data[0])):
			#print "temp =",temp
			#print "example_data =",example_data[j]
			wi_temp = temp * example_data[j]
			wi.append(wi_temp)

		#print "w =",len(w)
		#print "wi =",wi
		w_temp +=  wi								# update weight vector
		theta += LR * (example_label - Od)		# update theta(w_0)

	return (w_temp,theta)


def Stop(clf,prev_E,train_data,train_label,epsilon):		# returns 1 if doesn't stop
	w = clf[0]
	theta = clf[1]
	predict_label = Predict(w,theta,train_data,train_label)
	total_misclassification = 0
	N = len(train_data)
	for i in range(N):
		if train_label[i] != predict_label[i]:
			total_misclassification += 1
	new_E = 4 * total_misclassification / N #########################
	if abs(new_E - prev_E) <= epsilon:		# error falls below threshold, stop (return 0)
		return (0,new_E)	# stop
	else:
		return (1,new_E)	# keep training


def Predict(w,theta,data,label):
	predict_label = np.zeros(len(label))
	for i in range(len(data)):
		example_data = data[i]		# example = (1,260)
		#example_label = label[i]
		temp = np.dot(w,example_data) + theta
		if temp >= 0:
			predict_label[i] = 1
		elif temp < 0:
			predict_label[i] = -1

	return predict_label


def Evaluate_Clf(clf,test_data,test_label):
	w = clf[0]
	theta = clf[1]
	predict_label = Predict(w,theta,test_data,test_label)
	Correct = 0
	N = len(test_data)
	for i in range(N):
		if test_label[i] == predict_label[i]:
			Correct += 1

	return 	float(Correct)/N	




if __name__ == "__main__":

#for row in arff.load("badges.modified.data.fold1.arff"):
	fold_train = []
	fold_train.append(list(arff.load("out_train_1.arff")))
	fold_train.append(list(arff.load("out_train_2.arff")))
	fold_train.append(list(arff.load("out_train_3.arff")))
	fold_train.append(list(arff.load("out_train_4.arff")))
	fold_train.append(list(arff.load("out_train_5.arff")))
	fold_test = []
	fold_test.append(list(arff.load("out_test_1.arff")))
	fold_test.append(list(arff.load("out_test_2.arff")))
	fold_test.append(list(arff.load("out_test_3.arff")))
	fold_test.append(list(arff.load("out_test_4.arff")))
	fold_test.append(list(arff.load("out_test_5.arff")))


################# Extract data and label from ARFF file)############
	# extract from training data
	number_of_train_example = []
	for i in range(0,5):
		number_of_train_example.append(len(fold_train[i]))
	train_data = []
	train_label = []
	N_dim = len(fold_train[0][0]) - 1
	#print "N_dim_train =", N_dim
	for i in range(0,5):
		temp_train_fold = []
		temp_train_label = []
		for j in range(number_of_train_example[i]):
			temp_train_feat = []
			for k in range(0,N_dim):
				temp_train_feat.append(int(fold_train[i][j][k]))
			temp_train_fold.append(temp_train_feat)
			if fold_train[i][j][k+1] == '+':		# if labeled positive, add 1
				temp_train_label.append(1)
			elif fold_train[i][j][k+1] == '-':	# if labeled negative, add -1
				temp_train_label.append(-1)
		train_data.append(temp_train_fold)
		train_label.append(temp_train_label)

	# extract from testing data
	number_of_test_example = []
	for i in range(0,5):
		number_of_test_example.append(len(fold_test[i]))
	test_data = []
	test_label = []
	#N_dim = len(fold_test[0][0]) - 1
	#print "N_dim_test =", N_dim
	for i in range(0,5):
		temp_test_fold = []
		temp_test_label = []
		for j in range(number_of_test_example[i]):
			temp_test_feat = []
			for k in range(0,N_dim):				
				temp_test_feat.append(int(fold_test[i][j][k]))
			temp_test_fold.append(temp_test_feat)
			if fold_test[i][j][k+1] == '+':		# if labeled positive, add 1
				temp_test_label.append(1)
			elif fold_test[i][j][k+1] == '-':	# if labeled negative, add -1
				temp_test_label.append(-1)
		test_data.append(temp_test_fold)
		test_label.append(temp_test_label)

	#print number_of_example		# 294 in total
	#print "data =",len(data[0])
	#print "label =",len(label[0])

###################### Train Classifier using 5-fold Cross-Validation ############
	#LR = 0.00001		# Learning Rate
	#epsilon = 0.000266666666667
	Num_LR = 50
	Num_epsilon = 10
	LR_all = np.linspace(0.0001,0.5,Num_LR)
	epsilon_all = np.linspace(0.00001,0.001,Num_epsilon)

	count = 0
	Accuracy = []
	Std = []
	All_Correct = []
	All_Incorrect = []
	#M = len(train_fold[0])  ########## delete

	for LR in LR_all:
		for epsilon in epsilon_all:
			print "count =",count
			count += 1
			Acc_CV = np.zeros(5)
			prev_w = np.zeros(N_dim)	# initialize w to 0
			prev_theta = 0			# initialize theta to 0
			temp_all_correct = []
			temp_all_incorrect = []

			# 5-fold Cross-Validation
			for i in range(0,5):			# i= index of test fold
				train_fold_data = train_data[i]
				train_fold_label = train_label[i]
				test_fold_data = test_data[i]
				test_fold_label = test_label[i]

				'''
				if(i==0):
					train_fold_data = train_data[i][65:]
					train_fold_label = train_label[i][65:]
					#print "A =",len(train_fold_data)
					#print "B =",len(train_fold_label)
				if(i==1):
					train_fold_data = np.concatenate((data[i][0:65],data[i][122:]),axis=0)
					train_fold_label = np.concatenate((label[i][0:65],label[i][122:]),axis=0)
				if(i==2):
					train_fold_data = np.concatenate((data[i][:122],data[i][168:]),axis=0)
					train_fold_label = np.concatenate((label[i][:122],label[i][168:]),axis=0)
				if(i==3):
					train_fold_data = np.concatenate((data[i][:168],data[i][234:]),axis=0)
					train_fold_label = np.concatenate((label[i][:168],label[i][234:]),axis=0)
				if(i==4):
					train_fold_data = data[i][:234]
					train_fold_label = label[i][:234]
				'''
				prev_E = 0
				
				# keep training until error falls below threshold(epsilon)
				if_stop = 1
				while if_stop:
					# Train_Classifier returns (prev_w,prev_theta)	
					#print len(train_fold_data)
					#print len(train_fold_data[0][0])				
					clf = Train_Classifier(prev_w,prev_theta,train_fold_data,train_fold_label,LR)
					prev_w = clf[0]
					prev_theta = clf[1]
					# Stop returns (if_stop,prev_E)
					temp_stop = Stop(clf,prev_E,train_fold_data,train_fold_label,epsilon) 
					if_stop = temp_stop[0]	  # if_stop = 0 to stop, =1 to keep training
					prev_E = temp_stop[1]

				Acc_CV[i] = Evaluate_Clf(clf,test_fold_data,test_fold_label)
				count_correct = int(Acc_CV[i]*len(test_fold_data))
				count_incorrect = len(test_fold_data) - count_correct
				temp_all_correct.append(count_correct)
				temp_all_incorrect.append(count_incorrect)

			temp_Acc = np.mean(Acc_CV,axis=0)
			print "temp_Acc =",Acc_CV
			print "Accuracy =",temp_Acc
			Accuracy.append(temp_Acc)
			temp_std = np.std(Acc_CV,axis=0)
			Std.append(temp_std)
			All_Correct.append(temp_all_correct)
			All_Incorrect.append(temp_all_incorrect)

	temp_a = Accuracy[0]
	temp_index = 0
	for index in range(len(Accuracy)):	# find largest accuracy
		a = Accuracy[index]
		if a > temp_a:
			temp_a = a
			temp_index = index

	LR_index = temp_index / Num_epsilon
	epsilon_index = temp_index % Num_epsilon
	LR_best = LR_all[LR_index]
	epsilon_best = epsilon_all[epsilon_index]

	print "Maximum Accuracy =",temp_a
	print "Standard Deviation =",Std[temp_index]
	print "Best LR =",LR_best
	print "Best epsilon =",epsilon_best
	print "temp_index =",temp_index
	print "Count of correct =",All_Correct[temp_index]
	print "Count of incorrect =",All_Incorrect[temp_index]

	
	######### Result #######
	# Max Accuracy: 0.673711467785
	# Standard Deviation: 0.137081657891
	# LR: 0.244948979592	# may not result in the accuracy above due to precision change
							# if not running double loop, need to find LR in LR_all to 
							# obtain the above Accuracy 
	# epsilon: 1e-05










