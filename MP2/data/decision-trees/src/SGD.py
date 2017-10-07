import math
import random
import numpy as np
import os
import time
import arff  # Downloaded from https://pypi.python.org/pypi/arff/0.9


def Train_Classifier(w,theta,train_data,train_label,LR):
	for i in range(len(train_data)):
		example_data = train_data[i]		# example = (1,260)
		example_label = train_label[i]
		temp_Od = np.dot(w,example_data)
		if temp_Od >= 0:
			Od = 1;
		elif temp_Od < 0:
			Od = -1
		wi = []
		temp = LR * (example_label - Od)
		for j in range(len(train_data[0])):
			temp_wi = temp * example_data[j]
			wi.append(temp_wi)
		w +=  wi								# update weight vector
		theta += LR * (example_label - Od)		# update theta(w_0)

	return (w,theta)


def Stop(clf,prev_E,train_data,train_label,epsilon):		# returns 1 if doesn't stop
	w = clf[0]
	theta = clf[1]
	predict_label = Predict(w,theta,train_data,train_label)
	total_misclassification = 0
	N = len(train_data)
	for i in range(N):
		if train_label[i] != predict_label[i]:
			total_misclassification += 1
	new_E = 4 * total_misclassification / N 
	if abs(new_E - prev_E) <= epsilon:		# error falls below threshold, stop (return 0)
		return (0,new_E)	# stop
	else:
		return (1,new_E)	# keep training


def Predict(w,theta,data,label):
	predict_label = np.zeros(len(label))
	for i in range(len(data)):
		example_data = data[i]		# example = (1,260)
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
	fold = []
	fold.append(list(arff.load("badges.modified.data.fold1.arff")))
	fold.append(list(arff.load("badges.modified.data.fold2.arff")))
	fold.append(list(arff.load("badges.modified.data.fold3.arff")))
	fold.append(list(arff.load("badges.modified.data.fold4.arff")))
	fold.append(list(arff.load("badges.modified.data.fold5.arff")))

	########### Extract data and label from ARFF file)############
	number_of_example = []
	for i in range(0,5):
		number_of_example.append(len(fold[i]))
	data = []
	label = []
	for i in range(0,5):
		temp_fold = []
		temp_label = []
		for j in range(number_of_example[i]):
			temp_feat = []
			for k in range(0,260):
				temp_feat.append(int(fold[i][j][k]))
			temp_fold.append(temp_feat)
			if fold[i][j][k+1] == '+':		# if labeled positive, add 1
				temp_label.append(1)
			elif fold[i][j][k+1] == '-':	# if labeled negative, add -1
				temp_label.append(-1)
		data.append(temp_fold)
		label.append(temp_label)

	#print number_of_example		# 294 in total

	############# Train Classifier using 5-fold Cross-Validation ############
	#LR = 0.00001		# Learning Rate
	#epsilon = 0.000266666666667
	Num_LR = 50
	Num_epsilon = 5
	LR_all = np.linspace(0.0001,0.5,Num_LR)
	epsilon_all = np.linspace(0.00001,0.00005,Num_epsilon)

	count = 0
	Accuracy = []
	Acc_5 = []
	Std = []
	All_Correct = []
	All_Incorrect = []

	for LR in LR_all:
		for epsilon in epsilon_all:
			print "count =",count
			count += 1
			Acc_CV = np.zeros(5)
			prev_w = np.zeros(260)	# initialize w to 0
			prev_theta = 0			# initialize theta to 0
			temp_all_correct = []
			temp_all_incorrect = []

			# 5-fold Cross-Validation
			for i in range(0,5):			# i= index of test fold
				test_fold_data = data[i]
				test_fold_label = label[i]
				train_fold_data = np.concatenate((data[i-1],data[i-2],data[i-3],data[i-4]),axis=0)
				train_fold_label = np.concatenate((label[i-1],label[i-2],label[i-3],label[i-4]),axis=0)
				prev_E = 0
				
				# keep training until error falls below threshold(epsilon)
				if_stop = 1
				while if_stop:
					# Train_Classifier returns (prev_w,prev_theta)						
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

			Acc_5.append(Acc_CV)
			temp_Acc = np.mean(Acc_CV,axis=0)
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
	print "Accuracy over 5 Folds: ",Acc_5[temp_index]
	print "Count of correct =",All_Correct[temp_index]
	print "Count of incorrect =",All_Incorrect[temp_index]


	######### Result #######
	# Max Accuracy: 0.755149221488
	# Standard Deviation: 0.0689775020741
	# LR: 0.122524489796	# may not result in the accuracy above due to precision change
							# if not running double loop, need to find LR in LR_all to 
							# obtain the above Accuracy 
	# epsilon: 1e-05










