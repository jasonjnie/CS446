#!/bin/bash

mkdir bin

make

# Generate the example features (first and last characters of the
# first names) from the entire dataset. This shows an example of how
# the featurre files may be built. Note that don't necessarily have to
# use Java for this step.

# Generate test dataset
#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.all ./../badges.modified.data.all.arff
#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold1 ./../badges.modified.data.fold1.arff
#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold2 ./../badges.modified.data.fold2.arff
#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold3 ./../badges.modified.data.fold3.arff
#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold4 ./../badges.modified.data.fold4.arff
#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold5 ./../badges.modified.data.fold5.arff

# Concatenate four dataset
#cat ../badges/badges.modified.data.fold2 ../badges/badges.modified.data.fold3 ../badges/badges.modified.data.fold4 ../badges/badges.modified.data.fold5 > fold2345
#cat ../badges/badges.modified.data.fold1 ../badges/badges.modified.data.fold3 ../badges/badges.modified.data.fold4 ../badges/badges.modified.data.fold5 > fold1345
#cat ../badges/badges.modified.data.fold1 ../badges/badges.modified.data.fold2 ../badges/badges.modified.data.fold4 ../badges/badges.modified.data.fold5 > fold1245
#cat ../badges/badges.modified.data.fold1 ../badges/badges.modified.data.fold2 ../badges/badges.modified.data.fold3 ../badges/badges.modified.data.fold5 > fold1235
#cat ../badges/badges.modified.data.fold1 ../badges/badges.modified.data.fold2 ../badges/badges.modified.data.fold3 ../badges/badges.modified.data.fold4 > fold1234

# Generate training dataset
#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator fold2345 ./../fold2345.arff
#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator fold1345 ./../fold1345.arff
#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator fold1245 ./../fold1245.arff
#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator fold1235 ./../fold1235.arff
#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator fold1234 ./../fold1234.arff

# Using the features generated above, train a decision tree classifier
# to predict the data. This is just an example code and in the
# homework, you should perform five fold cross-validation. 
#java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.example.arff
java -cp lib/weka.jar:bin cs446.homework2.WekaTester  ./../fold2345.arff ./../fold1345.arff ./../fold1245.arff ./../fold1235.arff ./../fold1234.arff ./../badges.modified.data.fold1.arff ./../badges.modified.data.fold2.arff ./../badges.modified.data.fold3.arff ./../badges.modified.data.fold4.arff ./../badges.modified.data.fold5.arff 
#java -cp lib/weka.jar:bin cs446.homework2.WekaTester_Stump  ./../fold2345.arff ./../fold1345.arff ./../fold1245.arff ./../fold1235.arff ./../fold1234.arff ./../badges.modified.data.fold1.arff ./../badges.modified.data.fold2.arff ./../badges.modified.data.fold3.arff ./../badges.modified.data.fold4.arff ./../badges.modified.data.fold5.arff ./../badges.modified.data.all.arff out_train_1.arff out_test_1.arff out_train_2.arff out_test_2.arff out_train_3.arff out_test_3.arff out_train_4.arff out_test_4.arff out_train_5.arff out_test_5.arff






