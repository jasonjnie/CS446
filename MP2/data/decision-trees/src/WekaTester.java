package cs446.homework2;

import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.lang.Math;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import cs446.weka.classifiers.trees.Id3;

public class WekaTester {

    public static void main(String[] args) throws Exception {

	if (args.length != 10) {
	    System.err.println("Usage: WekaTester arff-file");
	    System.exit(-1);
	}


/////////////////// Parse Input Training and Testing Dataset /////////////
// index = 0 -> train fold 2345, test on fold 1
// index = 1 -> train fold 1345, test on fold 2
// index = 2 -> train fold 1245, test on fold 3
// index = 3 -> train fold 1235, test on fold 4
// index = 4 -> train fold 1234, test on fold 5

	Instances[] all_train_data = new Instances[5];
	Instances[] all_test_data = new Instances[5];
	for(int i=0; i<5; i++)
	{
		Instances temp_data_fold = new Instances(new FileReader(new File(args[i])));
		// The last attribute is the class label
		temp_data_fold.setClassIndex(temp_data_fold.numAttributes() - 1);
		all_train_data[i] = temp_data_fold;
		Instances temp_test_fold = new Instances(new FileReader(new File(args[i+5])));
		temp_test_fold.setClassIndex(temp_test_fold.numAttributes() - 1);
		all_test_data[i] = temp_test_fold;		
	}
	//System.out.println("Attribute" + data_fold1.numAttributes());		// Attribute = 261
	//System.out.println("Instances" + data_fold1.numInstances());		// Instances = 65

	// Choose appropriate DT depth
	int depth = 8;	// full tree
	//int depth = 4;
	//int depth = 8;
	double[] Acc_CV = new double[5];
	Evaluation[] All_Evaluation = new Evaluation[5];
	Id3[] All_Clf = new Id3[5];

	for(int i=0; i<5; i++)
	{
		// Train on 80% of the data and test on 20%
		//Instances train = data.trainCV(5,0);
		//Instances test = data.testCV(5, 0);

		Instances train_data = all_train_data[i];
		Instances test_data = all_test_data[i];

		// Create a new ID3 classifier. This is the modified one where you can
		// set the depth of the tree.
		Id3 classifier = new Id3();

		// An example depth. If this value is -1, then the tree is grown to full
		// depth.
		classifier.setMaxDepth(depth);

		// Train
		classifier.buildClassifier(train_data);
		All_Clf[i] = classifier;

		// Print the classfier
		//System.out.println(classifier);
		//System.out.println();

		// Evaluate on the test set
		Evaluation evaluation = new Evaluation(test_data);
		evaluation.evaluateModel(classifier, test_data);
		Acc_CV[i] = evaluation.pctCorrect()/100;
		All_Evaluation[i] = evaluation;
		
	}


	double Accuracy = (Acc_CV[0]+Acc_CV[1]+Acc_CV[2]+Acc_CV[3]+Acc_CV[4]) / 5;
	System.out.println("DT with depth: " + depth + " has accuracy : " + Accuracy);
	double Std = 0;
	for(int k=0; k<5; k++)
	{
		Std += (Acc_CV[k]-Accuracy) * (Acc_CV[k]-Accuracy);
	}
	Std = Math.sqrt(Std / 5);
	System.out.println("DT with depth: " + depth + " has Std : " + Std);
	System.out.println("Accuracy over 5 Folds: " + Acc_CV[0] + Acc_CV[1] + Acc_CV[2] + Acc_CV[3] + Acc_CV[4]);
	
	int temp_index = 0;
	double temp_Acc = Acc_CV[0];
	for(int i=0; i<5; i++)
	{
		if (Acc_CV[i] > temp_Acc)
		{
			temp_Acc = Acc_CV[i];
			temp_index = i;
		}
	}
	System.out.println(All_Clf[temp_index]);
	System.out.println(All_Evaluation[temp_index].toSummaryString());



	//System.out.println("Acc_CV = " + Acc_CV[0] + Acc_CV[1] + Acc_CV[2] + Acc_CV[3] + Acc_CV[4]);
    }
}


///////////// Result ////////////////
// DT with depth = -1: Accuracy = 72.22916900033604
// DT with depth = 4: Accuracy = 65.63546217321962
// DT with depth = 8: Accuracy = 69.67298224277629








