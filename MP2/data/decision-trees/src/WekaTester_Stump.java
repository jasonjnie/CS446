package cs446.homework2;

import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import cs446.weka.classifiers.trees.Id3;
import weka.core.converters.ArffSaver;

public class WekaTester_Stump 
{

    static int[] randomIndices (Instances set)
    {
		int indexArray[] = new int[(set.numInstances())/2];
		for (int i = 0; i < (set.numInstances())/2; i++)
		{
			Random randomizer = new Random();
			randomizer.setSeed(System.currentTimeMillis());
			boolean check = false;
			int index = randomizer.nextInt(set.numInstances());
			while (check == false)
			{
				if (checkIfPresent(index, indexArray))
				{
					index = randomizer.nextInt(set.numInstances());
				}
				else
					check = true;
			}
			indexArray[i] = index;
		}
		return indexArray;
    }


    static Instances randomSample (Instances set)
    {
		Instances toReturn = new Instances(set);	//Deep copy
		toReturn.delete();	///////////////////////////////////////////////
		int indicies[] = randomIndices(set);
		for (int i = 0; i < indicies.length; i++)
		{
			toReturn.add(set.instance(indicies[i]));
		}
		return toReturn;
    }

    static boolean checkIfPresent (int index, int choiceArray[])
    {
		for (int i = 0; i < choiceArray.length; i++)
		{
			if (choiceArray[i] == index)
				return true;
		}
		return false;
    }




    public static void main(String[] args) throws Exception 
    {
	    //System.out.println("arg =" + args.length);
		if (args.length != 21) 
		{
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
		Instances all_data = new Instances(new FileReader(new File(args[10])));
		//System.out.println("all_data" + all_data);
		all_data.setClassIndex(all_data.numAttributes() - 1);

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
		int depth = 4;
		double[] Acc_CV = new double[5];

		for(int i=0; i<5; i++)
		{
			// Train on 80% of the data and test on 20%
			//Instances train = data.trainCV(5,0);
			//Instances test = data.testCV(5, 0);

			//Instances train_data = all_train_data[i];
			//Instances test_data = all_test_data[i];

			Instances train_data = new Instances(all_train_data[i]);
			Instances test_data = new Instances(all_test_data[i]);

			Id3 clf_100[] = new Id3[100];
			//double clf_predict_train = new double[int(train_data.numInstances/2)][];
			//double clf_predict_test = new double[test_data.numInstances][];

			int train_Size = (int) Math.round(train_data.numInstances() / 2);

			for(int j=0; j<100; j++)
			{
				System.out.println("num of tree: " + j);
				//Instances half_train_data = randomSample(train_data);
				//Instances copy_instances = new Instances(train_data);
				//copy_instances.randomize(new Random(1));
				///Instances half_train_data = new Instances(copy_instances);	//Deep copy
				//half_train_data.delete();
				//for(int k=0; k<half_train_data.numInstances/2; k++)
				//{
				//	half_train_data.add(copy_instances.instance(k))
				//}

				
				Random rand = new Random(j);   // create seeded number generator
				Instances randData = new Instances(train_data); 
				randData.randomize(rand); 
				Instances half_train_data = new Instances(randData,0,train_Size); 

				//System.out.println("half_train_data = " + half_train_data.numInstances());


				//Instances half_train_data = new Instances(train_data);	//Deep copy
				//half_train_data.delete();	///////////////////////////////////////////////
				//int indicies[] = randomIndices(set);
				//for (int k = 0; k < 100; k++)
				//{
					//System.out.println("k = " + k);
				//	for (int l = k; l < (k+train_data.numInstances()/2); l++)
				//	half_train_data.add(train_data.instance(l));
				//}				




				// Create a new ID3 classifier. This is the modified one where you can
				// set the depth of the tree.
				Id3 classifier = new Id3();

				// An example depth. If this value is -1, then the tree is grown to full
				// depth.
				classifier.setMaxDepth(depth);

				// Train
				classifier.buildClassifier(half_train_data);
				clf_100[j] = classifier;

				// Print the classfier
				//System.out.println(classifier);
				//System.out.println();

				// Evaluate on the test set
				//Evaluation evaluation = new Evaluation(test_data);
				//evaluation.evaluateModel(classifier, test_data);
				//Acc_CV[i] = evaluation.pctCorrect();
				//System.out.println(evaluation.toSummaryString());
			}


			FastVector attributes = new FastVector(3); ///////////////
			String[] features = new String[100];
			for (int k=0; k<100;k++)
			{
				features[k]= Integer.toString(k);
			}

			FastVector zeroOne;
	    	FastVector labels;

	    	zeroOne = new FastVector(2);
			zeroOne.addElement("1");
			zeroOne.addElement("0");

			labels = new FastVector(2);
			labels.addElement("+");
			labels.addElement("-");

			for (String featureName : features) 
			{
			    attributes.addElement(new Attribute(featureName, zeroOne));
			}
			Attribute classLabel = new Attribute("Class", labels);
			attributes.addElement(classLabel);

			String nameOfDataset_train = "NEW_INSTANCES_TRAIN";
			Instances new_instances_train = new Instances(nameOfDataset_train, attributes, 0);
			new_instances_train.setClass(classLabel);
			String nameOfDataset_test = "NEW_INSTANCES_TEST";
			Instances new_instances_test = new Instances(nameOfDataset_test, attributes, 0);
			new_instances_test.setClass(classLabel);

			//loop l num of train_data[len(4_folds)][100]
			int M = train_data.numInstances();	// Num of Instance in current 4 folds
			//int M = 1;



			for (int l=0; l<M; l++)
			{
				int temp_label_count = 0;
				int temp_predict_count = 0;
				Instance instance = new Instance(101);
				instance.setDataset(new_instances_train);

				// prediction of each instance on each tree
				for (int featureId = 0; featureId < 100; featureId++) 
				{
			    	Attribute att = new_instances_train.attribute(features[featureId]);

			    	String featureLabel;
			    	Id3 temp_clf = clf_100[featureId];
			    	//System.out.println("instance = " + train_data.instance(l));
			    	double temp_train_predict = temp_clf.classifyInstance(train_data.instance(l));
			    	//System.out.println("predict = " + temp_train_predict);
			    	
			    	//System.out.println("temp_predict " + temp_predict);
			    	//System.out.println("predict = " + predict);
			    	//Evaluation evaluation = new Evaluation(test_data);
					//evaluation.evaluateModel(clf, test_data);
					//Acc_CV[i] = evaluation.pctCorrect();

			    	/*
				   	if( temp_train_predict == 1.0)
			    	{
			    		temp_predict_count += 1;
			    	}
			    	if (train_data.instance(l).classValue() == 0.0)
			    	{
			    		temp_label_count += 1;
		    		}
					*/

		    		
			    	if (temp_train_predict > 0) ////////////// >0
			    	{
						featureLabel = "1";
			    	} 
			    	else
			    	{
			    		featureLabel = "0";
			    	}
			    	
					//featureLabel = double.toString(temp_train_predict);
			    	instance.setValue(att, featureLabel);////////////////////
			    	//instance.setValue(att, temp_train_predict);
			    }

	    		//System.out.println("temp_predict_count = " + temp_predict_count);
	    		//System.out.println("true label = " + train_data.instance(l).classValue());

				/*
				double label;
				System.out.println("classvalue = " + train_data.instance(l).classValue());
				if (train_data.instance(l).classValue() == 0.0)
				{
					//System.out.println("temp" + "-1");
					label = 0;
				}
				else
				{
					//System.out.println("temp" + "1");
					label = 1;
				}
				*/
				//System.out.println("orig_label "+orig_label);
				
				//instance.setClassValue(label);

				instance.setClassValue(train_data.instance(l).classValue());
			    new_instances_train.add(instance);
			}

			//loop k num of test_data[len(1_fold)][100]
			int N = test_data.numInstances();	// Num of Instance in current 1 test fold
			for (int k=0; k<N; k++)
			{
				Instance instance = new Instance(101);
				instance.setDataset(new_instances_test);

				// prediction of each instance on each tree
				for (int featureId = 0; featureId < 100; featureId++) 
				{
			    	Attribute att = new_instances_test.attribute(features[featureId]);
			    	String featureLabel;
			    	Id3 temp_clf = clf_100[featureId];
			    	double temp_test_predict = temp_clf.classifyInstance(test_data.instance(k));
			    	
			    	if (temp_test_predict > 0) 
			    	{
						featureLabel = "1";
			    	} 
			    	else
			    	{
			    		featureLabel = "0";
			    	}
					
			    	//featureLabel = double.toString(temp_test_predict);
			    	instance.setValue(att, featureLabel);
				}
				instance.setClassValue(test_data.instance(k).classValue());
			    new_instances_test.add(instance);
			}


			ArffSaver saver_train = new ArffSaver();
			saver_train.setInstances(new_instances_train);
			saver_train.setFile(new File(args[11+i*2]));
			saver_train.writeBatch();	

			ArffSaver saver_test = new ArffSaver();
			saver_test.setInstances(new_instances_test);
			saver_test.setFile(new File(args[11+i*2+1]));
			saver_test.writeBatch();	
		}
	}
}
	//double Accuracy = (Acc_CV[0]+Acc_CV[1]+Acc_CV[2]+Acc_CV[3]+Acc_CV[4]) / 5;
	//System.out.println("DT with depth: " + depth + " has accuracy : " + Accuracy);









///////////// Result ////////////////
// DT with depth = -1: Accuracy = 72.22916900033604
// DT with depth = 4: Accuracy = 65.63546217321962
// DT with depth = 8: Accuracy = 69.67298224277629








