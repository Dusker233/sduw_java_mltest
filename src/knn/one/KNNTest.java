package knn.one;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Random;
import java.util.Scanner;

import javax.swing.JOptionPane;

import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.filter.normalize.NormalizeMidrange;
import net.sf.javaml.tools.data.FileHandler;

public class KNNTest
{
	
	public double TrainandClassify(int k, Dataset train, Dataset test)
	{
		KNN knn = new KNN();
		knn.Init(k);
		knn.Build(train);
		Random r = new Random();
		int size = test.size();
		int sigma = r.nextInt(1000);
		int[] ind = new int[sigma];
		for(int i = 0;i < sigma;i++)
			ind[i] = r.nextInt(size);
		int correct = 0;
		for(int i = 0;i < sigma;i++)
		{
			Instance inst = test.instance(ind[i]);
			System.out.println(inst);
			Object predicted = knn.classify(inst);
			System.out.println("Predicted Ans: " + predicted + ", Actual Ans: " + inst.classValue());
			if(predicted.equals(inst.classValue()))
				correct++;
		}
		double C = correct;
		return C / sigma;
		
	}
	
	public static void main(String args[]) throws IOException
	{
		Dataset data = FileHandler.loadDataset(new File("./resources/data/SIMULATED_00011.csv"), 5, ",");
        Dataset testSet = FileHandler.loadDataset(new File("./resources/data/SIMULATED_00011.csv"), 5, ",");
        NormalizeMidrange nmr = new NormalizeMidrange(1, 2);
        //normalization [0, 2]
        nmr.build(data);
        nmr.filter(data);
        nmr.filter(testSet);
        System.out.print("Please input k:");
        int k,  iter;
        Scanner sc = new Scanner(System.in);
        k = sc.nextInt();
        System.out.print("Please input number of tests:");
        iter = sc.nextInt();
        new KNNTest().TrainandClassifyMultiTime(k, iter, data, testSet);
        sc.close();
	}

	private void TrainandClassifyMultiTime(int k, int iter, Dataset data, Dataset testSet)
	{
		double rate = 0;
		for(int i = 1;i <= iter;i++)
			rate += new KNNTest().TrainandClassify(k, data, testSet);
		DecimalFormat df = new DecimalFormat("0.000");
		JOptionPane.showMessageDialog(null,
				"In " + iter + " times, Correct rate = " + df.format(rate / iter * 100) + "%",
				"Classify Result",
				JOptionPane.INFORMATION_MESSAGE);
	}
}