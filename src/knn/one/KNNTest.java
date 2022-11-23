package knn.one;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Random;
import java.util.Scanner;

import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.filter.normalize.NormalizeMidrange;
import net.sf.javaml.tools.data.FileHandler;

public class KNNTest
{
	
	public void TrainandClassify(int k, Dataset train, Dataset test)
	{
		KNN knn = new KNN();
		knn.Init(k);
		knn.Build(train);
		Random r = new Random();
		int size = test.size();
		int sigma = 200;
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
		try
		{
			FileWriter fw = new FileWriter("./Real_Ans.csv");
			//TODO output data for visualization
			fw.write("Real_index,Real_class\n");
			for(int i = 0;i < sigma;i++)
			{
				Instance inst = test.instance(ind[i]);
				fw.write(i + "," + inst.classValue() + "\n");
			}
			fw.close();
			fw = new FileWriter("./Predicted_Ans.csv");
			fw.write("Predicted_index,Predicted_class\n");
			for(int i = 0;i < sigma;i++)
			{
				Instance inst = test.instance(ind[i]);
				fw.write(i + "," + knn.classify(inst) + "\n");
			}
			fw.close();
		}
		catch(IOException e)
		{
			System.out.println("Something went wrong");
			e.printStackTrace();
		}
		System.out.println("Correct Rate: " + (double)correct / sigma * 100 + "%");
	}
	
	public static void main(String args[]) throws IOException
	{
		Dataset d = FileHandler.loadDataset(new File("./resources/SIMULATED_00004.csv"), 4, ",");
        NormalizeMidrange nmr = new NormalizeMidrange(1, 2);
        //normalization [0, 2]
        nmr.build(d);
        nmr.filter(d);
        Dataset[] folds = d.folds(2, new Random());
        Dataset data = folds[0], testSet = folds[1];
        System.out.print("Please input k:");
        int k,  iter;
        Scanner sc = new Scanner(System.in);
        k = sc.nextInt();
        new KNNTest().TrainandClassify(k, data, testSet);
        sc.close();
	}
}