package test.one;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import net.sf.javaml.classification.KDtreeKNN;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.tools.data.FileHandler;

public class Test
{

	public void trainAndClassify(int k, Dataset train, Dataset test)
	{
		KDtreeKNN knn = new KDtreeKNN(k);
		knn.buildClassifier(train);
		int cnt = 0;
		int size = test.size();
		for(int i = 0;i < test.size();i++)
		{
			Instance inst = test.instance(i);
			System.out.println(inst);
			Object classifyed = knn.classify(inst);
			System.out.println("classifyAns: " + classifyed + ", ans: " + inst.classValue());
			cnt += (classifyed.equals(inst.classValue()) ? 1 : 0);
		}
		System.out.println("in " + size + ": " + (double) (cnt) / size * 100 + "%");
	}
	
	public static void main(String[] args) throws IOException
	{
		// TODO Auto-generated method stub
		Dataset data = FileHandler.loadDataset(new File("./resources/data/SIMULATED_00005.csv"), 5, ",");
        Dataset testSet = FileHandler.loadDataset(new File("./resources/data/SIMULATED_00004.csv"), 5, ",");
        new Test().trainAndClassify(100, data, testSet);
	}

}
