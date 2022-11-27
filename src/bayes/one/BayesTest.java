package bayes.one;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import net.sf.javaml.classification.bayes.NaiveBayesClassifier;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.filter.normalize.NormalizeMidrange;
import net.sf.javaml.tools.data.FileHandler;

public class BayesTest
{

	public void TrainAndClassify(Dataset train, Dataset test)
	{
		NaiveBayesClassifier nbc = new NaiveBayesClassifier(true, true, false);
		nbc.buildClassifier(train);
		double correct = 0;
		int all = test.size();
		for(Instance i: test)
		{
			Object classifyed = nbc.classify(i);
			System.out.println(classifyed + " " + i.classValue());
			if(classifyed.equals(i.classValue()))
				correct++;
		}
		System.out.println(correct / all * 100 + "%");
	}
	
	public static void main(String[] args)throws IOException
	{
		// TODO Auto-generated method stub
		Dataset d = FileHandler.loadDataset(new File("./resources/data/SIMULATED_00011.csv"), 5, ",");
        NormalizeMidrange nmr = new NormalizeMidrange(1, 2);
        //normalization [0, 2]
        nmr.build(d);
        nmr.filter(d);
        Dataset folds[] = d.folds(2, new Random());
        Dataset data = folds[0], testSet = folds[1];
        new BayesTest().TrainAndClassify(data, testSet);
	}

}
