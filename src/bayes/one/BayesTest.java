package bayes.one;

import java.io.File;
import java.io.IOException;

import net.sf.javaml.classification.bayes.NaiveBayesClassifier;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.filter.normalize.NormalizeMidrange;
import net.sf.javaml.tools.data.FileHandler;

public class BayesTest
{

	public void TrainAndClassify(Dataset train, Dataset test)
	{
		NaiveBayesClassifier nbc = new NaiveBayesClassifier(true, false, true);
		nbc.buildClassifier(train);
		for(Instance i: test)
		{
			Object classifyed = nbc.classify(i);
			System.out.println(classifyed + " " + i.classValue());
		}
	}
	
	public static void main(String[] args)throws IOException
	{
		// TODO Auto-generated method stub
		Dataset data = FileHandler.loadDataset(new File("./resources/data/SIMULATED_00011.csv"), 5, ",");
        Dataset testSet = FileHandler.loadDataset(new File("./resources/data/SIMULATED_00011.csv"), 5, ",");
        NormalizeMidrange nmr = new NormalizeMidrange(1, 2);
        //normalization [0, 2]
        nmr.build(data);
        nmr.filter(data);
        nmr.filter(testSet);
        new BayesTest().TrainAndClassify(data, testSet);
	}

}
