package knn.one;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;

public class KNN
{
	private Dataset d;
	private int k;
	
	public void Init(int _k) {k = _k;}
	public void Build(Dataset train) {d = train;}
	
	private double GetEuclidDistance(Instance x, Instance y)
	{
		if(x.noAttributes() != y.noAttributes())
			throw new RuntimeException("Both instance should have the same number of values, Error!");
		double Dis = 0;
		for(int i = 0;i < x.noAttributes();i++)
		{
			if(!Double.isNaN(x.value(i)) && !Double.isNaN(y.value(i)))
					Dis += (y.value(i) - x.value(i)) * (y.value(i) - x.value(i));
		}
		return Math.sqrt(Dis);		
	}
	
	private Set<Instance> kNearest(Instance inst)
	{
		Set<Instance> ExpectedInstance = new HashSet<>();
		HashMap<Double, Instance> dis = new HashMap<>();
		for(Instance Candidate: d)
		{
			double dist = GetEuclidDistance(Candidate, inst);
			dis.put(dist, Candidate);
		}
		TreeMap<Double, Instance> SortedDis = new TreeMap<>();
		SortedDis.putAll(dis);
		int count = 0;
		for(Map.Entry<Double, Instance> e: SortedDis.entrySet())
		{
			if(!ExpectedInstance.contains(e.getValue()))
			{
				ExpectedInstance.add(e.getValue());
				count++;
				if(count == k)
					break;
			}
		}
		return ExpectedInstance;
	}
	
	public Object classify(Instance inst)
	{
		if(d == null)
			throw new RuntimeException("Training dataset is null");
		Set<Instance> NearNeighbors = kNearest(inst);
		Object[] ExpectedClass = new Object[k];
		int index = 0;
		for(Instance i: NearNeighbors)
			ExpectedClass[index++] = i.classValue();
		HashMap<Object, Integer> map = new HashMap<>();
		for(Object i: ExpectedClass)
		{
			if(map.containsKey(i))
			{
				int tmp = map.get(i);
				map.put(i, tmp + 1);
			}
			else
				map.put(i, 1);
		}
		Collection<Integer> count = map.values();
		int Maxcount = Collections.max(count);
		for(Map.Entry<Object, Integer> e: map.entrySet())
			if(Maxcount == e.getValue())
				return e.getKey();
		return null;
	}	
}
