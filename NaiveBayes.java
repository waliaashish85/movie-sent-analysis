import java.util.*;
import java.util.regex.*;
import java.io.*;

public class NaiveBayes {
	private  Map<String,Map<String,Integer>> consolidatedDictionary = new HashMap<String,Map<String,Integer>>();
	private Map<String,Integer> sortedMapOnValues(Map<String,Integer>map)
	{
		if(map!=null)
		{
			List list = new ArrayList(map.entrySet());
			//reverse sort list based on comparator
			Collections.sort(list, new Comparator(){
				public int compare(Object o1, Object o2) {
					return ((Comparable) ((Map.Entry) (o2)).getValue())
					.compareTo(((Map.Entry) (o1)).getValue());
				}
			});

			//put sorted list into map again
			Map sortedMap = new LinkedHashMap();
			for (Iterator it = list.iterator(); it.hasNext();) {
				Map.Entry entry = (Map.Entry)it.next();
				sortedMap.put(entry.getKey(), entry.getValue());				
			}
			return sortedMap;
		}
		else
		{
			return null;
		}
	}	
	public void train(List<String>wordsList,String category)
	{
		Map<String, Integer> freqMap;
		if(consolidatedDictionary.containsKey(category))
		{
			freqMap = consolidatedDictionary.get(category);
		}
		else
		{
			freqMap = new HashMap<String,Integer>();
		}
		for(String word:wordsList)
		{
			Integer count = freqMap.get(word);
			freqMap.put(word, (count == null ? 1 : count + 1));
		}
		consolidatedDictionary.put(category, freqMap);
	}
	private int vocabSize()
	{
		Set vocabSet = new HashSet();
		for(Map.Entry<String,Map<String,Integer>>categorisedDictEntry:consolidatedDictionary.entrySet())
		{
			if(categorisedDictEntry!=null)
			{
				Map<String,Integer> freqMap = (Map<String, Integer>)categorisedDictEntry.getValue();
				if(freqMap!=null)
				{
					for(Map.Entry<String,Integer>currentCategoryEntry:freqMap.entrySet())
					{
						if(currentCategoryEntry!=null)
						{
							vocabSet.add(currentCategoryEntry.getKey());
						}
					}
				}
			}
		}
		return vocabSet.size();
	}

	private int getSumOfFrequencies(String category)
	{
		if(category!=null)
		{
			Map<String,Integer> frequencyMap;
			for(Map.Entry<String,Map<String,Integer>> pairs:consolidatedDictionary.entrySet())
			{
				if(pairs!=null)
				{
					String ctgry = (String)pairs.getKey();
					frequencyMap = (Map<String,Integer>)pairs.getValue();
					if(frequencyMap!=null && ctgry!=null && ctgry.equals(category))
					{
						int sumOfFrequencies = 0;
						for(Map.Entry<String,Integer>categorisedFreqCount:frequencyMap.entrySet())
						{
							sumOfFrequencies+=categorisedFreqCount.getValue();
						}
						return sumOfFrequencies;
					}
				}
			}
		}
		return 0;
	}

	public String classifyDocBySentences(List<String>sentencesList)
	{
		int vocabSize = vocabSize();
		int sumOfPosFrequencies = getSumOfFrequencies("pos");
		int sumOfNegFrequencies = getSumOfFrequencies("neg");
		int pos_count=0;
		int neg_count=0;
		int frequency=0;
		Map<String,Integer> frequencyMap;
		for(String sentence:sentencesList)
		{
			if(sentence!=null)
			{
				String words[] = sentence.split("\\s+");
				double posBayesProbability = 0.0;
				double negBayesProbability = 0.0;
				double bayesProbability = 0.0;
				int sumOfFrequencies = 0;
				for(Map.Entry<String,Map<String,Integer>> pairs:consolidatedDictionary.entrySet())
				{
					if(pairs!=null)
					{					
						String category = (String)pairs.getKey();
						frequencyMap = (Map<String,Integer>)pairs.getValue();
						if(frequencyMap!=null)
						{
							if("pos".equals(category))
							 {
								sumOfFrequencies=sumOfPosFrequencies;
							 }
							 else if("neg".equals(category))
							 {
								 sumOfFrequencies=sumOfNegFrequencies;
							 }
							for(String word: words)
							{
								frequency = frequencyMap.get(word)==null?0:frequencyMap.get(word);
								bayesProbability+=Math.log10((double)(frequency+1)/(sumOfFrequencies+vocabSize)); //Laplace							
							}
							int categories = consolidatedDictionary.entrySet().size();
							bayesProbability+=Math.log10((double)1/categories); //Prior
							if("pos".equals(category))
							 {
								 posBayesProbability=bayesProbability;
							 }
							 else if("neg".equals(category))
							 {
								 negBayesProbability=bayesProbability;
							 }
						}							
					}
				}
				if(posBayesProbability>negBayesProbability)
				{
					pos_count++;
				}
				else
				{
					neg_count++;
				}
			}
		}
		if(pos_count>neg_count)
		{
			return "pos";
		}
		else if(neg_count>pos_count)
		{
			return "neg";
		}
		return null;
	}
	public String classifyDocByUnigramWords(List<String>wordsList)
	{
		Map<String, Double> conditionalProbability = new HashMap<String,Double>();
		Map<String,Integer> frequencyMap;
		int frequency=0;
		double bayesProbability = 0.0;
		int vocabSize = vocabSize();
		Iterator iterator = (Iterator)consolidatedDictionary.entrySet().iterator();
		while(iterator.hasNext())
		{
			Map.Entry pairs = (Map.Entry)iterator.next();
			if(pairs!=null)
			{
				String category = (String)pairs.getKey();
				frequencyMap = sortedMapOnValues((Map<String,Integer>)pairs.getValue());
				if(frequencyMap!=null)
				{
					int sumOfFrequencies = 0;
					for(Map.Entry<String,Integer>categorisedFreqCount:frequencyMap.entrySet())
					{
						sumOfFrequencies+=categorisedFreqCount.getValue();
					}
					for(String word:wordsList)
					{
						frequency = frequencyMap.get(word)==null?0:frequencyMap.get(word);
						bayesProbability+=Math.log10((double)(frequency+1)/(sumOfFrequencies+vocabSize)); //Laplace
					}
					int categories = consolidatedDictionary.entrySet().size();
					bayesProbability+=Math.log10((double)1/categories); //Prior
					conditionalProbability.put(category, bayesProbability);
				}
				else
				{
					System.err.println("Frequency Map is null for category "+category);
				}			
			}
			else
			{
				return null;
			}
		}
		double maxVal = -10000000000000000.00;
		String probableCategory = null;
		//Return the category with max value of bayes probability
		for(Map.Entry<String, Double>probability:conditionalProbability.entrySet())
		{
			if(probability.getValue()>maxVal)
			{
				maxVal = probability.getValue();
				probableCategory=probability.getKey();
			}
		}
		return probableCategory;
	}
}