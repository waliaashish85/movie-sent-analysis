import java.util.*;
import java.io.*;
import opennlp.tools.doccat.*;
import opennlp.tools.util.*;
import rita.*;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.bayes.net.estimate.MultiNomialBMAEstimator;
import weka.core.Instances;
import weka.core.converters.TextDirectoryLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stemmers.LovinsStemmer;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToBinary;
import weka.filters.unsupervised.attribute.StringToWordVector;
public class SentimentAnalysis {

	private static Map <String,Integer>positivePolarityDict = new HashMap<String,Integer>();
	private static Map <String,Integer>negativePolarityDict = new HashMap<String,Integer>();
	private static Map <String,Integer>dictForTestData = new HashMap<String,Integer>();
	private static Map<String,Map<String,Integer>> consolidatedDictionary = new HashMap<String,Map<String,Integer>>();
	private static String punctuations[]={",",".","!","?","'",":","*","\"","{","}","[","]","(",")","-"};
	private static List punctuationsList = Arrays.asList(punctuations);
	//Prepared list of stop words using NLTK
	//>>> from nltk.corpus import stopwords
	//>>> stopwords.words('english')
	private static String stopWords[] = {"i","me","myself","we","our","ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"};
	private static List stopWordsList = Arrays.asList(stopWords);	
	//Polarity data set of movie reviews is divided into 10 folds

	private static List<Split> prepareDataForTenFoldsCrossValidation(List<File> files)
	{
		if(files!=null)
		{
			List<Split>foldsList = new ArrayList<Split>();
			for(int i=0;i<10;i++)
			{
				Split split = new Split();
				split.setSplitNumber(i);
				for(File file:files)
				{
					if(file!=null)
					{
						if( file.getName().substring(2,3).equals(String.valueOf(i)))
						{
							split.addFilesToTestingFilesList(file);
						}
						else
						{
							split.addFilesToTrainingFilesList(file);
						}
					}
				}
				foldsList.add(split);
			}
			return foldsList;
		}
		else
		{
			return null;
		}
	}
	private static String concatenateContentsOfList(List<String> list)
	{
		if(list!=null)
		{
			StringBuilder content = new StringBuilder();
			for(String string:list)
			{
				if(string!=null)
				{
					content.append(string).append(" ");
				}
			}
			return content.toString();
		}
		else
		{
			return null;
		}
	}
	private static List<String> readFile(File file)
	{
		if(file!=null)
		{
			try
			{
				List<String> contentsList = new ArrayList<String>();
				BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
				StringBuilder contents = new StringBuilder();
				String line = null;
				while((line = bufferedReader.readLine())!=null)
				{
					contentsList.add(line);
				}
				return contentsList;
			}
			catch(FileNotFoundException fileNotfoundException)
			{
				System.err.println("File not found.");
			}
			catch(IOException ioException)
			{
				System.err.println("IO Exception occured while reading file "+ file.getName());
			}
		}
		return null;
	}
	private static List<String>segmentContent(List<String> contentsList, boolean stemming, boolean noStopWords, boolean noPunctuation)
	{
		if(contentsList!=null)
		{
			List<String> wordsList = new ArrayList<String>();
			for(String line : contentsList)
			{
				if(line!=null)
				{
					String words[]=line.split("\\s+");
					for(String word:words)
					{
						String finalWord = word;
						if(noStopWords && stopWordsList.contains(word))
						{
							finalWord=null;
						}
						if(noPunctuation && punctuationsList.contains(word))
						{
							finalWord=null;
						}
						if(stemming && finalWord!=null)
						{
							RiStemmer stemmer = new RiStemmer();
							finalWord = stemmer.stem(finalWord);
						}
						if(finalWord!=null)
						{
							wordsList.add(finalWord);
						}
					}

				}
			}
			return wordsList;
		}
		else
		{
			return null;
		}
	}
	//Using Maximum Entropy to classify text as positive or negative
	private static void runOpenNLPDocClassifierBasedOnMaxEnt(List<Split>foldsList, boolean stemming, boolean noStopWords, boolean noPunctuation)
	{
		int fold =0 ;
		int correctPrediction=0;
		int wrongPrediction=0;
		double avgAccuracy = 0.0;
		System.out.println("**************************************************************************************************************");
		System.out.println("Testing OpenNLP's MaxEnt based classifier using ten folds cross validation.");
		System.out.println("Used Stemmed Words = "+stemming+" ; Removed Stop Words = "+noStopWords +" ; Removed punctuations = "+noPunctuation);
		System.out.println("**************************************************************************************************************");
		for(Split split:foldsList)
		{
			fold++;
			//Training files
			StringBuilder trainingContent = new StringBuilder();
			for(File file: split.getTrainingFilesList())
			{
				String category = file.getParentFile().getName();
				List<String>wordsList=segmentContent(readFile(file), stemming, noStopWords, noPunctuation);
				trainingContent.append(category+" ").append(concatenateContentsOfList(wordsList)).append("\n");				
			}
			//Build training file as per format desired by OpenNLP DocCat Tool(http://opennlp.apache.org/documentation/1.5.2-incubating/manual/opennlp.html#tools.doccat.training.tool)
			try
			{
				BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter("fold"+String.valueOf(fold)+".train"));
				bufferedWriter.write(trainingContent.toString());
				bufferedWriter.close();
			}
			catch(IOException ioException)
			{
				System.err.println("Exception while writing file "+String.valueOf(fold)+".train");
			}

			//Reading the training file to train Open NLP's Doc Cat Tool
			DoccatModel model = null;
			InputStream dataIn = null;
			try {
				dataIn = new FileInputStream("fold"+String.valueOf(fold)+".train");
				ObjectStream<String> lineStream = new PlainTextByLineStream(dataIn, "UTF-8");
				ObjectStream<DocumentSample> sampleStream = new DocumentSampleStream(lineStream);
				model = DocumentCategorizerME.train("en", sampleStream);
			}
			catch (IOException e) {
				// Failed to read or parse training data, training failed
				e.printStackTrace();
			}
			finally {
				if (dataIn != null) {
					try {
						dataIn.close();
					}
					catch (IOException e) {
						// Not an issue, training already finished.
						// The exception should be logged and investigated
						// if part of a production system.
						e.printStackTrace();
					}
				}
			}
			//It's time to prepare model file
			OutputStream modelOut = null;
			try {
				modelOut = new BufferedOutputStream(new FileOutputStream("fold"+String.valueOf(fold)+".model"));
				model.serialize(modelOut);
			}
			catch (IOException e) {
				// Failed to save model
				e.printStackTrace();
			}
			finally {
				if (modelOut != null) {
					try {
						modelOut.close();
					}
					catch (IOException e) {
						// Failed to correctly save model.
						// Written model might be invalid.
						e.printStackTrace();
					}
				}
			}
			//It's time to read the model file, so that we can classify the movie reviews
			InputStream inputStream = null;
			DoccatModel doccatModel = null;
			try
			{
				inputStream=new FileInputStream("fold"+String.valueOf(fold)+".model");
				doccatModel=new DoccatModel(inputStream);
			}
			catch(FileNotFoundException fileNotFoundException)
			{
				System.err.println("fold"+String.valueOf(fold)+".model file not found.");
			}
			catch(IOException ioException)
			{
				System.err.println("Exception while reading file fold"+String.valueOf(fold)+".model");
			}			
			correctPrediction=0;
			wrongPrediction=0;
			//Testing File
			for(File file:split.getTestingFilesList())
			{
				String category = file.getParentFile().getName();
				List<String>wordsList=segmentContent(readFile(file),stemming, noStopWords, noPunctuation);
				String inputText = concatenateContentsOfList(wordsList);
				DocumentCategorizerME myCategorizer = new DocumentCategorizerME(doccatModel);
				double[] outcomes = myCategorizer.categorize(inputText);
				String predictedCategory = myCategorizer.getBestCategory(outcomes);
				if(category.equals(predictedCategory))
				{
					correctPrediction++;
				}
				else
				{
					wrongPrediction++;
				}
			}
			double accuracy = (double)(correctPrediction*1.0/(correctPrediction+wrongPrediction));
			avgAccuracy+=accuracy;
			System.out.println("Fold " + fold + " Accuracy: " + accuracy*100 + "%");
		}
		avgAccuracy = (double)avgAccuracy / foldsList.size();
		System.out.println("Accuracy: " + avgAccuracy*100);
	}
	private static void runSelfWrittenMultinomialNaiveBayes(List<Split>foldsList,boolean stemming, boolean noStopWords, boolean noPunctuation)
	{
		int fold =0 ;
		int correctPrediction=0;
		int wrongPrediction=0;
		double avgAccuracy = 0.0;
		System.out.println("******************************************************************************************************************************");
		System.out.println("Testing indigineously built Multinomial Naive Bayes Classifier (with Laplace smoothing) using ten folds cross validation.");
		System.out.println("Used Stemmed Words = "+stemming+" ; Removed Stop Words = "+noStopWords +" ; Removed punctuations = "+noPunctuation);
		System.out.println("******************************************************************************************************************************");
		//Train and classify Naive Bayes Classifier by employing ten folds cross validation
		for(Split split:foldsList)
		{
			fold++;
			NaiveBayes naiveBayesClassifier = new NaiveBayes();
			//Training files
			for(File file: split.getTrainingFilesList())
			{
				String category = file.getParentFile().getName();
				//Set 2nd argument as true if you want stemmed words in dictionary
				//Set 3rd argument as true if you don't want stop words in dictionary
				//Set 4th argument as true if you don't want punctuations in dictionary
				List<String>wordsList=segmentContent(readFile(file), stemming, noStopWords, noPunctuation);
				naiveBayesClassifier.train(wordsList,category);
			}
			correctPrediction=0;
			wrongPrediction=0;
			//Testing File
			for(File file:split.getTestingFilesList())
			{
				String category = file.getParentFile().getName();
				//Set 2nd argument as true if you want stemmed words in dictionary
				//Set 3rd argument as true if you don't want stop words in dictionary
				//Set 4th argument as true if you don't want punctuations in dictionary
				List<String>wordsList=segmentContent(readFile(file), stemming, noStopWords, noPunctuation);
				String predictedCategory = naiveBayesClassifier.classifyDocBySentences(readFile(file));
				if(category.equals(predictedCategory))
				{
					correctPrediction++;
				}
				else
				{
					wrongPrediction++;
				}
			}
			double accuracy = (double)(correctPrediction*1.0/(correctPrediction+wrongPrediction));
			avgAccuracy+=accuracy;
			System.out.println("Fold " + fold + " Accuracy: " + accuracy*100 + "%");
		}
		avgAccuracy = (double)avgAccuracy / foldsList.size();
		System.out.println("Accuracy: " + avgAccuracy*100);
	}
	private static void runNaiveBayesClassificationUsingWeka(String location,boolean stemming, boolean noStopWords,boolean binarize)
	{
		//Option 1 : To read from .arff file
		/*
		try
		{
		DataSource source = new DataSource(locationOfArff);
		Instances data = source.getDataSet();
		// Setting class attribute if the data format does not provide this information
		 // For example, the XRFF format saves the class attribute information as well
		 if (data.classIndex() == -1)
		 {
		   data.setClassIndex(data.numAttributes() - 1);
		 }
		}
		catch(Exception exception)
		{
			System.err.println("Exception in reading .arff file.");
		}
		 */
		System.out.println("*****************************************************************************************************");
		System.out.println("Testing Weka's Multinomial Naive Bayes Classifier using ten folds cross validation.");
		System.out.println("Used Stemmed Words = "+stemming+" ; Removed Stop Words = "+noStopWords+" ; Binarize words frequency ="+binarize);
		System.out.println("*****************************************************************************************************");
		//Option 2 : Convert the directory containing neg and pos folders into a dataset
		TextDirectoryLoader loader = new TextDirectoryLoader();
		try{
			loader.setDirectory(new File(location));
			Instances dataRaw = loader.getDataSet();
			if (dataRaw.classIndex() == -1)
			{
				dataRaw.setClassIndex(dataRaw.numAttributes() - 1);
			}
			//Choose LovinsStemmer
			LovinsStemmer stemmer = new LovinsStemmer();
			//Choose N gram tokenizer
			NGramTokenizer nGramTokenizer = new NGramTokenizer();
			nGramTokenizer.setNGramMaxSize(1);
			nGramTokenizer.setNGramMinSize(1);
			//Or we can use  \"\r\n\t.,;:\\'\\"()?!\"'
			nGramTokenizer.setDelimiters("\\s+");
			// Apply NGramTokenizer and stop words removal to StringToWordVector
			StringToWordVector filter = new StringToWordVector();
			filter.setInputFormat(dataRaw);	
			if(!binarize)
			{
				filter.setOutputWordCounts(true);
			}
			if(noStopWords)
			{
				filter.setUseStoplist(true);
			}
			filter.setWordsToKeep(80000);
			filter.setTokenizer(nGramTokenizer);
			if(stemming)
			{
				filter.setStemmer(stemmer);
			}
			//filter.setAttributeIndices("first-last");//No need to set it, this first-last setting is by default
			Instances dataFiltered = Filter.useFilter(dataRaw, filter);
			//Binarize word frequencies
			if(binarize)
			{
				NumericToBinary numericToBinaryFilter = new NumericToBinary();
				dataFiltered = Filter.useFilter(dataRaw, filter);
				dataFiltered.deleteAttributeAt(dataFiltered.numAttributes()-1);
			}
			//System.out.println("Filtered data Summary is :"+dataFiltered.toSummaryString());
			NaiveBayesMultinomial multinomialNaiveBayesClassifier = new NaiveBayesMultinomial();	   
			/*
	    	multinomialNaiveBayesClassifier.buildClassifier(dataFiltered);
	    	System.out.println(multinomialNaiveBayesClassifier);
			 */
			//The classifier (in our example NaiveBayesMultinomial) should not be trained when handed over to the crossValidateModel method
			Evaluation eval = new Evaluation(dataFiltered);
			eval.crossValidateModel(multinomialNaiveBayesClassifier, dataFiltered, 10, new Random(1));
			System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		}
		catch(IOException ioException)
		{
			System.err.println("Can't read files from the location "+location);
		}
		catch(Exception exception)
		{
			exception.printStackTrace();
			System.err.println("Exception");
		}
	}	
	public static void main(String args[])
	{
		if(args!=null && args.length==1)
		{
			File trainingDir = new File(args[0]);
			if(!trainingDir.isDirectory())
			{
				System.err.println("Invalid training directory specified");
			}
			//Add all the positive and negative review files
			List<File> files = new ArrayList<File>();
			for(File dir : trainingDir.listFiles())
			{			
				if(dir!=null && dir.isDirectory())
				{
					for(File file:dir.listFiles())
					{
						files.add(file);				
					}
				}
			}
			//Arrange negative and positive reviews in splits for ten folds cross validation
			List<Split>foldsList = prepareDataForTenFoldsCrossValidation(files);
			//runSelfWrittenMultinomialNaiveBayes(foldsList,false,false,false);
			//runSelfWrittenMultinomialNaiveBayes(foldsList,true,false,false);
			//runSelfWrittenMultinomialNaiveBayes(foldsList,false,true,true);
			runSelfWrittenMultinomialNaiveBayes(foldsList,true,true,true);
			//runNaiveBayesClassificationUsingWeka(args[0],false,false,false);
			//runNaiveBayesClassificationUsingWeka(args[0],false,true,false);
			//runNaiveBayesClassificationUsingWeka(args[0],true,false,false);
			runNaiveBayesClassificationUsingWeka(args[0],true,true,false);
			//runNaiveBayesClassificationUsingWeka(args[0],false,false,true);
			//runNaiveBayesClassificationUsingWeka(args[0],false,true,true);
			//runNaiveBayesClassificationUsingWeka(args[0],true,false,true);
			runNaiveBayesClassificationUsingWeka(args[0],true,true,true);
			//runOpenNLPDocClassifierBasedOnMaxEnt(foldsList,false,false,false);
			//runOpenNLPDocClassifierBasedOnMaxEnt(foldsList,true,false,false);
			//runOpenNLPDocClassifierBasedOnMaxEnt(foldsList,false,true,true);
			runOpenNLPDocClassifierBasedOnMaxEnt(foldsList,true,true,true);
		}
		else
		{
			System.err.println("Please provide path of the folder containing positive and negative reviews folder.");
		}
	}

}
