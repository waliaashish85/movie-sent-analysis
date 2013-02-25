import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class Split {
	private List <File> trainingFilesList = new ArrayList<File>();
	private List <File> testingFilesList = new ArrayList<File>();
	private int splitNum = 0;
	public void setSplitNumber(int number)
	{
		splitNum = number;
	}
	public int getSplitNumber()
	{
		return splitNum;
	}
	public void addFilesToTrainingFilesList(File file)
	{
		if(file!=null)
		{
			trainingFilesList.add(file);
		}
	}
	public List<File> getTrainingFilesList()
	{
		return trainingFilesList;
	}
	public void addFilesToTestingFilesList(File file)
	{
		if(file!=null)
		{
			testingFilesList.add(file);
		}
	}
	public List<File> getTestingFilesList()
	{
		return testingFilesList;
	}

}
