import java.util.ArrayList;
import java.util.Collections;

public class Dataset {

	private ArrayList<Data> training;
	private ArrayList<Data> testing;
	private int trainingPointer = 0, testingPointer = 0;
	private boolean bias = false;
	
	public Dataset(){
		training = new ArrayList<Data>();
		testing = new ArrayList<Data>();
	}
	
	public void addTrainingData(Data example){
		if (bias)
			example.addBias();
		training.add(example);
	}
	
	public void addTestingData(Data example){
		testing.add(example);
	}
	
	public Data nextTrainingExample(){
		//if (trainingPointer > training.size()){
			trainingPointer %= training.size();
		//	shuffleTraining();
		//}
		
		return training.get(trainingPointer++);
	}
	
	public Data nextTestingExample(){
		testingPointer %= testing.size();
		return testing.get(testingPointer++);
	}
	
	public void shuffleTraining(){
		Collections.shuffle(training);
	}

	public void setBias(boolean b) {
		// TODO Auto-generated method stub
		bias = b;
	}
	
	
}
