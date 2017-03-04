package dataManager;
import java.util.ArrayList;
import java.util.Collections;

import variables.Variables;

public class Dataset {

	private ArrayList<Data> training;
	private ArrayList<Data> testing;
	private int trainingPointer = 0, testingPointer = 0;
	private boolean bias = false;
	private boolean kfold = false;
	private int currentK;
	private double step;
	private DataIterator di;
	public Dataset(){
		kfold = Variables.dataSplitMethod == 0 ? false : true;
		if (kfold){
			di = new KFoldIterator();
			currentK = 0;
			training = new ArrayList<Data>();
		}else{
			di = new HoldoutIterator();
			training = new ArrayList<Data>();
			testing = new ArrayList<Data>();
		}
	}
	
	public void addTrainingData(Data example){
		if (bias)
			example.addBias();
		training.add(example);
		//testing.add(example);
	}
	
	public void addTestingData(Data example){
		// double check this is correct
		if (kfold){
			training.add(example);
			step = training.size()/(double)Variables.kValue;
		}
		else{
			testing.add(example);
		}
	}
	
	public Data nextTrainingExample(){
		if (kfold && trainingPointer > step*(currentK) && trainingPointer < (currentK+1) * step){
			trainingPointer = (int)Math.ceil((currentK+1)*step);
		}
		if (trainingPointer >= training.size() || currentK == Variables.kValue-1){
			trainingPointer %= training.size();
			
			shuffleTraining();
		}
		
		return training.get(trainingPointer++);
	}
	
	public Data nextTestingExample(){
		//if (kfold){
			
			
		//}else{
		testingPointer %= testing.size();
		return testing.get(testingPointer++);
		//}
	}
	
	public void shuffleTraining(){
		Collections.shuffle(training, Variables.r);
	}
	
	public void shuffleTesting(){
		Collections.shuffle(testing, Variables.r);
	}

	public void setBias(boolean b) {
		// TODO Auto-generated method stub
		bias = b;
	}
	
	public int getTrainingSize(){
		return training.size();
	}
	
	public int getTestingSize(){
		return testing.size();
	}
	
	
}
