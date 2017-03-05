package dataManager;

import variables.Variables;

public class KFoldIterator extends DataIterator{

	private Dataset ds;
	private int trainCount = 0;
	private int testCount = 0;
	private int currK = 0;
	
	
	public KFoldIterator(Dataset ds){
		this.ds = ds;
	}
	
	
	@Override
	public Data nextTrainingExample() {
		// TODO Auto-generated method stub
		Data d = ds.training.get(trainCount);
		trainCount++;
		return d;
	}

	@Override
	public Data nextTestingExample() {
		// TODO Auto-generated method stub
		Data d = ds.training.get(testCount);
		testCount++;
		return d;
	}


	@Override
	public boolean hasNextTrainingExample() {
		// TODO Auto-generated method stub
		
		
		// If we hit the k value that is being left out
		if (trainCount >= currK*(ds.getTrainingSize()/Variables.kValue) && trainCount < (currK+1)*(ds.getTrainingSize()/Variables.kValue)){
			// set the counter to jump over the gap
			trainCount = (currK+1)*(ds.getTrainingSize()/Variables.kValue);
		}
		
		// if we reached the end of the training data, so increase k value
		if (trainCount>ds.getTrainingSize()-1){
			currK++;
			// if we reach the end of the k values, then we are done the epoch
			if(currK == Variables.kValue-1){
				// reset all counters
				currK = 0;
				trainCount = 0;
				return false;
			}
			trainCount = 0;	
		}
		
		return true;
	}


	@Override
	public boolean hasNextTestingExample() {
		// TODO Auto-generated method stub
		if (testCount >= currK*(ds.getTrainingSize()/Variables.kValue) && testCount < (currK+1)*(ds.getTrainingSize()/Variables.kValue)){
			//testCount = currK*(ds.getTrainingSize()/Variables.kValue);
			return true;
		}
		testCount = currK*(ds.getTrainingSize()/Variables.kValue);
		return false;
	}


	@Override
	public int getTestingSize() {
		// TODO Auto-generated method stub
		
		return (currK+1)*(ds.training.size()/Variables.kValue) - currK*(ds.training.size()/Variables.kValue);
	}


	@Override
	public int getTrainingSize() {
		// TODO Auto-generated method stub
		return ds.training.size() - ((currK+1)*(ds.training.size()/Variables.kValue) - currK*(ds.training.size()/Variables.kValue));
	}

}
