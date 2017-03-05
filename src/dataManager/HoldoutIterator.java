package dataManager;

public class HoldoutIterator extends DataIterator{

	private Dataset ds;
	private int trainCount = 0;
	private int testCount = 0;
	
	public HoldoutIterator(Dataset ds){
		this.ds = ds;
	}
	
	@Override
	public Data nextTrainingExample() {
		// TODO Auto-generated method stub
		//trainCount = trainCount % ds.getTrainingSize();
		Data d = ds.training.get(trainCount);
		trainCount++;
		return d;
	}

	@Override
	public Data nextTestingExample() {
		// TODO Auto-generated method stub
		Data d = ds.testing.get(testCount);
		testCount++;
		return d;
	}

	@Override
	public boolean hasNextTrainingExample() {
		// TODO Auto-generated method stub
		boolean ret = trainCount < ds.getTrainingSize();
		if (!ret)
			trainCount %= ds.getTrainingSize();
		return ret;
	}

	@Override
	public boolean hasNextTestingExample() {
		// TODO Auto-generated method stub
		boolean ret = testCount < ds.getTestingSize();
		if (!ret)
			testCount %= ds.getTestingSize();
		return ret;
	}

	@Override
	public int getTestingSize() {
		// TODO Auto-generated method stub
		return ds.testing.size();
	}

	@Override
	public int getTrainingSize() {
		// TODO Auto-generated method stub
		return ds.training.size();
	}

}
