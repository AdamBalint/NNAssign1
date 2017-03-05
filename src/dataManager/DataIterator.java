package dataManager;

public abstract class DataIterator {

	public abstract Data nextTrainingExample();
	public abstract Data nextTestingExample();
	public abstract boolean hasNextTrainingExample();
	public abstract boolean hasNextTestingExample();
	public abstract int getTestingSize();
	public abstract int getTrainingSize();
}
