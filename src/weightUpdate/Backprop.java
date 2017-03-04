package weightUpdate;

import Jama.Matrix;
import neuralNetwork.Layer;

public class Backprop extends WeightUpdate{

	private double lr;
	private double mr;
	@Override
	public Matrix getWeightUpdate(Matrix grad, Layer l) {
		// TODO Auto-generated method stub
		Matrix last = l.getLastUpdate();
		if (last == null){
			last = new Matrix(grad.getRowDimension(), grad.getColumnDimension());
		}
		return (grad.times(-lr)).plus(last.times(mr));
		
		//Matrix last = l.getLastUpdate();
		//if (last == null)
		//	return grad.times(lr);
		/*
		return (grad.times(-lr)).plus(l.getLastUpdate().times(mr));//l.getError().arrayTimes(grad).times(l.getInput().transpose()).times(lr);
		*///return ret;
	}
	
	public Backprop(double learningRate, double momentumRate){
		lr = learningRate;
		mr = momentumRate;
	}

	@Override
	public WeightUpdate copy(Layer l) {
		// TODO Auto-generated method stub
		return new Backprop(lr, mr);
	}
	
	
	
	
}
