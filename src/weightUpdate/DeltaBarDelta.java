package weightUpdate;

import Jama.Matrix;
import neuralNetwork.Layer;

public class DeltaBarDelta extends WeightUpdate{

	Matrix lastGrad; // last gradient
	Matrix learningMat; // learning rate
	double beta = 0.5;
	
	
	public DeltaBarDelta(){}

	public DeltaBarDelta(int rowDim, int colDim){
		learningMat = new Matrix(rowDim, colDim);
		for (int i = 0; i < learningMat.getRowDimension(); i++){
			for (int j = 0; j < learningMat.getColumnDimension(); j++){
				learningMat.set(i, j, 0.1);
			}
		}
		lastGrad = new Matrix(rowDim, colDim);
	}
	
	@Override
	public Matrix getWeightUpdate(Matrix errGrad, Layer l) {
		// TODO Auto-generated method stub
		Matrix g = (errGrad.times(1-beta)).plus(lastGrad.times(beta));
		Matrix lastUpdate = l.getLastUpdate();
		if (lastUpdate == null){
			lastUpdate = new Matrix(g.getRowDimension(), g.getColumnDimension());
		}
		for (int i = 0; i < g.getRowDimension(); i++){
			for (int j = 0; j < g.getColumnDimension(); j++){
				double product = g.get(i, j)*errGrad.get(i, j);
				if (product > 0){
					learningMat.set(i, j, learningMat.get(i, j)+0.5);
				}else if (product < 0){
					learningMat.set(i, j, learningMat.get(i, j)*0.5);
				}
			}
		}
		lastGrad = errGrad.copy();
		
		
		return errGrad.arrayTimes(learningMat.times(-1));//.plus(lastUpdate.times(0.3));
	}

	@Override
	public WeightUpdate copy(Layer l) {
		// TODO Auto-generated method stub
		Matrix weights = l.getWeights();
		return new DeltaBarDelta(weights.getRowDimension(), weights.getColumnDimension());
	}

}
