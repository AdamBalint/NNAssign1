package weightUpdate;

import Jama.Matrix;
import neuralNetwork.Layer;

public class Quickprop extends WeightUpdate{

	Matrix prevGrad;
	double µ = 1.75;
	
	
	@Override
	public Matrix getWeightUpdate(Matrix errGrad, Layer l) {
		if (prevGrad == null){
			prevGrad = new Matrix(errGrad.getRowDimension(), errGrad.getColumnDimension());
		}
		// TODO Auto-generated method stub
		//Matrix lastWeight = l.getLastUpdate();
		Matrix update = new Matrix(errGrad.getRowDimension(), errGrad.getColumnDimension());
		Matrix lastUpdate = l.getLastUpdate();
		if (lastUpdate == null){
			lastUpdate = new Matrix(errGrad.getRowDimension(), errGrad.getColumnDimension());
			for (int i = 0; i < lastUpdate.getRowDimension(); i++){
				for (int j = 0; j < lastUpdate.getColumnDimension(); j++){
					lastUpdate.set(i, j, 0.1);
				}
			}
		}
		//lastUpdate.print(5, 20);
		for (int i = 0; i < errGrad.getRowDimension(); i++){
			for(int j = 0; j < errGrad.getColumnDimension(); j++){
				if (prevGrad == null || errGrad.get(i, j) >= prevGrad.get(i, j)){
					update.set(i, j, lastUpdate.get(i, j)*µ);
				}
				else{
					//System.out.println(prevGrad.get(i, j)+"\t"+errGrad.get(i, j));
					double denom = prevGrad.get(i, j)-errGrad.get(i, j);
					denom = Math.max(Math.abs(denom), 0.001)*Math.signum(denom);
					update.set(i, j, (errGrad.get(i, j)/(denom))*lastUpdate.get(i, j));
				}
			}
		}
		
		//.arrayRightDivide(prevGrad.minus(errGrad)).arrayTimes(l.getLastUpdate());
		/*for (int i = 0; i < update.getRowDimension(); i++){
			for (int j = 0; j < update.getColumnDimension(); j++){
				update.set(i, j, update.get(i, j));
			}
		}*/
		prevGrad = errGrad.copy();
		for (int i = 0; i < update.getRowDimension(); i++){
			for(int j = 0; j < update.getColumnDimension(); j++){
				update.set(i, j, Math.min(Math.abs(update.get(i, j)), 3)*Math.signum(update.get(i, j)));
			}
		}
		return l.getWeights().arrayTimes(update);
		
		//return null;
	}


	@Override
	public WeightUpdate copy(Layer l) {
		// TODO Auto-generated method stub
		return new Quickprop();
	}

}
