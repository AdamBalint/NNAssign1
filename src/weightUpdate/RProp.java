package weightUpdate;

import Jama.Matrix;
import neuralNetwork.Layer;

public class RProp extends WeightUpdate{

	double etaPlus = 1.2, etaMinus = 0.5;
	double deltaMax = 25, deltaMin = 1e-6;
	Matrix lastGrad = null;
	public Matrix changeMat;

	public RProp(){}

	public RProp(int rowDim, int colDim){
		changeMat = new Matrix(rowDim, colDim);
		for (int i = 0; i < changeMat.getRowDimension(); i++){
			for (int j = 0; j < changeMat.getColumnDimension(); j++){
				changeMat.set(i, j, 0.1);
			}
		}
		lastGrad = new Matrix(rowDim, colDim);
	}

	@Override
	public Matrix getWeightUpdate(Matrix grad, Layer l) {
		// TODO Auto-generated method stub
		/*if (lastGrad == null){
			lastGrad = grad.copy();
			return new Matrix(changeMat.getRowDimension(), changeMat.getColumnDimension());
		}*/
		Matrix lastUpdate = l.getLastUpdate();
		Matrix weightUpdate = new Matrix(changeMat.getRowDimension(), changeMat.getColumnDimension());
		for (int j = 0; j < grad.getColumnDimension(); j++){
			//double sumGradient = sumGradient(grad, j);
			for (int i = 0; i < grad.getRowDimension(); i++){
			
			

				// positive
				/*if (lastGrad.get(i, j) * grad.get(i, j) > 0){

					changeMat.set(i, j, Math.min(changeMat.get(i, j)*etaPlus, deltaMax));
					weightUpdate.set(i, j, changeMat.get(i, j) * -Math.signum(grad.get(i, j)));

					//if (grad.get(i, j) != 0)
					//lastGrad = grad.copy();
					//changeMat.set(i, j, Math.max(changeMat.get(i, j)*etaMinus, deltaMin));
					//grad.set(i, j, 0);
					lastGrad.set(i, j, -Math.signum(grad.get(i, j)));
					//negative
				}else if (lastGrad.get(i, j) * grad.get(i, j) < 0){

					changeMat.set(i, j, Math.max(changeMat.get(i, j)*etaMinus, deltaMin));
					lastGrad.set(i, j, 0);
					//Matrix lastWeight = l.getLastUpdate();
					//lastGrad.set(i, j, lastGrad.get(i, j));
					weightUpdate.set(i, j, 0);
					//weightUpdate.set(i, j, -lastWeight.get(i, j));
					//weightUpdate.set(i, j, changeMat.get(i, j) * -Math.signum(grad.get(i, j)));
					//if (grad.get(i, j) != 0)
					//	weightUpdate.set(i, j, changeMat.get(i, j) * -Math.signum(grad.get(i, j)));
					//changeMat.set(i, j, Math.min(changeMat.get(i, j)*etaPlus, deltaMax));
					//weightUpdate.set(i, j, changeMat.get(i, j) * Math.signum(grad.get(i, j)));
					//lastGrad = grad.copy();
					//lastGrad.set(i, j, grad.get(i, j));
				}else{
					//changeMat.set(i, j, changeMat.get(i, j));
					//weightUpdate.set(i, j, 0);
					weightUpdate.set(i, j, changeMat.get(i, j) * -Math.signum(grad.get(i, j)));
					//lastGrad = grad.copy();
					lastGrad.set(i, j, -Math.signum(grad.get(i, j)));
				}*/
				if (lastGrad.get(i, j) * grad.get(i, j) > 0){
					changeMat.set(i, j, Math.min(changeMat.get(i, j)*etaPlus, deltaMax));
					weightUpdate.set(i, j, -Math.signum(grad.get(i, j))*changeMat.get(i, j));
					lastGrad.set(i, j, grad.get(i, j));
				}
				else if (lastGrad.get(i, j) * grad.get(i, j) < 0){
					changeMat.set(i, j, Math.max(changeMat.get(i, j)*etaMinus, deltaMin));
					//grad.set(i, j, 0);
					
					
					// with and without backtracking
					//weightUpdate.set(i, j, -lastUpdate.get(i, j));
					weightUpdate.set(i, j, 0);
					
					
					
					
					
					//weightUpdate.set(i, j, -Math.signum(grad.get(i, j)*changeMat.get(i, j)));
					lastGrad.set(i, j, 0);
				}
				else{
					weightUpdate.set(i, j, -Math.signum(grad.get(i, j))*changeMat.get(i, j));
					lastGrad.set(i, j, grad.get(i, j));
				}

				/*boolean skipUpdate = false;
				if (lastGrad.get(i, j) * grad.get(i, j) > 0){
					changeMat.set(i, j, Math.min(changeMat.get(i, j)*etaPlus, deltaMax));
				}
				else if (lastGrad.get(i, j) * grad.get(i, j) < 0){
					changeMat.set(i, j, Math.max(changeMat.get(i, j)*etaMinus, deltaMin));

					skipUpdate = true;
				}

				if(!skipUpdate){
					if (grad.get(i, j) > 0){
						weightUpdate.set(i, j, -changeMat.get(i, j));
					}
					else if (grad.get(i, j) < 0){
						weightUpdate.set(i, j, changeMat.get(i, j));
					}else{
						weightUpdate.set(i, j, 0);
					}
					lastGrad.set(i, j, grad.get(i, j));
				}*/
				boolean skipUpdate = false;
				double delta = 0;
				//double gradient = sumGradient;
				/*if (lastGrad.get(i, j) * sumGradient > 0){
					delta = changeMat.get(i, j) * Math.signum(grad.get(i, j));
					changeMat.set(i, j, Math.min(changeMat.get(i, j) * etaPlus, deltaMax));
					lastGrad.set(i, j, sumGradient);
				}
				else if (lastGrad.get(i, j) * sumGradient < 0){
					changeMat.set(i, j, Math.max(changeMat.get(i, j) * etaMinus, deltaMin));
					//sumGradient = 0;
					lastGrad.set(i, j, 0);
					skipUpdate = true;
					weightUpdate.set(i, j, lastUpdate.get(i, j));
				}
				else{
					delta = changeMat.get(i, j) * Math.signum(grad.get(i, j));
					lastGrad.set(i, j, sumGradient);
				}

				//if (!skipUpdate){
					weightUpdate.set(i, j, -delta);
					
				//}*/

			}
			//lastGrad.set(i, j, sumGradient);
		}
		//lastGrad = grad.copy();

		//changeMat.print(5, 3);

		return weightUpdate;//.times(-1);

	}


	@Override
	public WeightUpdate copy(Layer l) {
		// TODO Auto-generated method stub
		Matrix weights = l.getWeights();
		return new RProp(weights.getRowDimension(), weights.getColumnDimension());
	}



}
