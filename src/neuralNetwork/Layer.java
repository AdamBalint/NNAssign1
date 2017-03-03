package neuralNetwork;
import Jama.Matrix;
import activations.ActivationFunction;
import weightUpdate.Quickprop;
import weightUpdate.RProp;
import weightUpdate.WeightUpdate;
import variables.Variables;
public class Layer {
	Matrix input;
	Matrix weights; // connecting previous layer layer
	Matrix output;
	Matrix der;
	Matrix value;
	Matrix err;
	Matrix lastErrSign;
	Matrix lastChange;
	ActivationFunction act;
	WeightUpdate weightUpdate;
	Matrix gradSum;
	
	
	//double momentumRate = 0.005;
	public Layer(int in, int out, ActivationFunction act, boolean output){
		// add 1 for the bias
		if (!output)
			weights = new Matrix(out, in);
		else
			weights = new Matrix(out, in);
		weightUpdate = Variables.wUpdate.copy(this);
		//lastChange = new Matrix(weights.getRowDimension(), weights.getColumnDimension());
		//weightUpdate = new RProp(weights.getRowDimension(), weights.getColumnDimension());
		//weightUpdate = new Quickprop();//new Quickprop(weights.getRowDimension(), weights.getColumnDimension());
		//weightUpdate = new Backprop(0.05);
		this.act = act;
		this.err = new Matrix(out, 1);
		gradSum = new Matrix(out, in);
	}
	
	public Matrix forwardPass(Matrix in){
		input = in.copy();
		in = weights.times(in);
		der = act.derivativeEval(in);
		//weights.print(5, 2);
		//in.print(5, 2);
		
		output = act.eval(in);
		// add the bias
		
		
		return output;
	}

	//public Matrix backprop(){
		
		
	//	return null;
	//}
	
	public void sumGradient(){
		Matrix errAtNode = err.arrayTimes(der).times(input.transpose()).times(-1);
		gradSum.plusEquals(errAtNode);
	}
	
	public void weightUpdate(){
		// -eta*(self.layers[i+1].D.dot(self.layers[i].Z)).T 
		// self.layers[i].W += W_grad
		
		//err.arrayTimes(der);
		
		
		Matrix change = weightUpdate.getWeightUpdate(gradSum, this);
		
		//errAtNode = errAtNode.times(input.transpose());
		//Matrix change = errAtNode.times(0.05);//(errAtNode.arrayTimes(bigInp)).times(0.2);
		
		//change.print(4, 2);
		weights = weights.plus(change);//.plus(lastChange.times(momentumRate));
		lastChange = change.copy();
		err = new Matrix(err.getRowDimension(), err.getColumnDimension());
		gradSum = new Matrix(gradSum.getRowDimension(), gradSum.getColumnDimension());
	}
	
	public void initData() {
		// TODO Auto-generated method stub
		for (int i = 0; i < weights.getRowDimension(); i++){
			for (int j = 0; j < weights.getColumnDimension(); j++){
				weights.set(i, j, Math.random()-0.5);
			}
		}
	}

	public Matrix getInput() {
		// TODO Auto-generated method stub
		return input;
	}
	public Matrix getError(){
		return err;
	}
	public Matrix getLastUpdate(){
		return lastChange;
	}
	public Matrix getWeights(){
		return weights.copy();
	}
}
