import Jama.Matrix;
import activations.ActivationFunction;

public class Layer {
	Matrix input;
	Matrix weights; // connecting previous layer layer
	Matrix output;
	Matrix der;
	Matrix value;
	Matrix err;
	Matrix lastChange;
	ActivationFunction act;
	public Layer(int in, int out, ActivationFunction act, boolean output){
		// add 1 for the bias
		if (!output)
			weights = new Matrix(out, in);
		else
			weights = new Matrix(out, in);
		lastChange = new Matrix(weights.getRowDimension(), weights.getColumnDimension());
		this.act = act;
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
	
	public void weightUpdate(){
		// -eta*(self.layers[i+1].D.dot(self.layers[i].Z)).T 
		// self.layers[i].W += W_grad
		
		Matrix errAtNode = err.arrayTimes(der);//err.arrayTimes(der);
		/*Matrix errAtNode = err.times(der.transpose());
		Matrix bigInp = new Matrix(weights.getRowDimension(), weights.getColumnDimension());
		for (int i = 0; i < bigInp.getRowDimension(); i++){
			bigInp.setMatrix(i,i, 0, bigInp.getColumnDimension()-1, input.transpose());
		}
		*/
		errAtNode = errAtNode.times(input.transpose());
		Matrix change = errAtNode.times(0.2);//(errAtNode.arrayTimes(bigInp)).times(0.2);
		
		//change.print(4, 2);
		weights = weights.plus(change).plus(lastChange.times(0.5));
	}
	
	public void initData() {
		// TODO Auto-generated method stub
		for (int i = 0; i < weights.getRowDimension(); i++){
			for (int j = 0; j < weights.getColumnDimension(); j++){
				weights.set(i, j, Math.random()-0.5);
			}
		}
	}
}
