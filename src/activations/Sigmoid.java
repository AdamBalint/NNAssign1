package activations;

import Jama.Matrix;

public class Sigmoid extends ActivationFunction{
	
	@Override
	public double eval(double in) {
		// TODO Auto-generated method stub
		return 1/(1+Math.exp(-in));
	}
	
	@Override
	public double derivativeEval(double in) {
		// TODO Auto-generated method stub
		double val = eval(in);
		return val*(1-val);
	}
	
	@Override
	public Matrix eval(Matrix in) {
		// TODO Auto-generated method stub
		//return 1/(1+Math.exp(-in));
		Matrix inc = in.copy();
		for (int i = 0; i < in.getRowDimension(); i++){
			for (int j = 0; j < in.getColumnDimension(); j++){
				inc.set(i, j, eval(inc.get(i, j)));
			}
		}
		return inc;
	}
	
	@Override
	public Matrix derivativeEval(Matrix in) {
		// TODO Auto-generated method stub
		Matrix inc = in.copy();
		for (int i = 0; i < in.getRowDimension(); i++){
			for (int j = 0; j < in.getColumnDimension(); j++){
				inc.set(i, j, derivativeEval(inc.get(i, j)));
			}
		}
		return inc.copy();
		//double val = eval(in);
		//return val*(1-val);
	}
	
	
}
