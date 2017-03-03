package dataManager;
import java.util.Arrays;

import Jama.Matrix;

// Based on column vectors
public class Data {

	private Matrix input, output;
	
	public Data(double[] inputVector, double[] outputVector){
		input = new Matrix(inputVector,1).transpose();
		output = new Matrix(outputVector,1).transpose();
	}
	
	public Matrix getInputVector(){
		return input;
	}
	
	public Matrix getOutputVector(){
		return output;
	}
	
	public void addBias(){
		double[] old = input.transpose().getRowPackedCopy();
		double[] n = Arrays.copyOf(old, old.length+1);
		n[n.length-1] = 1;
		input = new Matrix(n, 1).transpose();
	}
	
}
