package neuralNetwork;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import Jama.Matrix;
import activations.ActivationFunction;
import activations.ReLU;
import activations.Sigmoid;
import dataManager.Data;
import dataManager.Dataset;
import variables.Variables;
import weightUpdate.Backprop;
import weightUpdate.WeightUpdate;

public class NeuralNet {
	
	Layer[] layers; 
	private PrintWriter fw;
	ActivationFunction act = new Sigmoid();
	//ActivationFunction act = new ReLU();
	public NeuralNet(int input, int output, int[] hidden){
		layers = new Layer[hidden.length+1];
		
		for (int i = 0; i < layers.length; i++){
			if (i == 0 && hidden.length == 0)
				layers[i] = new Layer(input, output, act, true);
			else if (i == 0)
				layers[i] = new Layer(input, hidden[0], act, false);
			else if (i == (hidden.length))
				layers[i] = new Layer(hidden[hidden.length-1], output, act, true);
			else
				layers[i] = new Layer(hidden[i-1], hidden[i], act, false);
			layers[i].initData();
		}
	}
	
	public void setLogWriter(PrintWriter fw){
		this.fw = fw;
	}
	
	public void train(Dataset ds) throws IOException{
		
		int data = ds.getTrainingSize();
		// 200 pretty good for backprop
		for (int epoch = 0; epoch < 50; epoch++){
			ds.shuffleTraining();
			
			double testMSE = 0;
			while(ds.hasNextTestingExample()){
				Data d = ds.nextTestingExample();
				//layers[0].input = d.getInputVector();
				Matrix m = layers[0].forwardPass(d.getInputVector());
				for (int l = 1; l < layers.length; l++){
					m = layers[l].forwardPass(m);
				}
				
				//m.print(5, 2);
				Matrix res = d.getOutputVector();
				
				Matrix err = m.minus(res);//res.minus(m);
				for (int r = 0; r < res.getRowDimension(); r++){
					for (int c = 0; c < res.getColumnDimension(); c++){
						testMSE += Math.pow(err.get(r, c),2);
					}
				}
			}
			
			double mse = 0;
			while (ds.hasNextTrainingExample()){
				Data d = ds.nextTrainingExample();
				//layers[0].input = d.getInputVector();
				Matrix m = layers[0].forwardPass(d.getInputVector());
				for (int l = 1; l < layers.length; l++){
					m = layers[l].forwardPass(m);
				}
				
				//m.print(5, 2);
				Matrix res = d.getOutputVector();
				
				Matrix err = m.minus(res);//res.minus(m);
				for (int r = 0; r < res.getRowDimension(); r++){
					for (int c = 0; c < res.getColumnDimension(); c++){
						mse += Math.pow(err.get(r, c),2);
					}
				}
				
				
				backprop(err);
				/*if ((i+1) % (int)(data*1.1) == 0){
					weightUpdate();
				}*/
				if (Variables.wUpdate.getClass() == Backprop.class)
					weightUpdate();
			}
			if (Variables.wUpdate.getClass() != Backprop.class)
				weightUpdate();
			
			// Testing mse calculation - acts like validation set
			
			
			
			fw.println(mse/data + "\t" + testMSE/ds.getTestingSize());
			fw.flush();
			System.err.println(mse/data);
		}
	}
	
	private void weightUpdate() {
		// TODO Auto-generated method stub
		for(int l = 0; l < layers.length; l++){
			layers[l].weightUpdate();
		}
	}

	public void backprop(Matrix err){
		layers[layers.length-1].err = err;
		layers[layers.length-1].sumGradient();
		for (int l = layers.length-2; l >= 0; l--){
			layers[l].err = layers[l+1].weights.transpose().times(layers[l+1].err).
					arrayTimes(act.derivativeEval(layers[l].output));
			layers[l].sumGradient();
		}
	}
	
	public Matrix evaluate(Data d){
		
		Matrix m = layers[0].forwardPass(d.getInputVector());
		for (int l = 1; l < layers.length; l++){
			m = layers[l].forwardPass(m);
		}
		
		double maxVal = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < m.getRowDimension(); i++){
			for (int j = 0; j < m.getColumnDimension(); j++){
				maxVal = Math.max(maxVal, m.get(i, j));
			}
		}
		
		Matrix ret = new Matrix(m.getRowDimension(), m.getColumnDimension());
		boolean maxSet = false;
		for (int i = 0; i < m.getRowDimension(); i++){
			for (int j = 0; j < m.getColumnDimension(); j++){
				if (m.get(i, j) < maxVal || maxSet){
					ret.set(i, j, 0);
				}else{
					ret.set(i, j, 1);
					maxSet = true;
				}
			}
		}
		
		
		return ret;
	}
	
}
