import Jama.Matrix;
import activations.ActivationFunction;
import activations.Sigmoid;

public class NeuralNet {
	
	Layer[] layers; 
	
	ActivationFunction act = new Sigmoid();
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
	
	public void train(Dataset ds){
		for (int epoch = 0; epoch < 50000; epoch++){
			ds.shuffleTraining();
			double mse = 0;
			for (int i = 0; i < 16; i++){
				Data d = ds.nextTrainingExample();
				//layers[0].input = d.getInputVector();
				Matrix m = layers[0].forwardPass(d.getInputVector());
				for (int l = 1; l < layers.length; l++){
					m = layers[l].forwardPass(m);
				}
				//m.print(5, 2);
				Matrix res = d.getOutputVector();
				
				Matrix err = res.minus(m);
				for (int r = 0; r < res.getRowDimension(); r++){
					for (int c = 0; c < res.getColumnDimension(); c++){
						mse += Math.pow(err.get(r, c),2);
					}
				}
				
				
				backprop(err);
				weightUpdate();
			}
			System.err.println(mse);
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
		for (int l = layers.length-2; l >= 0; l--){
			layers[l].err = layers[l+1].weights.transpose().times(layers[l+1].err).arrayTimes(act.derivativeEval(layers[l].output));
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
		for (int i = 0; i < m.getRowDimension(); i++){
			for (int j = 0; j < m.getColumnDimension(); j++){
				ret.set(i, j, m.get(i, j) < maxVal ? 0 : 1);
			}
		}
		
		
		return ret;
	}
	
}
