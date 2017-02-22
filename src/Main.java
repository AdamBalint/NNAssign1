import java.util.Arrays;

import Jama.Matrix;

public class Main {

	public Main(){
		NeuralNet nn = new NeuralNet(4,2, new int[]{5,4});
		Dataset ds = new Dataset();
		ds.setBias(false);
		ds.addTrainingData(new Data(new double[]{0, 0, 0, 0}, new double[]{0, 1})); // 0
		ds.addTrainingData(new Data(new double[]{1, 1, 0, 0}, new double[]{0, 1})); // 12
		ds.addTrainingData(new Data(new double[]{0, 1, 1, 0}, new double[]{0, 1})); // 6
		ds.addTrainingData(new Data(new double[]{0, 0, 1, 1}, new double[]{0, 1})); // 3
		ds.addTrainingData(new Data(new double[]{1, 0, 1, 0}, new double[]{0, 1})); // 10
		ds.addTrainingData(new Data(new double[]{0, 1, 0, 1}, new double[]{0, 1})); // 5
		ds.addTrainingData(new Data(new double[]{1, 1, 1, 1}, new double[]{0, 1})); // 15
		ds.addTrainingData(new Data(new double[]{1, 0, 0, 1}, new double[]{0, 1})); // 9
		
		ds.addTrainingData(new Data(new double[]{1, 0, 0, 0}, new double[]{1, 0})); // 8
		ds.addTrainingData(new Data(new double[]{0, 1, 0, 0}, new double[]{1, 0})); // 4
		ds.addTrainingData(new Data(new double[]{0, 0, 1, 0}, new double[]{1, 0})); // 2
		ds.addTrainingData(new Data(new double[]{0, 0, 0, 1}, new double[]{1, 0})); // 1
		ds.addTrainingData(new Data(new double[]{1, 1, 1, 0}, new double[]{1, 0})); // 14
		ds.addTrainingData(new Data(new double[]{1, 0, 1, 1}, new double[]{1, 0})); // 11
		ds.addTrainingData(new Data(new double[]{0, 1, 1, 1}, new double[]{1, 0})); // 7
		ds.addTrainingData(new Data(new double[]{1, 1, 0, 1}, new double[]{1, 0})); // 13
		
		nn.train(ds);
		//Data d = new Data(new double[]{1, 1, 0, 1}, new double[]{1, 0});
		int correct = 0, total = 0;
		for (int i = 0; i < 16; i++){
			Data d = ds.nextTrainingExample();
			Matrix ret = nn.evaluate(d);
			System.out.println(Arrays.toString(ret.getRowPackedCopy()) + ", " + Arrays.toString(d.getOutputVector().getRowPackedCopy()));
			for (int j = 0; j < ret.getRowDimension(); j++){
				for (int k = 0; k < ret.getColumnDimension(); k++){
					if (ret.get(j, k) == 1 && d.getOutputVector().get(j, k) == 1)
						correct += 1;
				}
			}
			total += 1;
		}
		System.out.println("Correct: " + correct + " out of " + total);
	}
	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		new Main();
	}

}
