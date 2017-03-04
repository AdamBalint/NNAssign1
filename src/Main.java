import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Scanner;

import Jama.Matrix;
import activations.Sigmoid;
import activations.Tanh;
import dataManager.Data;
import dataManager.Dataset;
import neuralNetwork.NeuralNet;
import variables.Variables;
import weightUpdate.Backprop;
import weightUpdate.Quickprop;
import weightUpdate.RProp;


public class Main {

	public Main(){		
		Variables.act = new Sigmoid();
		//Variables.act = new Tanh();
		Variables.learningRate = 0.5;
		Variables.momentumRate = 0.0;
		Variables.wUpdate = new Backprop(Variables.learningRate, Variables.momentumRate);
		//Variables.wUpdate = new RProp();
		Variables.wUpdate = new Quickprop();
		
		NeuralNet nn = new NeuralNet(64,10, new int[]{48});
		//NeuralNet nn = new NeuralNet(4,2, new int[]{15, 15});
		Dataset ds = new Dataset();
		loadData(ds);
		//loadTestingData(ds);
		//loadCancerData(ds);
		//loadIrisData(ds);
		long tStart = System.currentTimeMillis();
		nn.train(ds);
		ds.shuffleTesting();
		int correct = 0, total = 0;
		for (int i = 0; i < ds.getTestingSize(); i++){
			Data d = ds.nextTestingExample();
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
		long tEnd = System.currentTimeMillis();
		long tDelta = tEnd - tStart;
		double elapsedSeconds = tDelta / 1000.0;
		System.out.println("Time to Complete: " + elapsedSeconds);
		/*ds.setBias(false);
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
		System.out.println("Correct: " + correct + " out of " + total);*/
	}
	
	public void loadTestingData(Dataset ds){
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
	}
	
	public void loadCancerData(Dataset ds){
		try {
			Scanner in = new Scanner(new File("a1digits/cancer.txt"));
			while(in.hasNext()){
				double[] data = new double[9];
				double[] res = new double[2];
				String line = in.nextLine();
				String[] nums = line.split(",");
				for (int i = 1; i < nums.length-1; i++){
					data[i-1] = Double.parseDouble(nums[1]);
					if(i == 1){
						res[Integer.parseInt(nums[nums.length-1])/2-1] = 1;
					}
				}
				ds.addTrainingData(new Data(data, res));
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void loadIrisData(Dataset ds){
		try{
			Scanner in = new Scanner(new File("a1digits/iris.txt"));
			while(in.hasNext()){
				double[] data = new double[4];
				double[] res = new double[3];
				String line = in.nextLine();
				String[] nums = line.split(",");
				for (int i = 0; i < nums.length-1; i++){
					data[i] = Double.parseDouble(nums[i]);
					if (i == 0)
						res[Integer.parseInt(nums[nums.length-1])] = 1;
				}
				ds.addTrainingData(new Data(data, res));
			}
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
	public void loadData(Dataset ds){
		for (int i = 0; i < 10; i++){
			Scanner in;
			try {
				in = new Scanner(new File("a1digits/digit_test_" + i + ".txt"));
			
				double[] res = new double[10];
				res[i] = 1;
				while (in.hasNext()){
					double[] img = new double[64];
					String line = in.nextLine();
					String[] nums = line.split(",");
					for (int pix = 0; pix < nums.length; pix++){
						img[pix] = Double.parseDouble(nums[pix]);
					}
					ds.addTestingData(new Data(img, res));
				}
				in.close();
				
				in = new Scanner(new File("a1digits/digit_train_" + i + ".txt"));
				
				res = new double[10];
				res[i] = 1;
				while (in.hasNext()){
					double[] img = new double[64];
					String line = in.nextLine();
					String[] nums = line.split(",");
					for (int pix = 0; pix < nums.length; pix++){
						img[pix] = Double.parseDouble(nums[pix]);
					}
					ds.addTrainingData(new Data(img, res));
				}
				in.close();
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		new Main();
	}

}
