import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
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
import weightUpdate.DeltaBarDelta;
import weightUpdate.Quickprop;
import weightUpdate.RProp;


public class Main {
	public PrintWriter fw = null;
	public Main(){	
		String name = genName();
		//Variables.act = new Sigmoid();
		//Variables.act = new Tanh();
		//Variables.learningRate = 0.5;
		//Variables.momentumRate = 0.0;
		//Variables.wUpdate = new Backprop(Variables.learningRate, Variables.momentumRate);
		//Variables.wUpdate = new RProp();
		//Variables.wUpdate = new Quickprop();
		//Variables.wUpdate = new DeltaBarDelta();
		try {
			PrintWriter corOut = new PrintWriter("CorrectNum/"+name+"-CorrectClass.txt");
			// loop here for runs for experiment
			for(int run = 0; run < 20; run++){
				
					fw = new PrintWriter("Data/"+name+"-"+run+".txt");
				
				
				NeuralNet nn = new NeuralNet(64,10, new int[]{Variables.hiddenNodeNum});
				nn.setLogWriter(fw);
				//NeuralNet nn = new NeuralNet(4,2, new int[]{15, 15});
				Dataset ds = new Dataset();
				loadData(ds);
				System.out.println(ds.getTrainingSize());
				//loadTestingData(ds);
				//loadCancerData(ds);
				//loadIrisData(ds);
				long tStart = System.currentTimeMillis();
				try{
				nn.train(ds);
				}catch(IOException e){
					System.out.println("Failed to train network");
					e.printStackTrace();
					System.exit(1);
				}
				ds.shuffleTesting();
				int correct = 0, total = 0;
				while(ds.hasNextTestingExample()){
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
				corOut.println(correct + "\t" + (((double)correct)/ds.getTestingSize()));
				corOut.flush();
				System.out.println("Correct: " + correct + " out of " + total);
				long tEnd = System.currentTimeMillis();
				long tDelta = tEnd - tStart;
				double elapsedSeconds = tDelta / 1000.0;
				System.out.println("Time to Complete: " + elapsedSeconds);
				fw.close();
			}
			corOut.close();
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			System.err.println("File not found");
			e1.printStackTrace();
			System.exit(1);
		}
		try {
			FileWriter out = new FileWriter("CompleteLog.txt");
			out.append(name+"\n");
			out.flush();
			out.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
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
	
	private String genName() {
		// TODO Auto-generated method stub
		String s = "";
		// learning_type-act-(hold/k#)-(learning_type_params)
		if(Variables.wUpdate.getClass() == Backprop.class)
			s += "bp-";
		else
			s += "rp-";
		if(Variables.act.getClass() == Sigmoid.class)
			s += "sig-";
		else
			s += "tanh";
		if (Variables.dataSplitMethod == 0)
			s += "hold-lr"+Variables.learningRate +"-mr" + Variables.momentumRate + "-";
		else
			s += "k" + Variables.kValue + "-";
		s += "h"+Variables.hiddenNodeNum;
		
		return s;
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
		System.out.println(args.length);
		System.out.println(Arrays.toString(args));
		if(args.length >= 2){
			// bp hidden_node_num activation_function data_split lr mr (k_value)
			if(args[0].equalsIgnoreCase("bp")){
				if(args.length >= 6){
					Variables.hiddenNodeNum = Integer.parseInt(args[1]);
					Variables.learningRate = Double.parseDouble(args[4]);
					Variables.momentumRate = Double.parseDouble(args[5]);
					Variables.wUpdate = new Backprop(Variables.learningRate, Variables.momentumRate);
					Variables.act = args[2].equalsIgnoreCase("0") ? new Sigmoid() : new Tanh();
					Variables.dataSplitMethod = Integer.parseInt(args[3]);
					if(Variables.dataSplitMethod == 1){
						Variables.kValue = Integer.parseInt(args[6]);
					}
				}
				else{
					System.err.println("Not enough parameters for backprop");
					System.exit(1);
				}
			}
			else if (args[0].equalsIgnoreCase("rprop")){
				// rprop hidden_node_num activation_function data_split (k_value)
				if(args.length >= 4){
					Variables.wUpdate = new RProp();
					Variables.hiddenNodeNum = Integer.parseInt(args[1]);
					Variables.act = args[2].equalsIgnoreCase("0") ? new Sigmoid() : new Tanh();
					Variables.dataSplitMethod = Integer.parseInt(args[3]);
					if(Variables.dataSplitMethod == 1){
						Variables.kValue = Integer.parseInt(args[4]);
					}
				}
				else{
					System.err.println("Not enough parameters for rprop");
					System.exit(1);
				}
			} 
			else{
				System.out.println("Incorrect training method");
				System.exit(1);
			}
		}
		
		
		new Main();
	}

}
