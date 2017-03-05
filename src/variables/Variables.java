package variables;

import java.util.Random;

import activations.ActivationFunction;
import weightUpdate.WeightUpdate;

public class Variables {

	public static ActivationFunction act;
	public static WeightUpdate wUpdate;
	public static double learningRate;
	public static int dataSplitMethod = 1; // 0 - holdout, 1 - k-fold
	public static int kValue = 5;
	public static double momentumRate;
	public static int seed = 0;
	public static Random r = new Random();
	public static int hiddenNodeNum;
	
}
