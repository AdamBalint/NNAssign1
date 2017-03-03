package variables;

import activations.ActivationFunction;
import weightUpdate.WeightUpdate;

public class Variables {

	public static ActivationFunction act;
	public static WeightUpdate wUpdate;
	public static double learningRate;
	public static int dataSplitMethod = 0; // 0 - holdout, 1 - k-fold
	public static int kValue;
	public static double momentumRate;
}
