package weightUpdate;

import Jama.Matrix;
import neuralNetwork.Layer;

public abstract class WeightUpdate {

	public abstract Matrix getWeightUpdate(Matrix errGrad, Layer l);
	public abstract WeightUpdate copy(Layer l);
}
