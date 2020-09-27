package neuralnetwork.learningalg;

import java.util.Vector;

import neuralnetwork.data.DataPackage;
import neuralnetwork.data.DataVector;
import neuralnetwork.exceptions.NeuroException;
import neuralnetwork.network.NNetFF;

public abstract class SupervisedLearnAlg {

	public static final double DEFAULT_ERROR = 0.1;
	public static final int DEFAULT_MAX_ITER = 1000;
	public static final double DEFAULT_ETA = 0.1;

	protected double max_error;
	protected long max_iter;
	protected double last_error;
	protected long last_iter;

	protected NNetFF netFF;

	public SupervisedLearnAlg() {
		max_error = DEFAULT_ERROR;
		max_iter = DEFAULT_MAX_ITER;
		netFF = null;
	}

	public SupervisedLearnAlg(SupervisedLearnAlg src) {
		max_error = src.max_error;
		max_iter = src.max_iter;
		netFF = src.netFF;
	}

	public void setNeuralNetwork(NNetFF inetwork) {
		netFF = inetwork;
	}

	public NNetFF getNeuralNetwork() {
		return this.netFF;
	}

	public void setMaxError(double err) {
		max_error = err;
	}

	public double getMaxError() {
		return max_error;
	}

	public void setMaxIter(long max) {
		max_iter = max;
	}

	public long getMaxIter() {
		return max_iter;
	}

	public double getLastError() {
		return last_error;
	}

	public long getLastIter() {
		return last_iter;
	}

	public void learnOneStep(Vector<Double> buf1, Vector<Double> buf2) throws NeuroException{}

	public void learnOneStep(DataVector input, DataVector output) throws NeuroException {}

	public double learn(DataPackage input, DataPackage output) throws NeuroException {
		return 1.0;
	}

	public double learnEx(DataPackage input, DataPackage output) throws NeuroException {
		return 1.0;
	}
}
