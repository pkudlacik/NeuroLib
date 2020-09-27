package neuralnetwork.learningalg;

import java.util.Vector;

import neuralnetwork.data.DataPackage;
import neuralnetwork.data.DataVector;
import neuralnetwork.exceptions.NeuroException;
import neuralnetwork.network.NNetFF;

public class BackPAlgM extends BackPAlgBare {

	public static final double DEFAULT_MOMENTUM = 0.1;
	public static final double DEFAULT_MSTEP = DEFAULT_MOMENTUM / DEFAULT_MAX_ITER;
	public static final double DEFAULT_LOWESTM = 0.0;

	protected double M;
	protected double stepM;
	protected double lowestM;
	NNetFF wBackup;

	public void setMomentum(double momentum) {
		M = momentum;
	}

	public void setMomentumStep(double step) {
		stepM = step;
	}

	public void setLowestMomentum(double momentum) {
		lowestM = momentum;
	}

	public BackPAlgM() {
		super();
		M = DEFAULT_MOMENTUM;
		stepM = DEFAULT_MSTEP;
		lowestM = DEFAULT_LOWESTM;
	}

	public BackPAlgM(BackPAlgM src) {
		super(src);

		M = src.M;
		stepM = src.stepM;
		lowestM = src.lowestM;
	}

	@Override
	protected void _prepareBuffers() {

		if (netFF == null) {
			return;
		}

		super._prepareBuffers();

		wBackup = new NNetFF();
		wBackup.assign(netFF);
	}

	/// Initializes learning process using step by step method.
	/// Metod prepares needed buffer and initializes it.
	/// IMPORTANT ! Must be called once before learning process using
	/// learnOneStep()
	public void initStepLearning() {
		// create buffers for weights backup and temporary errors
		// and copy weights from network to wBackup
		_prepareBuffers();

		//first change times momentum should be 0
		wBackup.initializeWeights(0.0, 0.0);
	}

	@Override
	public void learnOneStep(Vector<Double> input, Vector<Double> output) throws NeuroException {

		if (netFF == null) {
			throw new NeuroException("Neural Network is not assigned.");
		}

		int layer_count = netFF.getLayerCount();

		if (layer_count == 0) {
			throw new NeuroException("Neural Network has no layers");
		}

		// calculate intermediate and last results before changes
		netFF.setInput(input);

		int l, n, n_tmp, w;
		int prev = 0;
		int actu = 1;
		double temp;

		// calculate errors of the last layer
		for (n = 0; n < netFF.Lrs[layer_count - 1].nCount; n++) {

			// calculate error for this neuron
			buff[actu].neurons[n].out = (netFF.outActFun.deri(netFF.Lrs[layer_count - 1].neurons[n].mpot))
					* (output.get(n) - netFF.Lrs[layer_count - 1].neurons[n].out);

			// calculate weights' change for this neuron
			// first bias value (input = 1.0)
			buff[actu].neurons[n].Weights[netFF.Lrs[layer_count - 1].neurons[n].wCount - 1] = eta
					* buff[actu].neurons[n].out;

			// then the rest of values
			if (layer_count == 1) { // if only one layer => take input vector as
									// an input to the layer
				for (w = 0; w < netFF.Lrs[layer_count - 1].neurons[n].wCount; w++) {
					buff[actu].neurons[n].Weights[w] = eta * buff[actu].neurons[n].out * input.get(w);
				}
			} else { // if more than one layer => take output of previous layer as an input
				for (w = 0; w < netFF.Lrs[layer_count - 1].neurons[n].wCount - 1; w++) {
					buff[actu].neurons[n].Weights[w] = eta * buff[actu].neurons[n].out
							* netFF.Lrs[layer_count - 2].neurons[w].out;
				}
			}
		}

		actu = actu ^ 1; // change actual index to an opposite state (0 <-> 1)
		prev = prev ^ 1; // change previous index to an opposite state (0 <-> 1)

		// calculate errors for the rest of layers
		for (l = layer_count - 2; l >= 0; l--) {

			// calculate errors for one layer
			for (n = 0; n < netFF.Lrs[l].nCount; n++) {

				// calculate weighted sum to previous errors
				temp = 0.0;
				for (n_tmp = 0; n_tmp < netFF.Lrs[l + 1].nCount; n_tmp++) {
					temp += netFF.Lrs[l + 1].neurons[n_tmp].Weights[n] * buff[prev].neurons[n_tmp].out;
				}

				// calculate error for this neuron
				buff[actu].neurons[n].out = netFF.actFun.deri(netFF.Lrs[l].neurons[n].mpot) * temp;

				// calculate weights' change for this neuron
				// first bias value (input = 1.0)
				buff[actu].neurons[n].Weights[netFF.Lrs[l].neurons[n].wCount - 1] = eta * buff[actu].neurons[n].out;

				// then the rest of values
				if (l == 0) { // if the first layer => take input vector as an input to the layer
					for (w = 0; w < netFF.Lrs[l].neurons[n].wCount - 1; w++) {
						buff[actu].neurons[n].Weights[w] = eta * buff[actu].neurons[n].out * input.get(w);
					}
				} else { // if not the first layer => take output of previous layer as an input
					for (w = 0; w < netFF.Lrs[l].neurons[n].wCount - 1; w++) {
						buff[actu].neurons[n].Weights[w] = eta * buff[actu].neurons[n].out
								* netFF.Lrs[l - 1].neurons[w].out;
					}
				}
			}

			// change weights of a previous layer
			for (n = 0; n < netFF.Lrs[l + 1].nCount; n++) {
				for (w = 0; w < netFF.Lrs[l + 1].neurons[n].wCount; w++) {
					// calculate momentum part : last change times momentum factor
					temp = M * (wBackup.Lrs[l + 1].neurons[n].Weights[w]);
					// calculate and save weight change : actual error + momentum part
					wBackup.Lrs[l + 1].neurons[n].Weights[w] = buff[prev].neurons[n].Weights[w] + temp;
					// apply actual change
					netFF.Lrs[l + 1].neurons[n].Weights[w] += wBackup.Lrs[l + 1].neurons[n].Weights[w];
				}
			}

			actu = actu ^ 1; // change actual index to an opposite state (0 <-> 1)
			prev = prev ^ 1; // change previous index to an opposite state (0 <-> 1)

		}

		// change weights of the first layer
		for (n = 0; n < netFF.Lrs[0].nCount; n++) {
			for (w = 0; w < netFF.Lrs[0].neurons[n].wCount; w++) {
				// calculate momentum part : last change times momentum factor
				temp = M * (wBackup.Lrs[0].neurons[n].Weights[w]);
				// calculate and save weight change : actual error + momentum part
				wBackup.Lrs[0].neurons[n].Weights[w] = buff[prev].neurons[n].Weights[w] + temp;
				// apply actual change
				netFF.Lrs[0].neurons[n].Weights[w] += wBackup.Lrs[0].neurons[n].Weights[w];
			}
		}
	}

	@Override
	public void learnOneStep(DataVector input, DataVector output) throws NeuroException {
		learnOneStep(input.getData(), output.getData());
	}

	@Override
	public double learn(DataPackage input, DataPackage output) throws NeuroException {

		if (netFF == null) {
			throw new NeuroException("Neural Network is not assigned.");
		}
		if (netFF.lCount <= 0) {
			throw new NeuroException("Neural Network has no layers.");
		}
		if (input.getMinRowSize() < netFF.inputSize) {
			throw new NeuroException("DataVector size of input package is too small to match input size of the network.");
		}
		if (output.getMinRowSize() < netFF.Lrs[netFF.lCount - 1].nCount) {
			throw new NeuroException("Size of output DataVector is too small to match output size of the network.");
		}

		long iter_count = 0;
		int i, data_size;
		double old_M = M;

		last_error = max_error + 1.0;

		if (input.size() >= output.size()) {
			data_size = output.size();
		} else {
			data_size = input.size();
		}

		initStepLearning();
		if (M < lowestM) {
			M = lowestM;
		}

		while ((iter_count < max_iter) && (last_error > max_error)) {

			last_error = 0.0;
			for (i = 0; i < data_size; i++) {
				learnOneStep(input.get(i).getData(), output.get(i).getData());
				last_error += _calcError(input.get(i).getData(), output.get(i).getData());
			}

			M -= stepM;
			if (M < lowestM) {
				M = lowestM;
			}
			iter_count++;

		}

		last_iter = iter_count;
		M = old_M;

		return last_error;
	}

	@Override
	public double learnEx(DataPackage input, DataPackage output) throws NeuroException {

		if (netFF == null) {
			throw new NeuroException("Neural Network is not assigned.");
		}
		if (netFF.lCount <= 0) {
			throw new NeuroException("Neural Network has no layers.");
		}
		if (input.getMinRowSize() < netFF.inputSize) {
			throw new NeuroException("DataVector size of input package is too small to match input size of the network.");
		}
		if (output.getMinRowSize() < netFF.Lrs[netFF.lCount - 1].nCount) {
			throw new NeuroException("Size of output DataVector is too small to match output size of the network.");
		}

		long iter_count = 0;
		int i, data_size;
		double old_M = M;

		last_error = max_error + 1.0;

		if (input.size() >= output.size()) {
			data_size = output.size();
		} else {
			data_size = input.size();
		}

		initStepLearning();
		if (M < lowestM) {
			M = lowestM;
		}

		while ((iter_count < max_iter) && (last_error > max_error)) {

			last_error = 0.0;
			for (i = 0; i < data_size; i++) {
				learnOneStep(input.get(i).getData(), output.get(i).getData());
				double error = _calcError(input.get(i).getData(), output.get(i).getData());
				if (error > last_error) {
					last_error = error;
				}
			}

			M -= stepM;
			if (M < lowestM) {
				M = lowestM;
			}
			iter_count++;

		}

		last_iter = iter_count;
		M = old_M;

		return last_error;
	}

}
