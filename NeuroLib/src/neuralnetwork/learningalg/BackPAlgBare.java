package neuralnetwork.learningalg;

import java.util.Vector;

import neuralnetwork.data.DataPackage;
import neuralnetwork.data.DataVector;
import neuralnetwork.exceptions.NeuroException;
import neuralnetwork.network.Layer;
import neuralnetwork.network.NNetFF;

public class BackPAlgBare extends SupervisedLearnAlg {

	protected double eta;

	protected Layer buff[] = new Layer[2];

	public void setEta(double e) {
		eta = e;
	}

	public BackPAlgBare() {
		super();
		eta = DEFAULT_ETA;

	}

	// zastanowiæ siê o final
	public BackPAlgBare(BackPAlgBare src) {
		super(src);
		eta = src.eta;
		_prepareBuffers();
	}

	protected double _calcError(Vector<Double> input, Vector<Double> output) {

		double tmp_err;
		double output_error;

		int layer_count = netFF.getLayerCount();

		netFF.setInput(input);

		output_error = 0.0;

		for (int n = 0; n < netFF.Lrs[layer_count - 1].nCount; n++) {

			tmp_err = output.get(n) - netFF.Lrs[layer_count - 1].neurons[n].out;
			if (tmp_err < 0)
				output_error -= tmp_err;
			else
				output_error += tmp_err;
		}

		return output_error;
	}

	protected double _getMaxError(Vector<Double> input, Vector<Double> output) {

		double tmp_err;
		double output_error;

		int layer_count = netFF.getLayerCount();

		netFF.setInput(input);

		output_error = 0.0;

		for (int n = 0; n < netFF.Lrs[layer_count - 1].nCount; n++) {

			tmp_err = output.get(n) - netFF.Lrs[layer_count - 1].neurons[n].out;
			if (tmp_err < 0)
				tmp_err = -tmp_err;
			if (tmp_err > output_error)	
				output_error = tmp_err;
		}

		return output_error;
	}

	/// Extends data buffers used in learning process.
	/// Assigns maximum size needed for used neural network if buffers' sizes
	/// are too small.
	protected void _prepareBuffers() {

		for (int i = 0; i < buff.length; i++) {
			buff[i] = new Layer();
		}

		// create data buffers using maximum size of network vector (input or layer)
		int max = netFF.getMaxSize() + 1; // extending size by one (because of bias)
		if (max > buff[0].nCount) {
			buff[0].restructure(max, max);
			buff[1].restructure(max, max);
		}
	}

	/// Initializes learning process using step by step method.
	/// Metod prepares needed buffer and initializes it.
	/// IMPORTANT ! Must be called once before learning process using
	/// learnOneStep()
	public void initStepLearning() {
		// create buffers for weights backup and temporary errors
		_prepareBuffers();
	}

	@Override
	public void setNeuralNetwork(NNetFF inetwork) {
		this.netFF = inetwork;

		if (netFF == null) {
			System.out.println("Neural Network is not assigned.");
		}

		_prepareBuffers();
	}

	@Override
	public void learnOneStep(Vector<Double> input, Vector<Double> output) throws NeuroException {

		if (netFF == null) {
			throw new NeuroException("Neural Network is not assigned");
		}

		int layer_count = netFF.getLayerCount();

		if (layer_count == 0) {
			throw new NeuroException("Neural network has no layers");
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

				// calculate weighted sum of previous errors
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
					netFF.Lrs[l + 1].neurons[n].Weights[w] += buff[prev].neurons[n].Weights[w];
				}
			}

			actu = actu ^ 1; // change actual index to an opposite state (0 <-> 1)
			prev = prev ^ 1; // change previous index to an opposite state (0 <-> 1)

		}

		// change weights of the first layer
		for (n = 0; n < netFF.Lrs[0].nCount; n++) {
			for (w = 0; w < netFF.Lrs[0].neurons[n].wCount; w++) {
				netFF.Lrs[0].neurons[n].Weights[w] += buff[prev].neurons[n].Weights[w];
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
		int i;

		int data_size;

		last_error = max_error + 1.0;
		data_size = input.size();

		if (input.size() >= output.size()) {
			data_size = output.size();
		}

		_prepareBuffers();

		while ((iter_count < max_iter) && (last_error > max_error)) {
			last_error = 0.0;
			for (i = 0; i < data_size; i++) {
				learnOneStep(input.get(i).getData(), output.get(i).getData());
				last_error += _calcError(input.get(i).getData(), output.get(i).getData());
			}
			iter_count++;
		}
		last_iter = iter_count;
		return last_error;
	}

	@Override
	public double learnForMaxError(DataPackage input, DataPackage output) throws NeuroException {
		if (netFF == null) {
			throw new NeuroException("Neural Network is not assigned.");
		}
		if (netFF.lCount <= 0) {
			throw new NeuroException("Neural Network has no layers.");
		}
		if (input.getMinRowSize() < netFF.inputSize) {
			throw new NeuroException("DataVector size of input DataPackage is too small to match input size of the network.");
		}
		if (output.getMinRowSize() < netFF.Lrs[netFF.lCount - 1].nCount) {
			throw new NeuroException("Size of output DataVector is too small to match output size of the network.");
		}

		long iter_count = 0;
		int i;

		int data_size;

		last_error = max_error + 1.0;
		data_size = input.size();

		if (input.size() >= output.size()) {
			data_size = output.size();
		}

		_prepareBuffers();

		while ((iter_count < max_iter) && (last_error > max_error)) {
			last_error = 0.0;
			for (i = 0; i < data_size; i++) {
				learnOneStep(input.get(i).getData(), output.get(i).getData());
				double error = _getMaxError(input.get(i).getData(), output.get(i).getData());
				if (error > last_error) {
					last_error = error;
				}
			}
			iter_count++;
		}
		last_iter = iter_count;
		return last_error;
	}
}
