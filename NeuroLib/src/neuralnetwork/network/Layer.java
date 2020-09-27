package neuralnetwork.network;

import neuralnetwork.neuron.Neuron;

public class Layer {

	public Neuron[] neurons;
	public int nCount;

	public Layer() {
		nCount = 0;
	}

	public Layer(int nodes) {
		nCount = 0;
		neurons = null;
		if (nodes <= 0)
			return;
		neurons = new Neuron[nodes];
		nCount = nodes;
	}

	public Layer(Layer src) {
		assign(src);
	}

	public Layer assign(Layer src) {
		restructure(src.nCount);
		for(int i=0; i<nCount; i++)
			neurons[i].assign(src.neurons[i]);
		return this;
		
	}

	public void restructure(int nodes) {
		if (nodes <= 0){
			return;
		}
		neurons = new Neuron[nodes];
		for (int i=0; i< nodes; i++){
			neurons[i] = new Neuron();
		}
		nCount = nodes;
	}

	public void restructure(int nodes, int weights) {
		if (nodes <= 0)
			return;
		if (weights < 0)
			return;
		neurons = new Neuron[nodes];
		nCount = nodes;
		
		for (int i = 0; i < nCount; i++) {
			neurons[i] = new Neuron();
			neurons[i].restructure(weights);
		}
	}

}
