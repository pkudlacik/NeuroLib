package neuralnetwork.neuron;

public class Neuron {
	// wyjœcie
	public double out;

	public double mpot;

	// wagi
	public double[] Weights;

	public int wCount;

	public Neuron() {
		out = 0.0;
		wCount = 0;
		mpot = 0.0;
	}

	public Neuron(Neuron n) {
		assign(n);
	}

	public Neuron(int elems) {
		out = 0.0;
		wCount = 0;
		mpot = 0.0;
		Weights = null;

		if (elems <= 0)
			return;

		Weights = new double[elems];
		wCount = elems;

	}

	public void restructure(int elems) {
		if (elems <= 0)
			return;

		Weights = new double[elems];
		wCount = elems;
		mpot = 0.0;
		out = 0.0;

	}

	public Neuron assign(Neuron n) {
		if (n.wCount <= 0) {
			Weights = null;
			wCount = 0;
			out = 0;
			mpot = 0;
			return this;
		}
		Weights = new double[n.wCount];
		for (int i = 0; i < n.wCount; i++) {
			Weights[i] = n.Weights[i];
			wCount = n.wCount;
			mpot = n.mpot;
			out = n.out;
		}
		return this;
	}

}
