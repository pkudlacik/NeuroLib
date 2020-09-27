package neuralnetwork.network;

import java.util.Random;
import java.util.Vector;

import neuralnetwork.actfun.AFSigm;
import neuralnetwork.actfun.AFunction;
import neuralnetwork.data.DataPackage;
import neuralnetwork.data.DataVector;

public class NNetFF {

	public int lCount;
	public int inputSize;
	private int maxSize;

	private boolean externalActFun;
	private boolean externalOutActFun;

	public AFunction actFun;
	public AFunction outActFun;

	public Layer Lrs[];

	public NNetFF() {
		lCount = 0;
		inputSize = 0;
		maxSize = 0;
		externalActFun = false;
		externalOutActFun = false;

		actFun = new AFSigm();
		outActFun = new AFSigm();
	}

	public NNetFF(NNetFF src) {
		assign(src);
	}

	public NNetFF assign(NNetFF src) {
		if (src.externalActFun) {
			actFun = src.actFun;
			externalActFun = true;
		} else {
			actFun = new AFSigm();
			externalActFun = false;
		}
		if (src.externalOutActFun) {
			outActFun = src.outActFun;
			externalOutActFun = true;
		} else {
			outActFun = new AFSigm();
			externalOutActFun = false;
		}

		lCount = src.lCount;
		inputSize = src.inputSize;
		maxSize = src.maxSize;
		Lrs = new Layer[lCount];

		for (int i = 0; i < lCount; i++) {
			Lrs[i] = new Layer();
			Lrs[i].assign(src.Lrs[i]);
		}

		return this;
	}

	public void setActFun(AFunction funct) {
		externalActFun = true;
		actFun = funct;

	}

	public void setOutActFun(AFunction funct) {
		externalOutActFun = true;
		outActFun = funct;

	}

	public void setNetworkSize(int numLayers) {
		Lrs = new Layer[numLayers];
		for (int i = 0; i < numLayers; i++) {
			Lrs[i] = new Layer();
		}
		lCount = numLayers;
	}

	public void setInputSize(int size) {
		if (lCount > 0) {
			for (int i = 0; i < Lrs[0].nCount; i++) {
				Lrs[0].neurons[i].restructure(size + 1);
			}
		}
		inputSize = size;
		if (inputSize > maxSize)
			maxSize = inputSize;
	}

	public boolean setLayerSize(int layer_number, int numNodes) {

		int layer = layer_number - 1;

		if (layer < 0 || layer_number > lCount) {
			return false;
		}

		if (numNodes <= 0) {
			return false;
		}

		Lrs[layer].restructure(numNodes);

		if (numNodes > maxSize) {
			maxSize = numNodes;
		}

		// ustalenie ilosci wag neuronow warstwy na podstawie ilosci neuronow w
		// poprzedniej + waga biasu
		// (przyjecie ilosci wag rownej "inputSize" dla warstwy pierwszej,
		// dodatkowa waga dla biasu)

		int nWeights = inputSize + 1;
		if (layer > 0) {
			nWeights = Lrs[layer - 1].nCount + 1;
		}

		for (int i = 0; i < lCount; i++) {
			for (int j = 0; j < numNodes; j++) {
				Lrs[layer].neurons[j].restructure(nWeights);

			}
		}

		// jesli istnieje warstwa nastepna to trzeba zmienic ilosc wag jej
		// neuronow na wartosc rowna ilosci elementow zmienianej warstwy + bias
		if (layer + 1 < lCount) {
			for (int i = 0; i < Lrs[layer + 1].nCount; i++)
				Lrs[layer + 1].neurons[i].restructure(numNodes + 1);
		}
		return true;
	}

	public int getLayerCount() {
		return lCount;
	}

	public int getMaxSize() {
		return maxSize;
	}

	public void initializeWeights() {
		initializeWeights(-1.0, 1.0);
	}

	public void initializeWeights(double min, double max) {
		int l, n, w; // layers, nodes,weights

		Random r = new Random();

		double range = (max - min);
		if (range < 0)
			range *= -1.0;

		for (l = 0; l < lCount; l++) {

			for (n = 0; n < Lrs[l].nCount; n++) {

				for (w = 0; w < Lrs[l].neurons[n].wCount; w++) {

					Lrs[l].neurons[n].Weights[w] = min + r.nextDouble() * range;

				}
			}
		}

	}

	public int setInput(DataVector input) {
		return setInput(input.getData());
	}

	public int setInput(Vector<Double> input) {

		int l, n, w;

		double result;

		// the first layer
		for (n = 0; n < Lrs[0].nCount; n++) {

			result = Lrs[0].neurons[n].Weights[Lrs[0].neurons[n].wCount - 1];

			for (w = 0; w < Lrs[0].neurons[n].wCount - 1; w++) {
				result += Lrs[0].neurons[n].Weights[w] * input.get(w);
			}
			Lrs[0].neurons[n].mpot = result;
			Lrs[0].neurons[n].out = actFun.calc(result);
		}

		// the rest of layers - except the last one
		for (l = 1; l < lCount - 1; l++) {

			for (n = 0; n < Lrs[l].nCount; n++) {
				result = Lrs[l].neurons[n].Weights[Lrs[l].neurons[n].wCount - 1];

				// bias value (the last weight)
				for (w = 0; w < Lrs[l].neurons[n].wCount - 1; w++) {
					result += Lrs[l].neurons[n].Weights[w] * Lrs[l - 1].neurons[w].out;
				}

				Lrs[l].neurons[n].mpot = result;
				Lrs[l].neurons[n].out = actFun.calc(result);
			}
		}

		// the last layer
		if (l < lCount) {
			for (n = 0; n < Lrs[l].nCount; n++) {
				result = Lrs[l].neurons[n].Weights[Lrs[l].neurons[n].wCount - 1];
				for (w = 0; w < Lrs[l].neurons[n].wCount - 1; w++) {
					// the rest of weights
					result += Lrs[l].neurons[n].Weights[w] * Lrs[l - 1].neurons[w].out;
				}
				Lrs[l].neurons[n].mpot = result;
				Lrs[l].neurons[n].out = outActFun.calc(result);
			}
		}

		return Lrs[0].nCount;
	}

	public int getResult(Vector<Double> output) {

		if (lCount <= 0)
			return 0;

		output.clear();

		for (int n = 0; n < Lrs[lCount - 1].nCount; n++) {
			output.add(Lrs[lCount - 1].neurons[n].out);
		}

		return Lrs[lCount - 1].nCount;
	}

	public int getResult(DataVector output) {
		return getResult(output.getData());
	}

	public int process(Vector<Double> input, Vector<Double> output) {
		if (lCount <= 0)
			return 0;

		setInput(input);
		return getResult(output);
	}

	public int process(DataVector input, DataVector output) {
		if (lCount <= 0)
			return 0;

		setInput(input.getData());
		return getResult(output);
	}

	public void processAll(DataPackage input, DataPackage output) {
		if (lCount <= 0)
			return;

		output.clear();

		for (int i=0; i < input.size(); i++) {
			DataVector row = input.get(i);
			setInput(row.getData());
			DataVector result = new DataVector();
			getResult(result);
			output.add(result);
		}
	}

}
