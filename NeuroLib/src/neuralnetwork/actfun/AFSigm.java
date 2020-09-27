package neuralnetwork.actfun;

public class AFSigm extends AFunction {

	private double Beta;

	public AFSigm() {
		Beta = 0.5;
	}

	public AFSigm(double B) {
		Beta = B;
	}

	public AFSigm(AFSigm src) {
		Beta = src.Beta;
	}

	public double getBeta() {
		return Beta;
	}

	public void setBeta(double beta) {
		Beta = beta;
	}

	@Override
	public double calc(double x) {

		return (1.0 / (1.0 + Math.exp(-Beta * x)));
	}

	@Override
	public double deri(double x) {

		double eBx;
		eBx = Math.exp(-Beta * x);
		return ((Beta * eBx) / ((1.0 + eBx) * (1.0 + eBx)));
	}

	public double deri_fast(double out) {
		return (Beta * out * (1.0 - out));
	}

}
