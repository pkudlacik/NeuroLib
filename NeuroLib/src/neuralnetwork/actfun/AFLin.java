package neuralnetwork.actfun;

public class AFLin extends AFunction{

	private double Beta;
	
	public AFLin(){
		Beta = 1.0;
	}
	
	public AFLin(double B){
		Beta = B;
	}
	
	public AFLin(AFLin src){
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
	
		return Beta*x;
	}
	
	@Override
	public double deri(double x) {

		return Beta;
	}
}
