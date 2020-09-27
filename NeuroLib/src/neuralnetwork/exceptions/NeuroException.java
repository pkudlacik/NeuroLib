package neuralnetwork.exceptions;

public class NeuroException extends Exception {

	private static final long serialVersionUID = 1L;
	String err_msg;

	public NeuroException(String message) {
		err_msg = message;
	}

	public String what() {
		return err_msg;
	}

}
