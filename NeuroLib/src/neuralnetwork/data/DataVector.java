package neuralnetwork.data;

import java.util.Random;
import java.util.Vector;

public class DataVector {

	Vector<Double> data;

	public DataVector() {
		data = new Vector<Double>();
	}

	public DataVector(int capacity) {
		data = new Vector<Double>(capacity);
	}

	public DataVector(double[] buf) {
		data = new Vector<>(buf.length);
		for (int i = 0; i < buf.length; i++) {
			data.add(buf[i]);
		}
	}

	public DataVector(Vector<Double> buf) {
		data = new Vector<Double>(buf.size());
		assign(buf, 0, buf.size() - 1);
	}

	public DataVector(DataVector src) {
		data = new Vector<Double>();
		assign(src.data, 0, src.data.size() - 1);
	}

	public int size() {
		return data.size();
	}

	public Vector<Double> getData() {
		return data;
	}

	public void setData(Vector<Double> data) {
		this.data = data;
	}

	public Double get(int index) {
		return data.get(index);
	}

	public Double set(int index, Double element) {
		return data.set(index, element);
	}

	/**
	 * Merges two vectors (this and source) by adding content of source
	 * 
	 * @param source source data
	 * @return number of added values
	 */
	public int merge(DataVector source) {
		return merge(source.data);
	}

	/**
	 * Extends vector data by adding content of source
	 * 
	 * @param source source data
	 * @return number of added values
	 */
	public int merge(double [] source) {
		for (int i = 0; i < source.length; i++) {
			data.add(source[i]);
		}
		return source.length;
	}

	/**
	 * Extends vector data by adding content of source
	 * 
	 * @param source source data
	 * @return number of added values
	 */
	public int merge(Vector<Double> source) {
		for (int i = 0; i < source.size(); i++) {
			data.add(source.get(i));
		}
		return source.size();
	}

	/**
	 * Assigns data do DataVector from given source (the data is copied)
	 * 
	 * @param buf      table of doubles with data
	 * @param firstIdx first index to be copied
	 * @param lastIdx  last index to be copied
	 * @return number of copied values
	 */
	public int assign(double[] buf, int firstIdx, int lastIdx) {
		int size = buf.length;
		if (size == 0)
			return 0;

		if (firstIdx < 0)
			firstIdx = 0;
		if ((lastIdx > size - 1) || lastIdx < 0)
			lastIdx = size - 1;
		if (firstIdx > lastIdx)
			firstIdx = lastIdx;

		if ((lastIdx - firstIdx + 1) > data.capacity()) {
			data = new Vector<Double>(lastIdx - firstIdx + 1);
		} else {
			data.clear();
		}

		for (int i = 0; i < size; i++) {
			data.add(buf[i]);
		}

		return lastIdx - firstIdx + 1;
	}

	/**
	 * Assigns data do DataVector from given source (the data is copied)
	 * 
	 * @param source   DataVector with data
	 * @param firstIdx first index to be copied
	 * @param lastIdx  last index to be copied
	 * @return number of copied values
	 */
	public int assign(DataVector source, int firstIdx, int lastIdx) {
		return assign(source.data, firstIdx, lastIdx);
	}

	/**
	 * Assigns data do DataVector from source vector of doubles (the data is copied)
	 * 
	 * @param source   vector of doubles with data
	 * @param firstIdx first index to be copied
	 * @param lastIdx  last index to be copied
	 * @return number of copied values
	 */
	public int assign(Vector<Double> source, int firstIdx, int lastIdx) {
		int size = source.size();
		if (size == 0)
			return 0;

		if (firstIdx < 0)
			firstIdx = 0;
		if ((lastIdx > size - 1) || lastIdx < 0)
			lastIdx = size - 1;
		if (firstIdx > lastIdx)
			firstIdx = lastIdx;

		if ((lastIdx - firstIdx + 1) > data.capacity()) {
			data = new Vector<Double>(lastIdx - firstIdx + 1);
		} else {
			data.clear();
		}

		for (int i = firstIdx; i <= lastIdx; i++) {
			data.add(source.get(i));
		}

		return lastIdx - firstIdx + 1;
	}

	/**
	 * Returns DataVector values in form of a new array of doubles
	 * 
	 * @return content of DataVector in array of doubles
	 */
	public double[] toArrayOfDoubles() {

		double[] destination = new double[data.size()];

		for (int i = 0; i < data.size(); i++) {
			destination[i] = data.get(i);
		}

		return destination;
	}

	/**
	 * Returns an arithmetic average of all values
	 * 
	 * @return average
	 */
	public double getAverage() {

		double sum = 0.0;

		for (int i = 0; i < data.size(); i++) {
			sum += data.get(i);
		}

		return sum/data.size();
	}

	@Override
	public String toString() {

		String output = new String();
		
		boolean first = true;
		for (Double d : data) {
			if (first) {
				first = false;
			} else {
				output += ",\t";
			}
			output += d;
		}
		
		return output;
	}

	/**
	 * Creates new values in [min, max] range. Old content is cleared.
	 * 
	 * @param number number of new values
	 * @param min    minimum value
	 * @param max    maximum value
	 */
	public void createRandomContent(int number, double min, double max) {
		if (number <= 0)
			return;

		Random r = new Random();
		double range = (max - min);
		if (range < 0) {
			range *= -1.0;
		}

		if (number > data.capacity()) {
			data = new Vector<Double>(number);
		} else {
			data.clear();
		}
		for (int i = 0; i < number; i++) {
			data.add(min + r.nextDouble() * (range));
		}
	}

	/**
	 * Loads data given in text form. Does not generate any exceptions on parse error. The values are omitted
	 * 
	 * @param line              Text line containing separated double values to
	 *                          parse
	 * @param field_separator   Regular expression containing one or more field
	 *                          separators: ";" - only semicolon or "[ ;\\t]" -
	 *                          space, semicolon or tabulator
	 * @param decimal_separator Decimal separator ('.' or ','). Type null for
	 *                          default: '.'
	 * @param firstIdx          First index of loaded values - starting from 0
	 *                          (first ones can be omitted). Type null for all.
	 * @param lastIdx           Last index of loaded values (higher indexes will be
	 *                          omitted). Type null for all.
	 */
	public void parse(String line, String field_separator, Character decimal_separator, Integer firstIdx,
			Integer lastIdx, boolean ignoreParseErrors) {
		if (firstIdx == null) {
			firstIdx = 0;
		}

		if (lastIdx != null) {
			if (lastIdx < firstIdx) {
				lastIdx = null;
			}
		}

		if (decimal_separator == null) {
			decimal_separator = '.';
		}

		Vector<Double> vector = null;
		if (lastIdx != null) { // if lastIdx specified - limit capacity
			vector = new Vector<Double>(lastIdx + 1);
		} else {
			vector = new Vector<Double>();
		}
		String[] splitted = line.split(field_separator);
		boolean error = false;
		int index = 0;
		for (String v : splitted) {
			if (index >= firstIdx) { // omit first values
					v = v.trim();
					if (!v.isEmpty()) {
						if (decimal_separator != '.') {
							v = v.replace(decimal_separator, '.');
						}
						Double d = null;
						try {						
							d = Double.parseDouble(v);
						} catch (NumberFormatException e) {
							if (!ignoreParseErrors) {
								error = true;
							}
						}
						vector.add(d);
					}
			}
			index++;
			if (lastIdx != null && index > lastIdx) {
				break; // end loop if last value obtained
			}
		}
		if (!error && !vector.isEmpty()) {
			assign(vector, 0, vector.size() - 1);
		}
	}

}
