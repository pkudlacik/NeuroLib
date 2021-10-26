package neuralnetwork.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Random;
import java.util.Vector;

public class DataPackage {
	private List<DataVector> data = new LinkedList<DataVector>();

	private Vector<Double> mins = new Vector<Double>();
	private Vector<Double> maxs = new Vector<Double>();

	private int min_size = 0;
	private int max_size = 0;

	// regular expression describing one data row in the input file
	String regexp = "[ \\t]*(-?\\d+([.,]\\d+)?[ ;\\t]+)*(-?\\d+([.,]\\d+)?[ ;\\t]*)";
	// regular expression for separator between values of one row
	String sep_regexp = "[ ,;\\t]";
	char decimal_separator = '.';

	private boolean ignore_parse_errors = false;

	public DataPackage() {
	}

	/**
	 * Create a new object copying data from source package.
	 * 
	 * @param src
	 */
	public DataPackage(DataPackage src) {
		_assign(src);
	}

	private void _assign(DataPackage src) {
		min_size = src.min_size;
		max_size = src.max_size;
		for (Double d : src.mins) {
			mins.add(d);
		}
		for (Double d : src.maxs) {
			maxs.add(d);
		}
		for (DataVector src_row : src.data) {
			DataVector row = new DataVector(src_row);
			data.add(row);
		}
	}

	/**
	 * Copies the source package into this object. Previous content is cleared.
	 * 
	 * @param src
	 */
	public void assign(DataPackage src) {
		clear();
		_assign(src);
	}

	/**
	 * The method add() allows to add vectors of variable length. The method allows
	 * to check the minimum size of stored vectors.
	 * 
	 * @return minimum row size
	 */
	public int getMinRowSize() {
		return min_size;
	}

	/**
	 * The method add() allows to add vectors of variable length. The method allows
	 * to check the maximum size of stored vectors.
	 * 
	 * @return minimum row size
	 */
	public int getMaxRowSize() {
		return max_size;
	}

	/**
	 * Specifies a field separator string in a text file. It can be one character or
	 * more, defined with a regular expression. For instance ";" - only semicolon,
	 * but "[ ,;\\t]" - defines space, colon, semicolon or tabulator as a separator
	 * (this is default).
	 * 
	 * @param field_separator Regular expression containing one or more field
	 *                        separators
	 */
	public void setFieldSeparator(String field_separator) {
		this.sep_regexp = field_separator;
	}

	/**
	 * Defines a decimal separator. Default is '.', but for instance Polish or
	 * French locale use ','
	 * 
	 * @param decimal_separator Use '.' or ','
	 */
	public void setDecimalSeparator(char decimal_separator) {
		this.decimal_separator = decimal_separator;
	}

	public void setRowRegularExpression(String regexp) {
		this.regexp = regexp;
	}

	public boolean isIgnoreParseErrors() {
		return ignore_parse_errors;
	}

	public void setIgnoreParseErrors(boolean ignoreParseErrors) {
		this.ignore_parse_errors = ignoreParseErrors;
	}

	/**
	 * @return number of stored data vectors
	 */
	public int size() {
		return data.size();
	}

	/**
	 * Use this method for data retrieval only (like for each or obtaining an
	 * iterator to go through the list). Modifying vectors this way does not update
	 * the statistics (like min and max vector length or minimum and maximum of each
	 * column). It can cause normalize to work incorrectly.
	 * 
	 * In order to add a new vector use add() method of DataPackage.
	 * 
	 * @return list of data vectors
	 */
	public List<DataVector> getList() {
		return data;
	}

	/**
	 * Retrieves the data vector by given index.
	 * 
	 * @param index index of a data vector
	 * @return data vector object
	 */
	public DataVector get(int index) {
		return data.get(index);
	}

	/**
	 * @return the vector containing minimum values for subsequent columns
	 */
	public Vector<Double> getMins() {
		return mins;
	}

	/**
	 * @return the vector containing maximum values for subsequent columns
	 */
	public Vector<Double> getMaxs() {
		return maxs;
	}

	/**
	 * Returns the range for specified column (max - min of a column)
	 * 
	 * @param column column index
	 * @return range of a column
	 */
	public double getColumnRange(int column) {
		return maxs.get(column) - mins.get(column);
	}

	/**
	 * Adds a data vector updating the statistics. The most important are minimum
	 * and maximum values of each column - it is crucial for the normalization
	 * process (method normalize()). It also updates the minimum and maximum vector
	 * length. Do not add vectors using raw List<DataVector> returned by getList()
	 * method.
	 * 
	 * @param row
	 */
	public void add(DataVector row) {
		// 1. add row
		data.add(row);
		// 2. check if longer or shorter than max/min size
		int old_size = mins.size();
		int size = row.size();
		if (size < min_size || data.size() == 1)
			min_size = size;
		if (size > max_size) {
			max_size = size; // set the new size
			// 2.1 extend values in mins and maxs vector
			for (int i = old_size; i < size; i++) {
				// copy new content (first values are max and min)
				mins.add(i, row.get(i));
				maxs.add(i, row.get(i));
			}
		}
		// 3. check and set mins and maxs
		// - only remaining first if size > max_size
		for (int i = 0; i < old_size; i++) {
			// if value not null
			if (row.get(i) != null) {
				// check and set min/max
				if (mins.get(i) == null || mins.get(i) > row.get(i)) {
					mins.set(i, row.get(i));
				}
				if (maxs.get(i) == null || maxs.get(i) < row.get(i)) {
					maxs.set(i, row.get(i));
				}
			}
		}

	}

	/**
	 * Adds all rows from source package at the end of list.
	 * 
	 * @param src Source data package.
	 */
	public void add(DataPackage src) {
		for (DataVector vector : src.data) {
			add(vector);
		}
	}

	/**
	 * Merges data package with source package. Every subsequent data vector in
	 * package is merged with corresponding (subsequent) vector in source package by
	 * adding more columns. Therefore, the source package must have at least the
	 * same number of rows (otherwise the method returns without any action).
	 * 
	 * @param src Source data package.
	 */
	public void merge(DataPackage src) {

		if (src.size() < this.size()) {
			return;
		}

		ListIterator<DataVector> itSource = src.data.listIterator();
		DataPackage pack = new DataPackage();

		for (DataVector v : data) {
			v.merge(itSource.next());
			pack.add(v);
		}

		mins = pack.mins;
		maxs = pack.maxs;
		data = pack.data;
		min_size = pack.min_size;
		max_size = pack.max_size;

	}

	/**
	 * Splits data package by column. Creates two DataPackage objects containing the
	 * same number of rows but store subsets defined by given column. This object
	 * after split stores values up to given column, and the returned new object
	 * contains values above specified column index. For instance, if column = 2,
	 * this object will preserve columns 0,1 and 2. The remaining columns (3 and
	 * above) will be returned in new object.
	 * 
	 * @param column a column number to split by
	 * @return second part after split
	 */
	public DataPackage splitByColumn(int column) {
		DataPackage first = new DataPackage();
		DataPackage second = new DataPackage();
		if (column > (min_size - 2)) {
			return second;
		}

		for (DataVector v : data) {
			DataVector first_row = new DataVector(column + 1);
			DataVector second_row = new DataVector(v.size() - (column + 1));
			for (int i = 0; i <= column; i++) {
				first_row.getData().add(v.get(i));
			}
			for (int i = column + 1; i < v.size(); i++) {
				second_row.getData().add(v.get(i));
			}
			first.add(first_row);
			second.add(second_row);
		}

		mins = first.mins;
		maxs = first.maxs;
		data = first.data;
		min_size = max_size = column + 1;

		return second;
	}

	/**
	 * Splits data package by row. Creates two DataPackage objects containing the
	 * same number of columns but store subsets defined by given row index. This
	 * object after split stores values up to given row, and the returned new object
	 * contains values above specified row index. For instance, if row = 2, this
	 * object will preserve rows 0,1 and 2. The remaining rows (3 and above) will be
	 * returned in new object.
	 * 
	 * @param raw a raw number to split by
	 * @return second part after split
	 */
	public DataPackage splitByRow(int row) {
		DataPackage first = new DataPackage();
		DataPackage second = new DataPackage();
		if (row > (data.size() - 2)) {
			return second;
		}

		int idx = 0;
		for (DataVector v : data) {
			if (idx <= row) {
				DataVector first_row = new DataVector(v);
				first.add(first_row);
			} else {
				DataVector second_row = new DataVector(v);
				second.add(second_row);
			}
			idx++;
		}

		mins = first.mins;
		maxs = first.maxs;
		data = first.data;
		min_size = first.min_size;
		max_size = first.max_size;

		return second;
	}

	/**
	 * Remove specified number of random vectors. The list of removed rows is
	 * returned in form of a new DataPackage object
	 * 
	 * @param number Number of vectors (rows) to remove
	 * @return The list of removed vectors.
	 */
	public DataPackage removeRandomVectors(int number) {
		DataPackage removed = new DataPackage();

		if (number > data.size()) {
			number = data.size();
		}

		Random r = new Random();
		for (int i = 0; i < number; i++) {
			int randomIndex = (int) Math.round(r.nextDouble() * (data.size() - 1));
			DataVector vector = data.remove(randomIndex);
			removed.add(vector);
		}

		return removed;
	}

	/**
	 * Remove vectors for which specified column equal given value. The list of
	 * removed rows is returned in form of a new DataPackage object
	 * 
	 * @param columnIndex index of a column to search (0 is the first index)
	 * @param value       a value specifying
	 * @return The list of removed vectors.
	 */
	public DataPackage removeVectorsByColumnValue(int columnIndex, double value) {
		DataPackage removed = new DataPackage();

		for (int i = 0; i < data.size(); i++) {
			DataVector v = data.get(i);
			if (v.get(columnIndex) == value) {
				removed.add(data.remove(i));
				i--; // update index after item removal
			}
		}

		return removed;
	}

	/**
	 * Remove specified vector form package. A new package containing removed row is
	 * returned
	 * 
	 * @param number Number of vectors (rows) to remove
	 * @return The list of removed vectors.
	 */
	public DataPackage removeVector(int index) {
		DataPackage removed = new DataPackage();

		if (index < 0 || index >= data.size()) {
			return removed;
		}

		removed.add(data.remove(index));

		return removed;
	}

	/**
	 * Clears data and statistics stored in this object.
	 */
	public void clear() {
		data.clear();
		mins.clear();
		maxs.clear();
		min_size = max_size = 0;
	}

	/**
	 * Creates a new static content (the same value). Previous content is cleared.
	 * The method creates specified number of rows containing specified number
	 * (cols) of constant values (value).
	 * 
	 * @param rows  new number of rows
	 * @param cols  new number of columns (elements in each row)
	 * @param value a constant value of all package elements
	 */
	public void createConstantContent(int rows, int cols, double value) {
		if (rows <= 0 || cols <= 0)
			return;

		clear();

		for (int i = 0; i < rows; i++) {
			DataVector row = new DataVector(cols);
			for (int j = 0; j < cols; j++) {
				row.data.add(value);
			}
			add(row);
		}

		min_size = max_size = cols;
	}

	/**
	 * Creates a new random content in defined range. Previous content is cleared.
	 * The method creates specified number of rows containing specified number
	 * (cols) of random values in [min, max] range.
	 * 
	 * @param rows new number of rows
	 * @param cols new number of columns (elements in each row)
	 * @param min  minimum value
	 * @param max  maximum value
	 */
	public void createRandomContent(int rows, int cols, double min, double max) {
		if (rows <= 0 || cols <= 0)
			return;

		Random r = new Random();

		double range = (max - min);
		if (range < 0) {
			range *= -1.0;
		}

		clear();
		for (int i = 0; i < rows; i++) {
			DataVector row = new DataVector(cols);
			for (int j = 0; j < cols; j++) {
				row.data.add(min + r.nextDouble() * range);
			}
			add(row);
		}
		min_size = max_size = cols;
	}

	/**
	 * Scales all values in loaded list to [0,1] range. The process is performed for
	 * each column separately.
	 */
	public void normalize() {
		// calculate range
		int size = maxs.size();
		for (int i = 0; i < size; i++) {
			Double d = maxs.get(i);
			d -= mins.get(i);
			maxs.set(i, d);
		}

		// normalize
		for (DataVector v : data) {
			size = v.size();
			for (int i = 0; i < size; i++) {
				if (maxs.get(i) == 0.0) {
					v.set(i, 0.0);
				} else {
					v.set(i, (v.get(i) - mins.get(i)) / maxs.get(i));
				}
			}
		}

		// update mins and maxs
		for (int i = 0; i < size; i++) {
			if (maxs.get(i) != 0.0) {
				maxs.set(i, 1.0);
			}
			mins.set(i, 0.0);
		}
	}

	/**
	 * Loads all data from text file. Lines without properly separated double values
	 * are omitted. The separators are defined by setFieldSeparator() method. The
	 * decimal separator is defined by setDecimalSepartor() method.
	 */
	public boolean loadTextFile(String file) {
		return loadTextFile(file, null, 0, 0, null);
	}

	/**
	 * Loads data from text file. Lines without properly separated double values are
	 * omitted. The separators are defined by setFieldSeparator() method. The
	 * decimal separator is defined by setDecimalSepartor() method. Parameters allow
	 * to define maximum number of loaded vectors, line offset or choose limit the
	 * first and last column.
	 * 
	 * @param file            File name (a whole path if needed)
	 * @param numberOfVectors Maximum number of vectors to be loaded - empty or
	 *                        description lines are not counted (lines without
	 *                        properly separated double values). Type null for all.
	 * @param offset          Number of first lines to omit. Type null or 0 for all.
	 * @param firstIdx        First index of loaded values in a row - starting from
	 *                        0 (first ones can be omitted). Type null or 0 for all.
	 * @param lastIdx         Last index of loaded values in a row (higher indexes
	 *                        will be omitted). Type null for all.
	 * @return true on success, false on error
	 */
	public boolean loadTextFile(String file, Integer numberOfVectors, Integer offset, Integer firstIdx,
			Integer lastIdx) {
		if (offset == null)
			offset = 0;

		if (firstIdx == null)
			firstIdx = 0;

		if (lastIdx != null) {
			if (lastIdx < firstIdx) {
				lastIdx = null;
			}
		}

		FileInputStream fis = null;
		Integer counter = 0;
		int lines = 0;
		try {
			fis = new FileInputStream(file);
			BufferedReader br = new BufferedReader(new InputStreamReader(fis));
			String strLine;
			while ((strLine = br.readLine()) != null && counter != numberOfVectors) {
				lines++;
				if (lines > offset) {
					DataVector vector = new DataVector();
					vector.parse(strLine, sep_regexp, decimal_separator, firstIdx, lastIdx, ignore_parse_errors);
					if (!vector.data.isEmpty()) {
						counter++;
						add(vector);
					}
				}
			}
		} catch (IOException e) {
			System.out.println("error during accessing/reading a file: " + file);
			return false;
		} finally {
			if (fis != null) {
				try {
					fis.close();
				} catch (IOException e) {
					System.out.println("error closing a file: " + file);
					return false;
				}
			}
		}
		return true;
	}

	public double getAverage() {
		double sum = 0.0;

		for (DataVector v : data) {
			sum += v.getAverage();
		}

		return sum / data.size();
	}

	/**
	 * @return text info about minimum and maximum values stored in loaded set
	 */
	public String toStringMinsMaxs() {
		String output = "";

		output += "min: ";
		boolean first = true;
		for (Double d : mins) {
			if (first) {
				first = false;
			} else {
				output += ",\t";
			}
			output += d;
		}
		output += "\nmax: ";
		first = true;
		for (Double d : maxs) {
			if (first) {
				first = false;
			} else {
				output += ",\t";
			}
			output += d;
		}

		return output;
	}

	@Override
	public String toString() {
		String output = "";

		for (DataVector v : data) {
			output += v + "\n";
		}

		output += "Number of rows: " + data.size();
		output += "\n" + toStringMinsMaxs();

		return output;
	}

}