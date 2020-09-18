package ubc.ecee.cpen502.NeuralNetwork;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Random;

public class NeuralNet implements NeuralNetInterface, Serializable  {

	private static final long serialVersionUID = 1L;
	private int numInputs;
	private int numHidden;
	private double learningRate;
	private double momentumTerm;
	private double argA;
	private double argB;
	
	private double [][] inputToHiddenWeights;
	private double [][] inputToHiddenLastWeightChange;
	private double [] hiddenToOutputWeights;
	private double [] hiddenToOutputLastWeightChange;
	
	
	public NeuralNet(int numInputs, int numHidden, double learningRate, double momentumTerm, double argA, double argB) {
		this.numInputs = numInputs;
		this.numHidden = numHidden;
		this.learningRate = learningRate;
		this.momentumTerm = momentumTerm;
		this.argA = argA;
		this.argB = argB;
		
		// 1 row for each input + 1 for bias; 1 column for each hidden neuron
		inputToHiddenWeights = new double[numInputs + 1][numHidden]; //+1 is to accomodate a bias weight
		inputToHiddenLastWeightChange = new double[numInputs + 1][numHidden];
		hiddenToOutputWeights = new double[numHidden + 1];
		hiddenToOutputLastWeightChange = new double[numHidden + 1];
		
		//initialize neural network
		this.initializeWeights(0.5, -0.5);
	}
	
	public double[][] getInputToHiddenWeights() {
		return inputToHiddenWeights;
	}



	public double[][] getInputToHiddenLastWeightChange() {
		return inputToHiddenLastWeightChange;
	}



	public double[] getHiddenToOutputWeights() {
		return hiddenToOutputWeights;
	}



	public double[] getHiddenToOutputLastWeightChange() {
		return hiddenToOutputLastWeightChange;
	}


	public void setInputToHiddenWeights(double[][] inputToHiddenWeights) {
		this.inputToHiddenWeights = inputToHiddenWeights;
	}
	

	public void setHiddenToOutputWeights(double[] hiddenToOutputWeights) {
		this.hiddenToOutputWeights = hiddenToOutputWeights;
	}
	

	@Override
	public double outputFor(double[] X) {
		int j;
		double[] weights;
		double [] hiddenInputs = new double[numHidden];
		double [] hiddenOutputs = new double[numHidden];
		double lastLayerInput; 
		double lastLayerOutput;
		
		//compute the input and output value for each hidden neuron
		for(j=0; j<numHidden; j++) {
			weights = getColumn(inputToHiddenWeights, j);
			hiddenInputs[j] = weights[numInputs]+summation(numInputs, X, weights); //the bias term is the last term
			hiddenOutputs[j] = this.customSigmoid(hiddenInputs[j]);
		}
		//compute the input and output value for each output unit (only one in this case)
		lastLayerInput = hiddenToOutputWeights[numHidden] + summation(numHidden, hiddenOutputs, hiddenToOutputWeights);
		lastLayerOutput = this.customSigmoid(lastLayerInput);
		
		return lastLayerOutput;
	}

	//return The error in the output for that input vector
	@Override
	public double train(double[][] X, double[] argValue) {
		int i,j;
		double totalError = Double.MAX_VALUE;
		double temp = 0;
		double[] weights;
		
		double [] hiddenInputs = new double[numHidden];
		double [] hiddenOutputs = new double[numHidden];
		double lastLayerInput; // In the XOR problem there is only one output
		double lastLayerOutput = Double.MAX_VALUE;
		double outputErrorSignal;
		double [] hiddenErrorSignal = new double[numHidden];

		
		for(i=0; i<X.length; i++) {
			//Forward step
			//compute the input and output value for each hidden neuron
			for(j=0; j<numHidden; j++) {
				weights = getColumn(inputToHiddenWeights, j);
				hiddenInputs[j] = weights[numInputs]+summation(numInputs, X[i], weights); //the bias term is the last term
				hiddenOutputs[j] = this.customSigmoid(hiddenInputs[j]);
			}
			//compute the input and output value for each output unit (only one in this case)
			lastLayerInput = hiddenToOutputWeights[numHidden] + summation(numHidden, hiddenOutputs, hiddenToOutputWeights);
			lastLayerOutput = this.customSigmoid(lastLayerInput);
				
			//backpropagation	
			outputErrorSignal = this.customSigmoidDerivative(lastLayerInput)*(argValue[i]-lastLayerOutput);
			this.updateHiddenToOutputWeights(hiddenOutputs, outputErrorSignal); //update weights below
			
			for(j=0; j<numHidden; j++) { //compute error signal for each hidden unit
				hiddenErrorSignal[j] = this.customSigmoidDerivative(hiddenInputs[j])*outputErrorSignal*hiddenToOutputWeights[j];
			}
			this.updateInputToHiddenWeights(X[i], hiddenErrorSignal); //update weights below
			
		}
		//end of 1 epoch, compute the total error
		temp = 0;
		for(i=0; i<X.length; i++) {
			temp += Math.pow(this.outputFor(X[i]) - argValue[i], 2);
		}
		totalError = temp/2;
		
		return totalError;
	}
	
	public void updateHiddenToOutputWeights(double[] hiddenOutputs, double outputErrorSignal) {
		int j;
		double[] weightCorrectionHiddenToOutput = new double[numHidden+1];
		
		//compute weight correction for hidden to output weights
		for(j=0; j<numHidden; j++) {
			weightCorrectionHiddenToOutput[j] = learningRate*outputErrorSignal*hiddenOutputs[j];
		}
		weightCorrectionHiddenToOutput[numHidden] = learningRate*outputErrorSignal;
		
		
		//compute new hidden to output weights and update the last hidden to output weight change
		for(j=0; j<(numHidden+1); j++) {
			hiddenToOutputWeights[j] += momentumTerm*hiddenToOutputLastWeightChange[j] + weightCorrectionHiddenToOutput[j];
			hiddenToOutputLastWeightChange[j] = weightCorrectionHiddenToOutput[j];
		}
		
	}
	
	public void updateInputToHiddenWeights(double[] input, double[] hiddenErrorSignal) {
		int i,j;
		double[][] weightCorrectionInputToHidden = new double[numInputs+1][numHidden];
			
		//compute weight correction for input to hidden weights
		for(j=0; j<numHidden; j++) {
			for(i=0; i<numInputs; i++) {
				weightCorrectionInputToHidden[i][j] = learningRate*hiddenErrorSignal[j]*input[i];
			}
			weightCorrectionInputToHidden[numInputs][j] = learningRate*hiddenErrorSignal[j];
		}
			
		//compute new input to hidden weights and update the last input to hidden weight change
		for(j=0; j<numHidden; j++) {
			for(i=0; i<(numInputs+1); i++) {
				inputToHiddenWeights[i][j] += momentumTerm*inputToHiddenLastWeightChange[i][j] + weightCorrectionInputToHidden[i][j];
				inputToHiddenLastWeightChange[i][j] = weightCorrectionInputToHidden[i][j];
			}
		}
		
	}

	@Override
	public void save(File argFile) throws IOException {
		FileOutputStream fos = new FileOutputStream(argFile);
		ObjectOutputStream oos = new ObjectOutputStream(fos);
		oos.writeObject(this);
		oos.close();
		fos.close();
	}
	
	public void saveReadable(File argFile) throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(argFile));
		writer.append(this.toString());
		writer.close();
	}

	@Override
	public void load(String argFileName) throws IOException, ClassNotFoundException {
		FileInputStream fis = new FileInputStream(new File(argFileName));
		ObjectInputStream ois = new ObjectInputStream(fis);
		NeuralNet n = (NeuralNet)ois.readObject();
		this.setInputToHiddenWeights(n.getInputToHiddenWeights());
		this.setHiddenToOutputWeights(n.getHiddenToOutputWeights());
		ois.close();
		fis.close();
	}

	@Override
	public double sigmoid(double x) {
		double sigmoid_value = 0;
		sigmoid_value = 2 / (1+Math.exp(-x)) - 1;
		return sigmoid_value;
	}

	@Override
	public double customSigmoid(double x) {
		double sigmoid_value = 0;
		sigmoid_value = ((argB-argA) / (1 + Math.exp(-x))) - (-1*argA);
		return sigmoid_value;
	}
	
	public double customSigmoidDerivative(double x) {
		double customSigmoid, result, gamma, mu;
		gamma = argB-argA;
		mu = -1*argA;
		customSigmoid = this.customSigmoid(x);
		result = (1/gamma)*(mu+customSigmoid)*(gamma-mu-customSigmoid);
		return result;
	}

	//this method initializes weights using values between upperBound and lowerBound
	@Override
	public void initializeWeights(double upperBound, double lowerBound) {
		int i,j;
		double range = upperBound - lowerBound;
		
		//initialize a stream of pseudorandom numbers using the current time as seed
		Random rnd = new Random(System.currentTimeMillis()); 
		
		//initialize inputToHiddenWeights
		for(i=0; i<(this.numInputs+1); i++) {
			for(j=0; j<this.numHidden; j++) {
				this.inputToHiddenWeights[i][j] = lowerBound + range*rnd.nextDouble();
			}
		}
		
		//initialize inputToHiddenLastWeightChange
		for(i=0; i<(this.numInputs+1); i++) {
			for(j=0; j<this.numHidden; j++) {
				this.inputToHiddenLastWeightChange[i][j] = lowerBound + range*rnd.nextDouble();
			}
		}
		
		//initialize hiddenToOutputweights
		for(i=0; i<(this.numHidden+1); i++) {
			this.hiddenToOutputWeights[i] = lowerBound + range*rnd.nextDouble();
		}
		
		//initialize hiddenToOutputLastWeightChange
		for(i=0; i<(this.numHidden+1); i++) {
			this.hiddenToOutputLastWeightChange[i] = lowerBound + range*rnd.nextDouble();
		}
	}

	@Override
	public void zeroWeights() {
		int i,j;
		
		//initialize inputToHiddenWeights
		for(i=0; i<(this.numInputs+1); i++) {
			for(j=0; j<this.numHidden; j++) {
				this.inputToHiddenWeights[i][j] = 0;
			}
		}
		
		//initialize inputToHiddenLastWeightChange
		for(i=0; i<(this.numInputs+1); i++) {
			for(j=0; j<this.numHidden; j++) {
				this.inputToHiddenLastWeightChange[i][j] = 0;
			}
		}
		
		//initialize hiddenToOutputweights
		for(i=0; i<(this.numInputs+1); i++) {
			this.hiddenToOutputWeights[i] = 0;
		}
		
		//initialize hiddenToOutputLastWeightChange
		for(i=0; i<(this.numInputs+1); i++) {
			this.hiddenToOutputLastWeightChange[i] = 0;
		}
	}

	
	@Override
	public String toString() {
		int i,j;
		StringBuilder temp = new StringBuilder();
		temp.append("NeuralNet\ninputToHiddenWeights:\n");
		//write inputToHiddenWeights
		for(i=0; i<(this.numInputs+1); i++) {
			for(j=0; j<this.numHidden; j++) {
				temp.append(this.inputToHiddenWeights[i][j]+" ");
			}
			temp.append("\n");
		}
		temp.append("inputToHiddenLastWeightChange:\n");
		//write inputToHiddenLastWeightChange
		for(i=0; i<(this.numInputs+1); i++) {
			for(j=0; j<this.numHidden; j++) {
				temp.append(this.inputToHiddenLastWeightChange[i][j]+" ");
			}
			temp.append("\n");
		}
		temp.append("hiddenToOutputWeights:\n");
		//initialize hiddenToOutputweights
		for(i=0; i<(this.numHidden+1); i++) {
			temp.append(this.hiddenToOutputWeights[i]+" ");
			temp.append("\n");
		}
		temp.append("hiddenToOutputLastWeightChange:\n");
		//initialize hiddenToOutputLastWeightChange
		for(i=0; i<(this.numHidden+1); i++) {
			temp.append(this.hiddenToOutputLastWeightChange[i]+" ");
			temp.append("\n");
		}
		return temp.toString();
	}

	public static double summation(int numIteration, double[] inputs, double[] weights) {
		double result = 0.0;
		for(int i=0; i<numIteration; i++)
			result += inputs[i]*weights[i];
		return result;
	}
	
	private static double[] getColumn(double[][] array, int index){
	    double[] column = new double[array.length]; 
	    for(int i=0; i<column.length; i++){
	       column[i] = array[i][index];
	    }
	    return column;
	}
	
}
