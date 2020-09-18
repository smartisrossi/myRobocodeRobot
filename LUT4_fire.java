package ubc.ecee.cpen502.robots;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.NoSuchElementException;
import java.util.Random;
import java.util.StringTokenizer;

import ubc.ecee.cpen502.robots.AleFireRobot.enumActions;
import ubc.ecee.cpen502.robots.AleFireRobot.enumDistance;
import ubc.ecee.cpen502.robots.AleFireRobot.enumEnergy;

public class LUT4_fire implements CommonInterface_lut {
	
	private static final int numDimensions = 4;
	private double [][][][] lut;
	private double [][][][] visits;
	private int size;
	
	

	public LUT4_fire(int numDimensionsLevel1, int numDimensionsLevel2, int numDimensionsLevel3, int numDimensionsLevel4) {		
		//Levels: distance, enemy energy, my energy, actions
		lut = new double[numDimensionsLevel1][numDimensionsLevel2][numDimensionsLevel3][numDimensionsLevel4];
		visits = new double[numDimensionsLevel1][numDimensionsLevel2][numDimensionsLevel3][numDimensionsLevel4];
		initialiseLUT();
		size = numDimensionsLevel1*numDimensionsLevel2*numDimensionsLevel3*numDimensionsLevel4;
	}

	@Override
	public double outputFor(double[] X) {
		if(X.length != numDimensions) {
			throw new ArrayIndexOutOfBoundsException();
		} else {
			int distance, enemyEnergy, myEnergy, action;
			distance = (int)X[0];
			enemyEnergy = (int)X[1];
			myEnergy = (int)X[2];
			action = (int)X[3];
		
			return lut[distance][enemyEnergy][myEnergy][action];
		}
	}
	
	public int getNumVisits(double[] X) {
		if(X.length != numDimensions) {
			throw new ArrayIndexOutOfBoundsException();
		} else {
			int distance, enemyEnergy, myEnergy, action;
			distance = (int)X[0];
			enemyEnergy = (int)X[1];
			myEnergy = (int)X[2];
			action = (int)X[3];
		
			return (int)visits[distance][enemyEnergy][myEnergy][action];
		}
	}

	@Override
	public double train(double[] X, double argValue) {
		if(X.length != numDimensions) {
			throw new ArrayIndexOutOfBoundsException();
		} else {
			int distance, enemyEnergy, myEnergy, action;
			distance = (int)X[0];
			enemyEnergy = (int)X[1];
			myEnergy = (int)X[2];
			action = (int)X[3];
		
			lut[distance][enemyEnergy][myEnergy][action] = argValue;
			visits[distance][enemyEnergy][myEnergy][action]++;
			return argValue;
		}
	}
	
	public void initialiseLUT() {
		int distance, enemyEnergy, myEnergy, action;
		Random r = new Random(System.currentTimeMillis());
		for(distance = 0; distance<enumDistance.values().length; distance++) {
			for(enemyEnergy = 0; enemyEnergy<enumEnergy.values().length; enemyEnergy++) {
				for(myEnergy = 0; myEnergy<enumEnergy.values().length; myEnergy++) {
					for(action = 0; action< enumActions.values().length; action++) {
						lut[distance][enemyEnergy][myEnergy][action] = r.nextDouble();
						visits[distance][enemyEnergy][myEnergy][action] = 0;
					}
				}
			}
		}
	}

	@Override
	public void save(File argFile) throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(argFile));
		writer.append(this.toString());
		writer.close();
	}

	@Override
	public void load(String argFileName) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(argFileName));
		int distance, enemyEnergy, myEnergy, action;
		double Qvalue, num_visits;
		String line;
		while((line = reader.readLine()) != null && !line.isEmpty()) {
			try {
			StringTokenizer st = new StringTokenizer(line, "{,}= ");
			distance = enumDistance.valueOf(st.nextToken()).ordinal();
			enemyEnergy = enumEnergy.valueOf(st.nextToken()).ordinal();
			myEnergy = enumEnergy.valueOf(st.nextToken()).ordinal();
			action = enumActions.valueOf(st.nextToken()).ordinal();
			Qvalue = Double.valueOf(st.nextToken());
			st.nextToken();
			num_visits = Double.valueOf(st.nextToken());
			lut[distance][enemyEnergy][myEnergy][action] = Qvalue;
			visits[distance][enemyEnergy][myEnergy][action] = num_visits;
			}catch(NoSuchElementException e) {
				System.out.println("problem with:"+line);
			}
		}
		reader.close();
	}
	
	public String toString() {
		int distance, enemyEnergy, myEnergy, action;
		StringBuilder temp = new StringBuilder();
		for(distance = 0; distance<enumDistance.values().length; distance++) {
			for(enemyEnergy = 0; enemyEnergy<enumEnergy.values().length; enemyEnergy++) {
				for(myEnergy = 0; myEnergy<enumEnergy.values().length; myEnergy++) {
					for(action = 0; action< enumActions.values().length; action++) {
						temp.append("{"+enumDistance.values()[distance]+", "+
								enumEnergy.values()[enemyEnergy]+", "+ 
								enumEnergy.values()[myEnergy]+", "+
								enumActions.values()[action]+"} = "+
								lut[distance][enemyEnergy][myEnergy][action]+", visits: "+
								visits[distance][enemyEnergy][myEnergy][action] +"\n");
					}
				}
			}
		}
		return temp.toString();
	}

	public double[][][][] getLut() {
		return lut;
	}

	public int getSizeLut() {
		return size;
	}

	public static int getNumdimensions() {
		return numDimensions;
	}

}
