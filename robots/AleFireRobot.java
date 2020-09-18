package ubc.ecee.cpen502.robots;

import java.awt.Color;
import java.awt.event.KeyEvent;
import java.util.Random;

import robocode.AdvancedRobot;
import robocode.BulletHitEvent;
import robocode.DeathEvent;
import robocode.HitByBulletEvent;
import robocode.HitRobotEvent;
import robocode.HitWallEvent;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;

public class AleFireRobot extends AdvancedRobot {
	private static final int numDimensions = 4;
	
	private static final int numRounds = 5000;
	private static final long startTime = System.currentTimeMillis();
	
	private static final int search = 0;
	private static final int action = 1;
	private int operationalMode = 0;
	
	private boolean onPolicy = false;
	private double epsilon = 0.2;
	private double alfa = 0.9;
	private double gamma = 0.99;
	private double currentReward;
	private static int nRounds = 1;
	private static int nWins = 0;
	private enumActions lastAction;
	private enumActions currentAction;
	private double[] lastState;
	private ScannedRobotEvent scannedRobotEvent;
	private static CommonInterface_lut mind = new LUT4_fire(enumDistance.values().length, enumEnergy.values().length, enumEnergy.values().length, enumActions.values().length);
	private LogFile logLearningRate;
	//private LogFile logLUT1;
	private LogFile logLUTlast;
	private Random r;
	
	public enum enumDistance {level1, level2, level3, level4, level5, level6, level7, level8, level9, level10}; 
	public enum enumEnergy {verylow, low, medium, high, veryhigh};
	public enum enumActions {advance, distance, fire1, fire2, fire3, goup, godown, goright, goleft};
	//actions go up, go down, do right and go left are relative to the enemy position
	
	private int distance, enemy_energy;
	
	
	public void run() {
		
		setColors(Color.BLACK, Color.WHITE, Color.RED);
		
		logLearningRate = new LogFile("performance"+startTime+".txt", this);
		//logLUT1 = new LogFile("LUT1_startTime"+startTime+".txt", this);
		logLUTlast = new LogFile("LUT"+numRounds+"_"+startTime+".txt", this);
		if(this.getRoundNum() == 0) {
			logLearningRate.reportLog("Epsilon: "+epsilon+", alfa: "+alfa+", gamma: "+gamma+", on policy: "+onPolicy);
			//logLUT1.reportLog(mind.toString());
		}
		if(this.getRoundNum() == (numRounds - 1)) {
			logLUTlast.reportLog(mind.toString());
		}
		
		r = new Random(System.currentTimeMillis());
		lastAction = null;
		currentAction = null;
		lastState = null;
		operationalMode = search;
		currentReward = 0.0;
		
		//alfa = 1/(this.getRoundNum()+1);
		/*if(this.getRoundNum()>= 2500) {
			epsilon = 0;
		}*/
		
		while(true) {
			switch(operationalMode){
			case search:
				currentReward = 0;
				turnRadarRight(360);
				break;
			case action:	
				//choose action from actual state using policy derived from Q
				if(r.nextDouble() < epsilon)
					currentAction = randomAction();
				else
					currentAction = bestAction(distance, enemy_energy, enumEnergyOf(getEnergy()).ordinal());
				
				double[] currentState = getState(currentAction);
				
				//update Q value for the (lastState, lastAction)
				if(lastAction != null && lastState != null) {
					mind.train(lastState, computeQ(lastState, currentState, currentReward));
				}
				lastAction = currentAction;
				lastState = new double[currentState.length];
				for(int i=0; i<currentState.length; i++)
					lastState[i] = currentState[i];
				
				//take action
				applyAction(currentAction);
				
				operationalMode = search;
				
				break;
			}
		}
	}
	
	
	public void onScannedRobot(ScannedRobotEvent e) {
		operationalMode = action;
		scannedRobotEvent = e;
		distance = enumDistanceOf(e.getDistance()).ordinal();
		enemy_energy = enumEnergyOf(e.getEnergy()).ordinal();
	}
	
	
	public void onBulletHit(BulletHitEvent event) {
		operationalMode = action;
		//positive reward
		currentReward = +0.8;
		enemy_energy = enumEnergyOf(event.getEnergy()).ordinal();
	}
	
	public void onHitByBullet(HitByBulletEvent event) {
		operationalMode = action;
		//negative reward
		currentReward = -0.6;
	}
	
	public void onHitRobot(HitRobotEvent event) {
		operationalMode = action;
		//negative reward
		currentReward = -0.1;
	}
	
	public void onHitWall(HitWallEvent event) {
		operationalMode = action;
		//negative reward
		currentReward = -0.1;
	}
	
	public void onKeyPressed(KeyEvent e) {
		//change epsilon
		if(e.getKeyCode() == KeyEvent.VK_E) {
			epsilon = 0.3;
			setColors(Color.GREEN, Color.WHITE, Color.RED);
		}
		//default: on policy, pressing O: off-policy
		if(e.getKeyCode() == KeyEvent.VK_O) {
			onPolicy = false;
		}
	}
	
	public void onDeath(DeathEvent event) {
		double[] state = getState(lastAction);
		//reward -1
		currentReward = -1;
		
		//update Q value for the (lastState, lastAction)
		mind.train(lastState, computeQ(lastState, state, currentReward));
		
		if(nRounds < 100) {
			nRounds++;
		}else {
			nRounds = 1;
			logLearningRate.reportLog(nWins/100.0+"");
			nWins = 0;
		}
	}
	
	public void onWin(WinEvent event) {
		double[] state = getState(lastAction);
		//reward +1
		currentReward = +1;
		
		//update Q value for the (lastState, lastAction)
		mind.train(lastState, computeQ(lastState, state, currentReward));
		
		nWins++;
		if(nRounds < 100) {
			nRounds++;
		}else {
			nRounds = 1;
			logLearningRate.reportLog(nWins/100.0+"");
			nWins = 0;
		}	
	}
	
	public double computeQ(double[] actualState, double[] futureState, double currentReward) {
		double newQ = 0.0;
		double actualQ = mind.outputFor(actualState);
		
		double Qvalue_futureState;
		
		if(onPolicy) {
			Qvalue_futureState = mind.outputFor(futureState);
		}else {
			//update the Q value using the best action
			enumActions bestAction = bestAction((int)futureState[0], (int)futureState[1], (int)futureState[2]);
			futureState[3] = bestAction.ordinal();
			Qvalue_futureState = mind.outputFor(futureState);
		}
		
		newQ = actualQ + alfa*(currentReward+gamma*Qvalue_futureState-actualQ);
		
		return newQ;
	}
	
	public enumActions bestAction(int distance, int enemy_energy, int my_energy) {
		//initialize best action
		enumActions bestAction = enumActions.values()[0];
		double bestQ = 0;
		double[] X = new double[numDimensions];
		X[0] = distance;
		X[1] = enemy_energy;
		X[2] = my_energy;
		
		for(int i=0; i<enumActions.values().length; i++) {
			X[3] = enumActions.values()[i].ordinal();
			if(mind.outputFor(X) > bestQ) {
				bestQ = mind.outputFor(X);
				bestAction = enumActions.values()[i];
			}
		}
		
		return bestAction;
	}
	
	public enumActions randomAction() {
		Random r = new Random(System.currentTimeMillis());
		int randomIndex = r.nextInt(enumActions.values().length);
		return enumActions.values()[randomIndex];
	}
	
	
	public enumDistance enumDistanceOf(double actualDistance) {
		int level = (int)Math.floor(actualDistance / 100);
		if(level > enumDistance.level10.ordinal())
			level = enumDistance.level10.ordinal();
		return enumDistance.values()[level];
	}
	
	public enumEnergy enumEnergyOf(double actualEnergy) {
		int level = (int)Math.floor(actualEnergy / 20);
		if(level > enumEnergy.veryhigh.ordinal())
			level = enumEnergy.veryhigh.ordinal();
		return enumEnergy.values()[level];
	}
	
	//up, down, left, right are defined considering the enemy as point of reference
	
	public void applyAction(enumActions action) {	
		
		switch(action) {
		case advance:
			turnRight(scannedRobotEvent.getBearing());
			ahead(scannedRobotEvent.getDistance()-10);
			break;
		case distance:
			turnRight(getOppositeAngle(scannedRobotEvent.getBearing()));
			ahead(scannedRobotEvent.getDistance()-10);
			break;
		case fire1:
			double absoluteBearing = getHeading() + scannedRobotEvent.getBearing();
			double bearingFromGun = normaliseAngle(absoluteBearing-getGunHeading()) ;
			turnGunRight(bearingFromGun);
			fire(1);
			break;
		case fire2:
			absoluteBearing = getHeading() + scannedRobotEvent.getBearing();
			bearingFromGun = normaliseAngle(absoluteBearing-getGunHeading()) ;
			turnGunRight(bearingFromGun);
			fire(2);
			break;
		case fire3:
			absoluteBearing = getHeading() + scannedRobotEvent.getBearing();
			bearingFromGun = normaliseAngle(absoluteBearing-getGunHeading()) ;
			turnGunRight(bearingFromGun);
			fire(3);
			break;
		case goup:
			turnRight(scannedRobotEvent.getBearing());
			ahead(100);
			break;
		case godown:
			turnRight(scannedRobotEvent.getBearing()+180);
			ahead(100);
			break;
		case goright:
			turnRight(scannedRobotEvent.getBearing()+90);
			ahead(100);
			break;
		case goleft:
			turnRight(scannedRobotEvent.getBearing()-90);
			ahead(100);
			break;
		}
	}

	
	private double normaliseAngle(double angle) {
		if(angle < 0 && angle > 180) 
			return 360 + angle;
		else
			return angle;
	}
	
	private double getOppositeAngle(double angle) {
		if(angle >= 0) 
			return -180+angle;
		else
			return 180+angle;
	}

	
	private double[] getState(enumActions currentAction) {
		int my_energy;
		
		my_energy = enumEnergyOf(getEnergy()).ordinal();
		//create the actual state
		double[] state = new double[numDimensions];
		//distance
		state[0] = distance;
		//enemy energy
		state[1] = enemy_energy;
		//my energy
		state[2] = my_energy;
		//actions
		state[3] = currentAction.ordinal();
		
		return state;
	}

}
