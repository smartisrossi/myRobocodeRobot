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
import ubc.ecee.cpen502.NeuralNetwork.CommonInterface;
import ubc.ecee.cpen502.NeuralNetwork.NeuralNet;
import ubc.ecee.cpen502.experience_replay.CircularStateQueue;
import ubc.ecee.cpen502.experience_replay.Experience;

public class AleFireRobotNN_exp extends AdvancedRobot {
	private static final int numDimensions = 4;
	
	private static final int numRounds = 5000;
	private static final long startTime = System.currentTimeMillis();
	
	private static final int search = 0;
	private static final int action = 1;
	private int operationalMode = 0;
	
	private static final double maxDistance = 1000;
	private static final double maxEnergy = 120;
	private static final double maxQ = 9.580329123707422;
	
	private static final int distanceTarget = 99;
	private static final int enemyEnergyTarget = 99;
	private static final int myEnergyTarget = 79;
	private static final int actionTarget = 3;
	private double newQ;
	private double actualQ;
	
	private CircularStateQueue experiences = new CircularStateQueue(15);
	
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
	private static int NNinputDimensions = numDimensions - 1 + enumActions.values().length;;
	private static CommonInterface mind = new NeuralNet(NNinputDimensions, 13, 0.03, 0.9, -1, 1);
	private LogFile logLearningRate;
	private LogFile logWeightlast;
	private LogFile logWeightfirst;
	private LogFile logQdifference;
	private Random r;
	
	public enum enumActions {advance, distance, fire1, fire2, fire3, goup, godown, goright, goleft};
	//actions go up, go down, do right and go left are relative to the enemy position
	
	private double distance, enemy_energy;
	
	
	public void run() {
		
		setColors(Color.BLACK, Color.WHITE, Color.RED);
		
		logLearningRate = new LogFile("performance"+startTime+".txt", this);
		logQdifference = new LogFile("Qdifference"+startTime+".txt", this);
		logWeightfirst = new LogFile("initial_weights_"+startTime+".txt", this);
		logWeightlast = new LogFile("weight"+numRounds+"_"+startTime+".txt", this);
		if(this.getRoundNum() == 0) {
			logLearningRate.reportLog("Epsilon: "+epsilon+", alfa: "+alfa+", gamma: "+gamma+", on policy: "+onPolicy);
			logWeightfirst.reportLog(mind.toString());
		}
		if(this.getRoundNum() == (numRounds - 1)) {
			logWeightlast.reportLog(mind.toString());
		}
		
		r = new Random(System.currentTimeMillis());
		lastAction = null;
		currentAction = null;
		lastState = null;
		operationalMode = search;
		currentReward = 0.0;
		
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
					currentAction = bestAction(distance, enemy_energy, getEnergy());
				
				double[] currentState = getState(currentAction);
				
				//update Q value for the (lastState, lastAction)
				if(lastAction != null && lastState != null) {
					Experience exp = new Experience(lastState, currentReward, currentState);
					experiences.add(exp);
					double[][] trainVector = new double[experiences.size()][NNinputDimensions];
					double[] labelVector = new double[experiences.size()];
					double truePosition;
					
					//normalize input values before training the neural network
					for(int j=0; j<experiences.size(); j++) {
						//distance
						trainVector[j][0] = experiences.get(j).getActualState()[0]/maxDistance;
						//enemy energy
						trainVector[j][1] =  experiences.get(j).getActualState()[1]/maxEnergy;
						//my energy
						trainVector[j][2] =  experiences.get(j).getActualState()[2]/maxEnergy;
						//actions
						truePosition =  experiences.get(j).getActualState()[3];
						for(int i = 0; i<enumActions.values().length; i++) {
							if(i == truePosition)
								trainVector[j][i+3] = +1;
							else
								trainVector[j][i+3] = -1;
						}
					
						
						labelVector[j] = computeQ(experiences.get(j).getActualState(), experiences.get(j).getFutureState(), experiences.get(j).getFutureReward())/maxQ;	
					}
					if(stateEqualsTarget(lastState)) {
						double Qdifference = actualQ-newQ;
						logQdifference.reportLog(""+Qdifference);
					}
					mind.train(trainVector, labelVector);
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
		distance = e.getDistance();
		enemy_energy = e.getEnergy();
	}
	
	
	public void onBulletHit(BulletHitEvent event) {
		operationalMode = action;
		//positive reward
		currentReward = +0.8;
		enemy_energy = event.getEnergy();
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
		Experience exp = new Experience(lastState, currentReward, state);
		experiences.add(exp);
		double[][] trainVector = new double[experiences.size()][NNinputDimensions];
		double[] labelVector = new double[experiences.size()];
		double truePosition;
		
		//normalize input values before training the neural network
		for(int j=0; j<experiences.size(); j++) {
			//distance
			trainVector[j][0] = experiences.get(j).getActualState()[0]/maxDistance;
			//enemy energy
			trainVector[j][1] =  experiences.get(j).getActualState()[1]/maxEnergy;
			//my energy
			trainVector[j][2] =  experiences.get(j).getActualState()[2]/maxEnergy;
			//actions
			truePosition =  experiences.get(j).getActualState()[3];
			for(int i = 0; i<enumActions.values().length; i++) {
				if(i == truePosition)
					trainVector[j][i+3] = +1;
				else
					trainVector[j][i+3] = -1;
			}
		
			
			labelVector[j] = computeQ(experiences.get(j).getActualState(), experiences.get(j).getFutureState(), experiences.get(j).getFutureReward())/maxQ;	
		}
		if(stateEqualsTarget(lastState)) {
			double Qdifference = actualQ-newQ;
			logQdifference.reportLog(""+Qdifference);
		}
		mind.train(trainVector, labelVector);
		
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
		Experience exp = new Experience(lastState, currentReward, state);
		experiences.add(exp);
		double[][] trainVector = new double[experiences.size()][NNinputDimensions];
		double[] labelVector = new double[experiences.size()];
		double truePosition;
		
		//normalize input values before training the neural network
		for(int j=0; j<experiences.size(); j++) {
			//distance
			trainVector[j][0] = experiences.get(j).getActualState()[0]/maxDistance;
			//enemy energy
			trainVector[j][1] =  experiences.get(j).getActualState()[1]/maxEnergy;
			//my energy
			trainVector[j][2] =  experiences.get(j).getActualState()[2]/maxEnergy;
			//actions
			truePosition =  experiences.get(j).getActualState()[3];
			for(int i = 0; i<enumActions.values().length; i++) {
				if(i == truePosition)
					trainVector[j][i+3] = +1;
				else
					trainVector[j][i+3] = -1;
			}
		
			
			labelVector[j] = computeQ(experiences.get(j).getActualState(), experiences.get(j).getFutureState(), experiences.get(j).getFutureReward())/maxQ;	
		}
		if(stateEqualsTarget(lastState)) {
			double Qdifference = actualQ-newQ;
			logQdifference.reportLog(""+Qdifference);
		}
		mind.train(trainVector, labelVector);
		
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
		newQ = 0.0;
		double[] inputVector = new double[NNinputDimensions];
		double truePosition;
		
		//normalize input values before using the neural network
		//distance
		inputVector[0] = actualState[0]/maxDistance;
		//enemy energy
		inputVector[1] = actualState[1]/maxEnergy;
		//my energy
		inputVector[2] = actualState[2]/maxEnergy;
		//actions
		truePosition = actualState[3];
		for(int i = 0; i<enumActions.values().length; i++) {
			if(i == truePosition)
				inputVector[i+3] = +1;
			else
				inputVector[i+3] = -1;
		}
		
		actualQ = mind.outputFor(inputVector);
		
		double Qvalue_futureState;
		
		if(onPolicy) {
			//normalize input values before using the neural network
			//distance
			inputVector[0] = futureState[0]/maxDistance;
			//enemy energy
			inputVector[1] = futureState[1]/maxEnergy;
			//my energy
			inputVector[2] = futureState[2]/maxEnergy;
			//actions
			truePosition = futureState[3];
			for(int i = 0; i<enumActions.values().length; i++) {
				if(i == truePosition)
					inputVector[i+3] = +1;
				else
					inputVector[i+3] = -1;
			}
			Qvalue_futureState = mind.outputFor(inputVector);
		}else {
			//update the Q value using the best action
			enumActions bestAction = bestAction((int)futureState[0], (int)futureState[1], (int)futureState[2]);
			futureState[3] = bestAction.ordinal();
			
			//normalize input values before using the neural network
			//distance
			inputVector[0] = futureState[0]/maxDistance;
			//enemy energy
			inputVector[1] = futureState[1]/maxEnergy;
			//my energy
			inputVector[2] = futureState[2]/maxEnergy;
			//actions
			truePosition = futureState[3];
			for(int i = 0; i<enumActions.values().length; i++) {
				if(i == truePosition)
					inputVector[i+3] = +1;
				else
					inputVector[i+3] = -1;
			}
			Qvalue_futureState = mind.outputFor(inputVector);
		}
		
		newQ = actualQ + alfa*(currentReward+gamma*Qvalue_futureState-actualQ);
		
		return newQ;
	}
	
	public enumActions bestAction(double distance, double enemy_energy, double my_energy) {
		//initialize best action
		enumActions bestAction = enumActions.values()[0];
		double bestQ = 0, truePosition, NNoutput;
		double[] X = new double[numDimensions];
		double[] inputVector = new double[NNinputDimensions];
		X[0] = distance;
		X[1] = enemy_energy;
		X[2] = my_energy;
		
		for(int i=0; i<enumActions.values().length; i++) {
			X[3] = enumActions.values()[i].ordinal();
			
			//normalize input values before using the neural network
			//distance
			inputVector[0] = X[0]/maxDistance;
			//enemy energy
			inputVector[1] = X[1]/maxEnergy;
			//my energy
			inputVector[2] = X[2]/maxEnergy;
			//actions
			truePosition = X[3];
			for(int j = 0; j<enumActions.values().length; j++) {
				if(j == truePosition)
					inputVector[j+3] = +1;
				else
					inputVector[j+3] = -1;
			}
			NNoutput = mind.outputFor(inputVector);
			if(NNoutput > bestQ) {
				bestQ = NNoutput;
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
		double my_energy;
		
		my_energy = getEnergy();
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

	private boolean stateEqualsTarget(double[] state) {
		int actualDistance = (int)Math.round(state[0]);
		int actualEnemyEnergy = (int)Math.round(state[1]);
		int actualMyEnergy = (int)Math.round(state[2]);
		int actualAction = (int)state[3];
		if(actualDistance <= distanceTarget && 
				actualEnemyEnergy <= enemyEnergyTarget && actualEnemyEnergy>=(enemyEnergyTarget-20) &&
				actualMyEnergy <= myEnergyTarget && actualMyEnergy >= (myEnergyTarget-20) &&
				actualAction == actionTarget)
			return true;
		else
			return false;
	}
}
