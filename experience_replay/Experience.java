package ubc.ecee.cpen502.experience_replay;

public class Experience {

	private double[] actualState;
	private double futureReward;
	private double[] futureState;
	public Experience(double[] actualState, double futureReward, double[] futureState) {
		this.actualState = new double[actualState.length];
		for(int i=0; i<actualState.length; i++)
			this.actualState[i] = actualState[i];
		
		this.futureState =  new double[futureState.length];
		for(int i=0; i<futureState.length; i++)
			this.futureState[i] = futureState[i];
		
		this.futureReward = futureReward;
		
	}
	public double[] getActualState() {
		return actualState;
	}
	public double getFutureReward() {
		return futureReward;
	}
	public double[] getFutureState() {
		return futureState;
	}
	
	
	
	
}
