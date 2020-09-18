package ubc.ecee.cpen502.robots;

import java.io.IOException;
import java.io.PrintStream;

import robocode.AdvancedRobot;
import robocode.RobocodeFileOutputStream;

public class LogFile {
	
	private String filename;
	private AdvancedRobot robot;

	public LogFile(String filename, AdvancedRobot robot) {
		this.filename = filename;
		this.robot = robot;
	}
	
	public void reportLog(String logSentence) {
		PrintStream w = null;
		try {
			w = new PrintStream(new RobocodeFileOutputStream(robot.getDataFile(filename).getPath(),true));
			w.println(logSentence);
			if (w.checkError()) {
				System.out.println("I could not write the log!");
			}
			w.close();
		} catch (IOException e) {
			System.out.println("Log file not writable");
		}
	}

}
