<img src=Robocode-logo.png width="40%">

# Learning agent for Robocode
This repository contains three versions of a Robocode tank that learns how to defeat its enemy. In particular, the agent implements a Reinforcement Learning algorithm, the Q-learning algorithm. The difference among the Robocode tanks involves only the RL algorithm implementation; there is a tabular Q-learning version, an agent which uses a Neural Network for function approximation and the third one adds Experience Replay. The project has been developed for the "CPEN 502 - Architectures for Learning Systems" course (Fall 2019), University of British Columbia. 

All the information about Robocode can be found [here](https://robocode.sourceforge.io/).

The agents were trained and tested against the “Fire” robot, which is one of the standard tanks of Robocode. In this setting, the agents reach a satisfying win rate, especially the tank that uses Q-learning and Experience Replay. 

The tanks can be selected directly from the Robocode GUI following the coming steps:
- Create the JAR archive of the selected agent
- Copy the JAR file in the robocode/robots/ folder

Further details on this and on Robocode in general can be found on the [Wiki page](https://robowiki.net/wiki/Robocode)

