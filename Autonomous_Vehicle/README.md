# Autonomous_Vehicle_inPython

The Ai.py file contains the Deep-Q reinforcement learning based brain which is integrated with the map.py.
1. Graphics is created using Kivy modules in Python. A car with 3 sensors in front used in the simulation.
2. Sand used for its simulation so that the car can learn through experiences stored in a batch of 100.
3. Reward = -1 given to the taxi when it crashes into the sand or reaches outskirts of the city, Reward = -0.2 given when it moves further away from the destination, Reward = +0.1 given when it approaches in correct direction of the destination, Reward =+1 given when it reaches the goal.

