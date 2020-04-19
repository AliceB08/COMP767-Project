import numpy as np
import random
import matplotlib.pyplot as plt
import math
from watermaze import watermaze, RWMTask, DMPTask
import pickle

# demo of how to use the watermaze module
# create the watermaze object
maze = watermaze()
# set the starting location
maze.startposition()
num_episodes = 10000
trajectories = {}
for e in range(num_episodes):
    index = 0
    maze.startposition()
    maze.t = 0
    # run forward for one trial (using random actions for sake of illustration)
    while (not maze.timeup() and not maze.atgoal()):
        # select a random action - this is what your actor critic network needs to provide
        A = np.random.randint(0, 8)
        # move the rat
        maze.move(A)
    # print out why the trial ended (note, if the rat reached the goal, then you must deliver a reward)
    if maze.atgoal():
        traj = maze.get_trajectory()
        trajectories[index] = traj
        index+=1
    else:
        print("No more time for you dumb-dumb...")

filename = "./data/watermaze_data.pkl"
f = open(filename, "wb")
pickle.dump(trajectories, f)
f.close()


