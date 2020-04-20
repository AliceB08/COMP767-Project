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
num_episodes = 100000

index = 0
trajectories = {}
for e in range(num_episodes):
    maze.startposition()
    maze.t = 0
    ego_vels, target_poses, target_hds = [], [], []
    i = 0
    # run forward for one trial (using random actions for sake of illustration)
    while (not maze.timeup() and not maze.atgoal()):
        # select a random action - this is what your actor critic network needs to provide
        A = np.random.randint(0, 8)
        # move the rat
        traj = maze.move(A)
        ego_vels.append(traj["ego_vel"])
        target_poses.append(traj["target_pos"])
        target_hds.append(traj["target_hd"])
        i+=1
    if maze.atgoal():
        final_traj = {}
        final_traj['init_pos'] = traj['init_pos']
        final_traj["init_hd"] = traj["init_hd"]
        if i>=20:
            final_traj["ego_vel"] = ego_vels[-20:]
            final_traj["target_pos"] = target_poses[-20:]
            final_traj["target_hd"] = target_hds[-20:]
            trajectories[index] = final_traj
            index+=1



print(len(trajectories.keys()))
filename = "./data/watermaze_data.pkl"
f = open(filename, "wb")
pickle.dump(trajectories, f)
f.close()


