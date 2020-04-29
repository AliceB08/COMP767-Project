import numpy as np
import random
import matplotlib.pyplot as plt
import math
from watermaze import watermaze, RMWTask, DMPTask
import pickle
from tqdm import tqdm

# demo of how to use the watermaze module
# create the watermaze object
maze = watermaze()
# set the starting location
maze.startposition()
num_episodes = 500

index = 0
trajectories = {}
for e in tqdm(range(num_episodes)):
    maze.startposition()
    maze.t = 0
    ego_vels, target_poses, target_hds = [], [], []
    final_traj = {}
    # run forward for one trial (using random actions for sake of illustration)
    while not maze.timeup():
        # select a random action - this is what your actor critic network needs to provide
        A = np.random.randint(0, 8)
        # move the rat
        traj = maze.move(A)
        ego_vels.append(traj['ego_vel'])
        target_poses.append(traj['target_pos'])
        target_hds.append(traj['target_hd'])
        if maze.t==1:
            # first step of trajectory taken
            final_traj['init_pos'] = traj['prev_pos'].copy()
            final_traj['init_hd'] = traj['prev_hd'].copy()
    final_traj['ego_vel'] = np.array(ego_vels)
    final_traj['target_pos'] = np.array(target_poses)
    final_traj['target_hd'] = np.array(target_hds)
    trajectories[index] = final_traj
    index+=1

print(len(trajectories.keys()))
# filename = "./data/tmp_watermaze_data.pkl"
filename = "new_watermaze_data_2.pkl"
f = open(filename, "wb")
pickle.dump(trajectories, f)
f.close()
