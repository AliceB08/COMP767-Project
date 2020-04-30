import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from tqdm import tqdm
from watermaze import watermaze, RMWTask, DMPTask
import torch
from torch import nn
from torch.utils import data 

from watermaze_dataloader import Dataset
from model_lstm import GridTorch, GridTorch_nonNeg
from utils import *

# Parameters
ENV_SIZE = 2.2
RADIUS= 60
BATCH_SIZE = 2 #1024
SEED = 9999
N_PC = [256]
PC_SCALE = [10] # [0.01] [10]
N_HDC = [8]
data_params = {"batch_size": BATCH_SIZE, "shuffle": False, "num_workers": 1}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pattern = 'orthogonal' 	# 'hexagonal'
model_path = 'exp2/model_epoch_4799.pt'
target_ensemble_path = 'exp2/target_ensembles.pt'

def plot_trajectories(env,gt_trajectory,gt_hd=None,init_pos=None,init_hd=None,pred_trajectory=None,pred_hd=None):
	len_trajectory = gt_trajectory.shape[0]
	# create the figure 
	fig = plt.figure()
	ax = fig.gca()

	# create the pool perimeter
	pool_perimeter = plt.Circle((0, 0), env.radius, fill=False, color='b', ls='-')
	ax.add_artist(pool_perimeter)

	# create theplatform
	platform = plt.Circle(env.platform_location, env.platform_radius, fill=False, color='r', ls='-')
	ax.add_artist(platform)

	# plot the path
	plt.plot(gt_trajectory[0:len_trajectory,0],gt_trajectory[0:len_trajectory,1], color='k', ls='-',label='Ground truth trajectory')
	# plot the head directions
	if gt_hd is not None:
		plt.quiver(gt_trajectory[0:len_trajectory,0],gt_trajectory[0:len_trajectory,1],2*np.cos(gt_hd),2*np.sin(gt_hd),units='inches',scale=25)
	if pred_trajectory is not None:
		plt.plot(pred_trajectory[0:len_trajectory,0],pred_trajectory[0:len_trajectory,1], color='magenta', ls='-',label='Predicted trajectory')
		if pred_hd is not None:
			plt.quiver(pred_trajectory[0:len_trajectory,0],pred_trajectory[0:len_trajectory,1],2*np.cos(pred_hd),2*np.sin(pred_hd),units='inches',scale=25)

	# plot the final location and starting location
	plt.plot(gt_trajectory[0,0],gt_trajectory[0,1],color='b', ls='', marker='o',markersize=4, markerfacecolor='b',label='Trajectory start (GT)')
	plt.plot(gt_trajectory[len_trajectory-1,0],gt_trajectory[len_trajectory-1,1],color='r', ls='', marker='o',markersize=6, markerfacecolor='r',label='Trajectory end (GT)')
	if init_pos is not None:
		plt.plot(init_pos[0],init_pos[1],color='g', ls='',marker='o',markersize=5,label='Initial position (GT)')
		if init_hd is not None:
			plt.quiver(init_pos[0],init_pos[1],2*np.cos(init_hd),2*np.sin(init_hd),units='inches',scale=15)

	# adjust the axis
	ax.axis('equal')
	ax.set_xlim((-env.radius-50, env.radius+50))
	ax.set_ylim((-env.radius-50, env.radius+50))
	plt.xticks(np.arange(-env.radius, env.radius+20, step=20))
	plt.yticks(np.arange(-env.radius, env.radius+20, step=20))
	ax.set_xlabel('X Position (cm)')
	ax.set_ylabel('Y Position (cm)')

	# turn on the grid
	plt.grid(True)
	plt.tight_layout()
	plt.legend()

	# show the figure
	plt.show()

def decode_trajectory_from_pc_activations(pc_means,pc_activations,k=3):
	'''
	Args:
		pc_means : Numpy array of size (n_pcs,2) containing center (x,y) locations
		pc_activations: Numpy array of size (traj,time,n_pcs) containing PC activations for each point in a trajectory
		k : Number of most active PCs to consider in order to decode position
	Return:
		A numpy array of size (traj,time,2) containing decoded position for each point in a trajectory
	'''
	# find top k active PCs
	sort_idx = np.argsort(pc_activations,axis=-1)
	topk_idx = sort_idx[...,-k:]
	topk_pc_x = np.take(pc_means[:,0],topk_idx)
	topk_pc_y = np.take(pc_means[:,1],topk_idx)
	pos_x = np.mean(topk_pc_x,axis=-1)
	pos_y = np.mean(topk_pc_y,axis=-1)
	return np.concatenate((pos_x[...,np.newaxis],pos_y[...,np.newaxis]),axis=-1)

def decode_direction_from_hdc_activations(hdc_means,hdc_activations,k=1):
	'''
	Args:
		hdc_means : Numpy array of size (n_hdcs,) containing peak theta
		hdc_activations: Numpy array of size (traj,time,n_hdcs) containing HDC activations for each point in a trajectory
		k : Number of most active HDCs to consider in order to decode position
	Return:
		A numpy array of size (traj,time,) containing decoded heading direction for each point in a trajectory
	'''
	# find top k active HDCs
	sort_idx = np.argsort(hdc_activations,axis=-1)
	topk_idx = sort_idx[...,-k:]
	topk_hdc = np.take(hdc_means,topk_idx)
	hd_dir = np.mean(topk_hdc,axis=-1)
	return hd_dir

# Load dataset
dataset = Dataset(batch_size=data_params['batch_size'])
data_generator = data.DataLoader(dataset, batch_size=data_params['batch_size'])
# create and load target_ensembles
place_cell_ensembles = get_place_cell_ensembles(env_size=ENV_SIZE, neurons_seed=SEED, n_pc=N_PC, radial=True, 
	radius=RADIUS, pc_scale=PC_SCALE)
head_direction_ensembles = get_head_direction_ensembles(neurons_seed=SEED, n_hdc=N_HDC,
	radial=True, radius=RADIUS)
target_ensembles = place_cell_ensembles + head_direction_ensembles
saved_targets = torch.load(target_ensemble_path)
place_cell_ensembles[0].means = torch.Tensor(saved_targets[0].means)
place_cell_ensembles[0].variances = torch.Tensor(saved_targets[0].variances)
head_direction_ensembles[0].means = torch.Tensor(saved_targets[1].means)
head_direction_ensembles[0].kappa = torch.Tensor(saved_targets[1].kappa)

# load the saved model
if pattern=='orthogonal':
	model = GridTorch(target_ensembles,n_pcs=N_PC[0],n_hdcs=N_HDC[0])
else:
	model = GridTorch_nonNeg(target_ensembles,n_pcs=N_PC[0],n_hdcs=N_HDC[0])
model.load_state_dict(torch.load(model_path,map_location='cpu'))
model.eval()
maze = watermaze(T=60)
for X,y in data_generator:
	init_pos, init_hd, ego_vel = X
	target_pos, target_hd = y
	initial_conds, _ = encode_initial_conditions(init_pos, init_hd, place_cell_ensembles, head_direction_ensembles)
	initial_conds = tuple(map(lambda x: x.to(device), initial_conds))

	with torch.no_grad():
		outs = model.forward(ego_vel.transpose(1, 0), initial_conds)
		logits_hd, logits_pc, bottleneck_acts, _, _ = outs

	gt_trajectory = target_pos.numpy().copy()
	gt_hd = target_hd.numpy().copy()
	# breakpoint()
	pred_pc_activations = logits_pc.transpose(1,0).numpy()
	pred_hdc_activations = logits_hd.transpose(1,0).numpy()
	pred_trajectory = decode_trajectory_from_pc_activations(saved_targets[0].means.numpy(),pred_pc_activations)
	pred_hd = decode_direction_from_hdc_activations(saved_targets[1].means.numpy(),pred_hdc_activations)
	# print(init_pos[0],gt_trajectory[0][:5],pred_trajectory[0][:5])
	plot_trajectories(maze,gt_trajectory[0],gt_hd=gt_hd[0],init_pos=init_pos[0],init_hd=init_hd[0],pred_trajectory=pred_trajectory[0],pred_hd=pred_hd[0])
