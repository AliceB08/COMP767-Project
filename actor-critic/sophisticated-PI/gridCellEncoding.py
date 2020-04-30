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

class GridCellEncoding():
	def __init__(self,env,model_path,target_ensemble_path,pattern='orthogonal',n_pcs=[256],pc_scale=[0.01],
		n_hdcs=[8],nh_lstm=128,nh_bottleneck=256,seed=9999):
		'''
		Args
		    env: environment (watermaze) object where Path Integration is being performed
		    model_path: path to trained Path Integration model (outputs x,y,theta)
		'''
		self.env = env
		self.pattern = pattern
		self.n_pcs = n_pcs
		self.pc_scale = pc_scale
		self.n_hdcs = n_hdcs
		self.nh_lstm = nh_lstm
		self.nh_bottleneck = nh_bottleneck
		# define the target ensembles
		self.place_cell_ensembles = get_place_cell_ensembles(env_size=self.env.radius*2, neurons_seed=seed, n_pc=self.n_pcs, 
			radial=True, radius=self.env.radius,pc_scale=self.pc_scale)
		self.head_direction_ensembles = get_head_direction_ensembles(neurons_seed=seed, n_hdc=self.n_hdcs,
			radial=True, radius=self.env.radius)
		self.target_ensembles = self.place_cell_ensembles + self.head_direction_ensembles
		# load the saved means from file
		saved_targets = torch.load(target_ensemble_path)
		self.place_cell_ensembles[0].means = torch.Tensor(saved_targets[0].means)
		self.place_cell_ensembles[0].variances = torch.Tensor(saved_targets[0].variances)
		self.head_direction_ensembles[0].means = torch.Tensor(saved_targets[1].means)
		self.head_direction_ensembles[0].kappa = torch.Tensor(saved_targets[1].kappa)

		self.device = torch.device("cpu")
	    # define the model
		if pattern=='orthogonal':
			self.model = GridTorch(self.target_ensembles,nh_lstm=self.nh_lstm,nh_bottleneck=self.nh_bottleneck,
				n_pcs=self.n_pcs[0],n_hdcs=self.n_hdcs[0])
		else:
			self.model = GridTorch_nonNeg(self.target_ensembles,nh_lstm=self.nh_lstm,nh_bottleneck=self.nh_bottleneck,
				n_pcs=self.n_pcs[0],n_hdcs=self.n_hdcs[0])
		# load the pretrained model
		self.model.load_state_dict(torch.load(model_path,map_location='cpu'))
		self.model.eval()

    	#store additonal information flags and vars
		self.is_initialized = False
		self.featureLength = nh_bottleneck
		self.bottleneck_activations = None
		self.h_t = None
		self.c_t = None

	def getEncoding(self,ego_vel=None):
		# this should be the super class function common to all feature encoders
		if ego_vel is None:
			ego_vel = torch.Tensor([0.,1.,0.])
		if self.is_initialized:
			return self.getPositionEncoding(ego_vel)
		else:
			return self.setInitConditions()

	def setInitConditions(self):
		# set the initial cell and state embeddings at start of trajectory
		assert self.env.t==0, "The agent is already underway, call this at the beginning of trajectory."
		# init_cond = [self.env.position[:,self.env.t],[np.arctan2(self.env.prevdir[1], self.env.prevdir[0])]]
		# breakpoint()
		init_pos = torch.Tensor(self.env.position[:,self.env.t]).unsqueeze(0)
		init_hd = torch.Tensor([np.arctan2(self.env.prevdir[1], self.env.prevdir[0])]).unsqueeze(0)
		initial_conds, _ = encode_initial_conditions(init_pos, init_hd, self.place_cell_ensembles, self.head_direction_ensembles)
		initial_conds = tuple(map(lambda x: x.to(self.device), initial_conds))
		init = torch.cat(initial_conds,dim=-1)
		with torch.no_grad():
			init_state = self.model.state_embed(init)
			init_cell = self.model.cell_embed(init)
			self.h_t, self.c_t = init_state, init_cell
			self.bottleneck_activations = self.model.bottleneck(self.h_t)
		self.is_initialized = True
		output_encoding = np.ravel(self.bottleneck_activations.numpy().copy())
		return output_encoding

	def resetEncoder(self):
		assert self.env.atgoal() or self.env.timeup(), "The trajectory has not yet ended, encoder should not be reset!"
		self.is_initialized = False

	def getPositionEncoding(self,ego_vel):
		# breakpoint()
		assert self.is_initialized, "LSTM not initialized, call setInitConditions() before."
		ego_vel_tensor = torch.Tensor(ego_vel)
		with torch.no_grad():
			_, (self.h_t,self.c_t) = self.model.lstm(ego_vel_tensor.view(1,1,-1),(self.h_t.unsqueeze(0),self.c_t.unsqueeze(0)))
			self.h_t, self.c_t = self.h_t.squeeze(0), self.c_t.squeeze(0)
			self.bottleneck_activations = self.model.bottleneck(self.h_t)
		output_encoding = np.ravel(self.bottleneck_activations.numpy().copy())
		return output_encoding

	def decode_trajectory_from_pc_activations(self,pc_activations,k=3):
		'''
		Args:
			pc_activations: Numpy array of size (traj,time,n_pcs) containing PC activations for each point in a trajectory
			k : Number of most active PCs to consider in order to decode position
		Return:
			A numpy array of size (traj,time,2) containing decoded position for each point in a trajectory
		'''
		# find top k active PCs
		# breakpoint()
		pc_means = self.place_cell_ensembles[0].means.numpy()
		sort_idx = np.argsort(pc_activations,axis=-1)
		topk_idx = sort_idx[...,-k:]
		topk_pc_x = np.take(pc_means[:,0],topk_idx)
		topk_pc_y = np.take(pc_means[:,1],topk_idx)
		pos_x = np.mean(topk_pc_x,axis=-1)
		pos_y = np.mean(topk_pc_y,axis=-1)
		return np.concatenate((pos_x[...,np.newaxis],pos_y[...,np.newaxis]),axis=-1)

	def decode_direction_from_hdc_activations(self,hdc_activations,k=1):
		'''
		Args:
			hdc_activations: Numpy array of size (traj,time,n_hdcs) containing HDC activations for each point in a trajectory
			k : Number of most active HDCs to consider in order to decode position
		Return:
			A numpy array of size (traj,time,) containing decoded heading direction for each point in a trajectory
		'''
		# find top k active HDCs
		hdc_means = self.head_direction_ensembles[0].means.numpy()
		sort_idx = np.argsort(hdc_activations,axis=-1)
		topk_idx = sort_idx[...,-k:]
		topk_hdc = np.take(hdc_means,topk_idx)
		hd_dir = np.mean(topk_hdc,axis=-1)
		return hd_dir

	def plot_pathIntegration_output(self,gt_pos=None,gt_hd=None):
		with torch.no_grad():
			curr_pc_activations = self.model.pc_logits(self.bottleneck_activations)
			curr_hd_activations = self.model.hd_logits(self.bottleneck_activations)
		curr_pos = self.decode_trajectory_from_pc_activations(curr_pc_activations).squeeze(0)
		curr_hd = self.decode_direction_from_hdc_activations(curr_hd_activations).squeeze(0)
		# create the figure 
		fig = plt.figure()
		ax = fig.gca()

		# create the pool perimeter
		pool_perimeter = plt.Circle((0, 0), self.env.radius, fill=False, color='b', ls='-')
		ax.add_artist(pool_perimeter)

		# create theplatform
		platform = plt.Circle(self.env.platform_location, self.env.platform_radius, fill=False, color='r', ls='-')
		ax.add_artist(platform)

		# plot the agent's predicted positon and heading
		plt.plot(curr_pos[0],curr_pos[1],color='k',marker='o',markersize=7, markerfacecolor='k',ls='',label='Predicted position')
		plt.quiver(curr_pos[0],curr_pos[1],2*np.cos(curr_hd),2*np.sin(curr_hd),units='inches',scale=20)
		# plot the agent's gt position and heading
		if gt_pos is not None:
			plt.plot(gt_pos[0],gt_pos[1],color='r',marker='x',markersize=7, markerfacecolor='r',ls='',label='Ground truth position')
			if gt_hd is not None:
				plt.quiver(gt_pos[0],gt_pos[1],2*np.cos(gt_hd),2*np.sin(gt_hd),units='inches',scale=20,color='g')

		# adjust the axis
		ax.axis('equal')
		ax.set_xlim((-self.env.radius-50, self.env.radius+50))
		ax.set_ylim((-self.env.radius-50, self.env.radius+50))
		plt.xticks(np.arange(-self.env.radius, self.env.radius+20, step=20))
		plt.yticks(np.arange(-self.env.radius, self.env.radius+20, step=20))
		ax.set_xlabel('X Position (cm)')
		ax.set_ylabel('Y Position (cm)')

		# turn on the grid
		plt.grid(True)
		plt.tight_layout()
		plt.legend()

		# show the figure
		plt.show()

if __name__=='__main__':
	maze = watermaze(T=60)
	model_path = 'exp1/model_epoch_4999.pt'
	target_ensemble_path = 'exp1/target_ensembles.pt'
	pattern = 'orthogonal'
	pc_scale = [0.01]
	encoder = GridCellEncoding(env=maze,model_path=model_path,target_ensemble_path=target_ensemble_path,pattern=pattern,pc_scale=pc_scale)
	maze.startposition()
	maze.t=0
	encoder.getEncoding()
	while(not maze.timeup() and not maze.atgoal()):
		action = np.random.randint(0,8)
		trajectory = maze.move(action)
		encoder.getEncoding(trajectory['ego_vel'])
		# encoder.getEncoding((trajectory['target_pos'],[trajectory['target_hd']]))
		encoder.plot_pathIntegration_output(gt_pos=trajectory['target_pos'],gt_hd=trajectory['target_hd'])
	encoder.resetEncoder()