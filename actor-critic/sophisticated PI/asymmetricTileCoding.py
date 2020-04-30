import os, sys, random, math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
import warnings
import scipy
from scipy.sparse import csr_matrix
# from mountainCar import MountainCar
from watermaze import watermaze, RMWTask, DMPTask

class AsymmetricTileCoding():
	def __init__(self,dims=2,bins=8,tilings=8,state_limits=[[0,0],[1,1]],seed=13):
		self.dims = dims
		self.bins = bins
		self.tilings = tilings
		self.states_min = np.array(state_limits[0])
		self.states_max = np.array(state_limits[1])
		self.seed = seed
		np.random.seed(seed)
		if self.dims!=2:
			raise NotImplementedError
		if self.tilings<4*self.dims:
			warnings.warn("Number of tilings should ideally be higher for proper asymmetric coding",RuntimeWarning)
		self.interval = np.array([(self.states_max[dim]-self.states_min[dim])/(self.bins) for dim in range(self.dims)])
		self.offsets = []
		for dim in range(self.dims):
			dim_displacement = 2*dim+1	# select sequential odd numbers starting from 1
			dim_offsets = np.linspace(0,dim_displacement*self.interval[dim],self.tilings+1)[:-1]
			dim_offsets = self.states_min[dim]+dim_offsets - np.mean(dim_offsets) 	# centering the offsets to have roughly symmetric spanning of the state space
			self.offsets.append(dim_offsets)
		self.offsets = np.array(self.offsets)
		# select random points in env for reference/plotting
		sampling_radius = np.max(self.states_max-self.states_min)/2
		r_loc = sampling_radius*np.random.uniform(0,1,(self.tilings*self.bins**self.dims,1))
		theta_loc = 2*np.pi*np.random.uniform(0,1,(self.tilings*self.bins**self.dims,1))
		X_loc = r_loc*np.cos(theta_loc)
		Y_loc = r_loc*np.sin(theta_loc)
		self.pos = np.hstack((X_loc,Y_loc))
		self.sigma = 10

	def getEncoding(self,loc):
		# this should be the super class function common to all feature encoders
		enc = self.construct_flattened_tile_idx(state=loc) 	# returned numpy array that has indices of non-zero tiles --> shape (tilings,)
		return csr_matrix((np.ones(self.tilings,),(enc,np.zeros(self.tilings,))),shape=(self.tilings*self.bins**self.dims,1)) 	# returns a sparse array of shape (tilings*bins^dims,1)

	def getCodedState(self,state):
		state_dim = 1 if np.shape(state)==() else np.shape(state)[0]
		assert state_dim==self.dims, "State to be coded ({}) is of different dimension than the Asymmetric Tile Coding object ({})!".format(state_dim,self.dims)
		'''
		tile_idxs = []
		for dim in range(self.dims):
			tile_idx = np.floor((state[dim]-self.offsets[dim])/self.interval[dim]).astype(int)
			tile_idx[tile_idx>self.bins-1]=self.bins-1	# ensuring any index is not greater than bins+1 (max state idx)
			tile_idx[tile_idx<0]=0						# ensuring any index is not less than 0 (min state idx)
			tile_idxs.append(tile_idx)
		tile_idxs = np.array(tile_idxs)
		# single line implementation below :)
		'''
		tile_idxs = np.floor((state[:,np.newaxis]-self.offsets)/self.interval[:,np.newaxis]).astype(int)
		tile_idxs[tile_idxs>self.bins-1]=self.bins-1
		tile_idxs[tile_idxs<0]=0
		# assert np.all(tile_idxs2.shape==tile_idxs.shape) and np.all(tile_idxs==tile_idxs2), "Error in vectorized TileCoding"
		return tile_idxs.T 		# returned numpy array will be of shape (num_tilings,dims) --> each row stores the indices of the bin that should be 1

	def construct_flattened_tile_idx(self,state):
		tileCode = self.getCodedState(state)
		'''
		flattened_tile_idxs = []
		for tile in range(tilings):
			code = tileCode[tile]
			idx = np.sum(code*(np.logspace(dims-1,0,dims,base=bins)))
			flattened_tile_idxs.append((bins**dims)*tile+idx)
		# Single line implementation below :)
		'''
		# flattened_tile_idxs = np.array([(bins**dims)*tile+np.sum(tileCode[tile]*(np.logspace(dims-1,0,dims,base=bins))) for tile in range(tilings)]).astype(int)
		flattened_tile_idxs = ((self.bins**self.dims)*np.linspace(0,self.tilings-1,self.tilings)+
			np.sum(np.multiply(tileCode,np.logspace(self.dims-1,0,self.dims,base=self.bins)),axis=1)).astype(int)
		return flattened_tile_idxs

if __name__=='__main__':
	ATC = AsymmetricTileCoding(dims=2,bins=8,tilings=8,state_limits=[[0,0],[1,1]])
	print(ATC.offsets)
	M = MountainCar()
	for steps in range(10):
		r,is_terminated = M.takeAction(-1)
		print(r,is_terminated,M.curr_state,M.getNormalizedState())
		tileCodes = ATC.getCodedState(M.getNormalizedState())
		print(tileCodes.shape,tileCodes)
		if is_terminated:
			R.reset()
	print(ATC.offsets)