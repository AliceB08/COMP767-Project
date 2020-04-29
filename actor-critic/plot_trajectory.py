import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
import scipy
from scipy.sparse import csr_matrix
from tqdm import tqdm
from watermaze import watermaze, RMWTask, DMPTask

#################### v NEEDS to be rewritten v ###################################################
def get_nearest_cell_pos(PC_pos,activation,k=3):
	'''
	Decode position using centers of k maximally active place cells.
    Args: 
    	activation: Place cell activations of shape [batch_size, sequence_length, Np].
    	k: Number of maximally active place cells with which to decode position.
	Returns:
		pred_pos: Predicted 2d position with shape [batch_size, sequence_length, 2].
	'''
	idx = np.argsort(activation,axis=-1)
	topk_idx = idx[:,:,-k:]
	pred_pos = np.mean(np.take(self.us,topk_idx,axis=0),axis=-2)
	return pred_pos
#################### ^ NEEDS to be rewritten ^ ###################################################

def plot_trajectories(env,gt_trajectory,gt_hd=None,init_pos=None,init_hd=None):
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
	plt.plot(gt_trajectory[0:len_trajectory,0],gt_trajectory[0:len_trajectory,1], color='k', ls='-')
	# plot the head directions
	if gt_hd is not None:
		plt.quiver(gt_trajectory[0:len_trajectory,0],gt_trajectory[0:len_trajectory,1],2*np.cos(gt_hd),2*np.sin(gt_hd),units='inches',scale=25)

	# plot the final location and starting location
	plt.plot(gt_trajectory[0,0],gt_trajectory[0,1],color='b', marker='o',markersize=4, markerfacecolor='b')
	plt.plot(gt_trajectory[len_trajectory-1,0],gt_trajectory[len_trajectory-1,1],color='r', marker='o',markersize=6, markerfacecolor='r')
	if init_pos is not None:
		plt.plot(init_pos[0],init_pos[1],color='g',marker='o',markersize=5)
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

	# show the figure
	plt.show()

if __name__=='__main__':
	maze = watermaze(T=60)
	data = np.load('tmp_watermaze_data.pkl',allow_pickle=True)
	for idx in range(len(data)):
		trajectory = data[idx]['target_pos']
		hd = data[idx]['target_hd']
		plot_trajectories(maze,trajectory,gt_hd=hd,init_pos=data[idx]['init_pos'],init_hd=data[idx]['init_hd'])