import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from tqdm import tqdm
from watermaze import watermaze

## Maybe add a FeatureEncoding super class for which PlaceCellEncoding, GridCellEncoding, TileCoding etc. can be subclasses

class PlaceCellEncoding():
    def __init__(self,env,N=493,sigma=16):
        '''
        Args
            env: environment (watermaze) object where place fields are to be randomly distributed
            N: number of place cells
            sigma: breadth of field of each place cell
        '''
        self.env = env
        if N is not None:
            self.N = N
        else:
             self.N = 493   
        self.sigma = sigma
        # uniformly distribute the place fields in ploar coordinates
        r_loc = self.env.radius*np.random.uniform(0,1,(self.N,1))
        theta_loc = 2*np.pi*np.random.uniform(0,1,(self.N,1))
        X_loc = r_loc*np.cos(theta_loc)
        Y_loc = r_loc*np.sin(theta_loc)
        self.pos = np.hstack((X_loc,Y_loc))
        
    def getEncoding(self,loc):
        # this should be the super class function common to all feature encoders
        return self.getPlaceEncoding(loc)
        
    def getPlaceEncoding(self,loc):
        '''
        Args
            loc: a tuple (x,y) representing the x and y axis position of the point to be encoded
        '''
        diff = self.pos - loc
        rates = np.exp(-np.sum(diff**2,axis=1)/(2*self.sigma*self.sigma))
        return rates
    
    def plotPlaceCells(self):
        '''
        Creates a visualization of how place cell centers are distributed in the environment
        '''
        # create the figure 
        fig = plt.figure()
        ax = fig.gca()

        # create the pool perimeter
        pool_perimeter = plt.Circle((0, 0), self.env.radius, fill=False, color='b', ls='-')
        ax.add_artist(pool_perimeter)

        # create theplatform
        platform = plt.Circle(self.env.platform_location, self.env.platform_radius, fill=False, color='r', ls='-')
        ax.add_artist(platform)
        
        # plot place cell centers
        placeCells = plt.scatter(self.pos[:,0],self.pos[:,1],s=self.sigma)
        ax.add_artist(placeCells)
        
        # turn on the grid
        plt.grid(True)
        plt.tight_layout()
        ax.axis('equal')
        ax.set_xlim((-self.env.radius-50, self.env.radius+50))
        ax.set_ylim((-self.env.radius-50, self.env.radius+50))
        plt.xticks(np.arange(-self.env.radius, self.env.radius+20, step=20))
        plt.yticks(np.arange(-self.env.radius, self.env.radius+20, step=20))
        ax.set_xlabel('X Position (cm)')
        ax.set_ylabel('Y Position (cm)')
        plt.show()
        
    def plotPlaceCellActivations(self,loc):
        '''
        Creates a visualization of how place cell activations vary as the agent (denoted by 'X') moves in the environment
        Note: this should generate a live plot which shows the agent moving in the env and the place cell activations changing
        '''
        placeCodes = self.getPlaceEncoding(loc)
        # create the figure 
        fig = plt.figure('PlaceCellEncoding')
        ax = fig.gca()

        # create the pool perimeter
        pool_perimeter = plt.Circle((0, 0), self.env.radius, fill=False, color='b', ls='-')
        ax.add_artist(pool_perimeter)

        # create theplatform
        platform = plt.Circle(self.env.platform_location, self.env.platform_radius, fill=False, color='r', ls='-')
        ax.add_artist(platform)
        
        # plot place cell centers
        placeCells = plt.scatter(self.pos[:,0],self.pos[:,1],s=self.sigma,c=placeCodes)
        ax.add_artist(placeCells)
        plt.scatter(loc[0],loc[1],marker='x',c='k',s=50)
        # turn on the grid
        plt.grid(True)
        plt.tight_layout()
        ax.axis('equal')
        ax.set_xlim((-self.env.radius-50, self.env.radius+50))
        ax.set_ylim((-self.env.radius-50, self.env.radius+50))
        plt.xticks(np.arange(-self.env.radius, self.env.radius+20, step=20))
        plt.yticks(np.arange(-self.env.radius, self.env.radius+20, step=20))
        ax.set_xlabel('X Position (cm)')
        ax.set_ylabel('Y Position (cm)')
        plt.colorbar()
        plt.title('Place cell activations')
        plt.show(block=False)
        plt.pause(0.2)
        plt.clf()