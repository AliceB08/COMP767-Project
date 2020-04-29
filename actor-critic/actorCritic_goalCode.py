import numpy as np
import random, warnings
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
import scipy
from scipy.sparse import csr_matrix
from tqdm import tqdm
from watermaze import watermaze, RMWTask, DMPTask
from placeCellEncoding import PlaceCellEncoding
from asymmetricTileCoding import AsymmetricTileCoding

class ActorCritic():
    def __init__(self,env,featureCoding='Place',numCells=None,bins=8,gamma=1,append_goal=False):
        '''
        Args
            env: environment (watermaze) object where planning is to be performed
            featureCoding: specifies the kind of feature coding -- 'Place' for place cells, 'Tile' for tile coding
            numCells: number of place cells for state space approximation
            bins: Number of bins and tilings for tile code
            gamma: discount factor
        '''
        self.env = env
        self.gamma = gamma
        self.append_goal = append_goal
        self.numActions = len(self.env.direction)  # 8 actions for 8 directions of motion
        if featureCoding=='Place': 
            self.featureCode = PlaceCellEncoding(self.env,N=numCells,sigma=16)
        elif featureCoding=='Tile':
            self.featureCode = AsymmetricTileCoding(dims=2,bins=bins,tilings=bins,state_limits=[[-self.env.radius,-self.env.radius],[self.env.radius,self.env.radius]])
        self.featureLength = self.getFeatureEncoding(self.env.position[:,0]).shape[0]
        if featureCoding =='Place':
            self.goal_coord_features = np.zeros(self.featureLength,)
        elif featureCoding=='Tile':
            self.goal_coord_features = csr_matrix((self.featureLength,1))
        self.featureLengthFactor = 2 if self.append_goal else 1  # factor to determine length of actor and critic weights
        self.criticWeights = np.zeros((self.featureLengthFactor*self.featureLength,1))
        self.actorWeights = np.zeros((self.featureLengthFactor*self.featureLength,self.numActions))
        
    def getFeatureEncoding(self,loc):
        # returns the feature encoding for an (x,y) location in space
        return self.featureCode.getEncoding(loc)

    def append_goal_to_enc(self,features):
        if self.append_goal:
            if scipy.sparse.issparse(features):
                return scipy.sparse.vstack([features,self.goal_coord_features])
            else:
                return np.hstack([features,self.goal_coord_features])
        else:
            return features
    
    def get_actor_output(self,features):
        # returns the actor's output (preferences) for each action
        features = self.append_goal_to_enc(features)
        if scipy.sparse.issparse(features):
            return self.actorWeights.T*features
        else:
            return np.sum(self.actorWeights*features.reshape(-1,1),axis=0)
    
    def get_critic_output(self,features):
        # returns the critic's output (estimated value function)
        features = self.append_goal_to_enc(features)
        if scipy.sparse.issparse(features):
            output = np.sum(self.criticWeights.T*features)  # np.sum is redundant and required to get the element of the (1,1) matrix
        else:
            output = np.sum(self.criticWeights*features.reshape(-1,1))
        # checking for Nans while debugging
        if np.isnan(output):
            print('critic output',self.criticWeights.min(),self.criticWeights.max())
        return output
        
    def get_action(self,loc):
        # choose an action using a stochastic policy, with prob proportional to softmax of actor outputs
        loc_features = self.getFeatureEncoding(loc)
        actor_output = np.ravel(self.get_actor_output(loc_features))
        actor_output -= actor_output.max()  # to avoid overflow
        action_prob = np.exp(actor_output)
        action_prob = action_prob/np.sum(action_prob)
        # checking for Nans while debugging
        if np.isnan(action_prob).any():
            print(loc_features.min(),loc_features.max(),self.actorWeights.min(),self.actorWeights.max(),actor_output,action_prob)
        action = np.random.choice(self.numActions,1,p=action_prob)[0]
        return action       
    
    def apply_function_approx(self,features):
        # more general function to approximate the value function --> here just calls the critic outputs
        return self.get_critic_output(features)
    
    def plot_value_function(self):
        '''
        Plots the estimated value function over the environment locations (here place cell centres are used)
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
        
        locs = self.featureCode.pos
        value_fns = np.array([self.get_critic_output(self.getFeatureEncoding(l)) for l in locs])
        ValueFnPlot = plt.scatter(locs[:,0],locs[:,1],s=self.featureCode.sigma,c=value_fns)
        ax.add_artist(ValueFnPlot)
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
        plt.title('Estimated value function')
        plt.show()

    def plot_critic_weights(self):
        '''
        Plots the weights of the critic corresponding to each place cell, the color of the place cell corresponding to critic weight
        '''
        if self.append_goal:
            warnings.warn("Critic weights plotter not implemented for append goal")
            return
        # create the figure 
        fig = plt.figure('Critic Weights')
        ax = fig.gca()

        # create the pool perimeter
        pool_perimeter = plt.Circle((0, 0), self.env.radius, fill=False, color='b', ls='-')
        ax.add_artist(pool_perimeter)

        # create theplatform
        platform = plt.Circle(self.env.platform_location, self.env.platform_radius, fill=False, color='r', ls='-')
        ax.add_artist(platform)
        
        locs = self.featureCode.pos
        CriticWeightPlot = plt.scatter(locs[:,0],locs[:,1],s=self.featureCode.sigma,c=self.criticWeights.reshape(-1))
        ax.add_artist(CriticWeightPlot)
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
        plt.title('Critic weights')
        plt.show(block=False)

    def plot_actor_preferences(self):
        '''
        Plots the action with maximum value in actor output at location of place cell centres (color coded by action value)
        '''
        # create the figure 
        fig = plt.figure('Actor preference')
        ax = fig.gca()

        # create the pool perimeter
        pool_perimeter = plt.Circle((0, 0), self.env.radius, fill=False, color='b', ls='-')
        ax.add_artist(pool_perimeter)

        # create theplatform
        platform = plt.Circle(self.env.platform_location, self.env.platform_radius, fill=False, color='r', ls='-')
        ax.add_artist(platform)

        locs = self.featureCode.pos
        action_values = np.array([np.argmax(self.get_actor_output(self.getFeatureEncoding(l))) for l in locs])
        list_actions = np.array(list(self.env.direction.values()))
        ActionPlot = plt.quiver(locs[:,0],locs[:,1],5*np.cos(np.take(list_actions,action_values)),5*np.sin(np.take(list_actions,action_values)))
        ax.add_artist(ActionPlot)
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
        plt.title('Preferred Action')
        plt.show(block=False)
    
    def TD_lambda(self,alpha,lamda=0,day=1,episodes=10,verbose=False):
        '''
        Implements a TD-lambda algorithm for training the actor and critic networks. 
        Uses a linear actor and critic models
        '''
        time_taken = np.zeros((episodes,))
        self.env.update_platform_location(day)
        for e in range(episodes):
            # reset the env
            self.env.startposition()
            self.env.t = 0
            eligibility_trace = np.zeros((self.featureLengthFactor*self.featureLength,1))
            # get current location
            curr_loc = self.env.position[:,self.env.t]
            curr_loc_encoding = self.getFeatureEncoding(curr_loc)
            while(not self.env.timeup() and not self.env.atgoal()):
                # select an action from the actor network
                A = self.get_action(curr_loc)
                # move the rat (agent)
                self.env.move(A)
                # get next location
                next_loc = self.env.position[:,self.env.t]
                next_loc_encoding = self.getFeatureEncoding(next_loc)
                # if goal is reached, set reward to 1 otherwise 0. Calculate TD-error accordingly
                if self.env.atgoal():
                    self.goal_coord_features = curr_loc_encoding.copy()
                    reward = 1
                    TD_error = reward - self.get_critic_output(curr_loc_encoding)
                else:
                    reward = 0
                    TD_error = self.gamma*self.get_critic_output(next_loc_encoding) - self.get_critic_output(curr_loc_encoding)
                # Linear function approx is used, hence weight derivative is the current feature encoding
                feature_encoding = self.append_goal_to_enc(curr_loc_encoding)
                weight_derivative = feature_encoding.reshape(-1,1)
                # calculate the eligibility trace
                eligibility_trace = lamda*self.gamma*eligibility_trace + weight_derivative
                del_func = alpha*TD_error*eligibility_trace if not scipy.sparse.issparse(curr_loc_encoding) else (
                    alpha/self.featureCode.tilings)*TD_error*eligibility_trace
                self.criticWeights +=  del_func     # update critic weights
                self.actorWeights[:,A] += np.ravel(del_func)      # update actor weights
                # check for Nans while debugging
                if np.isnan(self.criticWeights).any():
                    print('critic weights',self.get_critic_output(next_loc_encoding),self.get_critic_output(curr_loc_encoding),TD_error,eligibility_trace.min(),eligibility_trace.max())
                if np.isnan(self.actorWeights[:,A]).any():
                    print(self.get_critic_output(next_loc_encoding),self.get_critic_output(curr_loc_encoding),TD_error,eligibility_trace.min(),eligibility_trace.max())
                if verbose:
                    # if verbose, for each action, plot the place cell activations
                    # If goal is reached, print the TD-error and the critic weights and actor preferences learned
                    if not scipy.sparse.issparse(curr_loc_encoding):
                        self.featureCode.plotPlaceCellActivations(curr_loc)
                    if reward!=0:
                        print(TD_error,eligibility_trace.shape)
                        # print('Critic weights and updates are '+('equal!' if (self.criticWeights==alpha*eligibility_trace).all() else 'not equal!'))
                        self.plot_critic_weights()
                        self.plot_actor_preferences()
                # copy the next state (loc) variables the current state (loc) variables for the next step
                curr_loc = next_loc.copy()
                curr_loc_encoding = next_loc_encoding.copy()
            time_taken[e] = self.env.t      # note the time taken by the agent to complete the episode
            if verbose:
                # if verbose, for certain episodes plot the path taken and the estimated value function so far
                if e%1==0:
                    print(e,self.env.t,self.env.atgoal())
                    self.env.plotpath()
                    self.plot_value_function()
        return time_taken
                
if __name__=='__main__':
    all_tasks = {0: 'Fixed target', 1: 'RMW', 2: 'DMP'}
    np.random.seed(10)
    days = 10
    sessions = 3
    episodes = 200
    featureCoding = 'Place'     # 'Tile'
    append_goal = True
    task = 2    # 0 - watermaze simple, 1 - RMW, 2 - DMP
    if task==0:
        maze = watermaze(T=60)
    elif task==1:
        maze = RMWTask(T=60,days=days)
    elif task==2:
        maze = DMPTask(T=60,days=days)
    maze.startposition()
    AC = ActorCritic(env=maze,featureCoding=featureCoding,numCells=493,bins=8,gamma=0.9,append_goal=append_goal)
    # time_arr = AC.TD_lambda(alpha=0.01,lamda=0.0,episodes=20,verbose=True)
    time_mean = []
    time_std = []
    for d in tqdm(range(1,1+days*sessions)):
        if featureCoding=='Place':
            time_arr = AC.TD_lambda(alpha=0.003,lamda=0.5,day=1+(d-1)//sessions,episodes=episodes)     # hyperparams for placeCellEncoding
        elif featureCoding=='Tile':
            time_arr = AC.TD_lambda(alpha=0.1,lamda=0.6,day=1+(d-1)//sessions,episodes=episodes)    # hyperparams for tileCoding
        # AC.plot_actor_preferences()
        # AC.plot_value_function()
        tqdm.write("{} {}".format(time_arr.mean(),time_arr.std()))
        time_mean.append(time_arr.mean())
        time_std.append(time_arr.std()/np.sqrt(len(time_arr)))
    # plt.errorbar(np.linspace(1,days*sessions,days*sessions),time_mean,time_std)
    ## Plotting similar to paper figure
    tick_arr = []
    tick_label_arr = []
    for d in range(days):
        plt.errorbar(np.linspace(d+d*sessions+1,d+(d+1)*sessions,sessions),time_mean[d*sessions:(d+1)*sessions],time_std[d*sessions:(d+1)*sessions],color='k')
        tick_arr.extend(np.linspace(d+d*sessions+1,d+(d+1)*sessions,sessions))
        label_arr = ['']*sessions
        label_arr[sessions//2] = str(d+1)
        tick_label_arr.extend(label_arr)
    plt.xticks(tick_arr,tick_label_arr,fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylabel('Steps',size=14)
    plt.xlabel('Days',size=14)
    plt.title('Escape latency for AC agent({}) using {} coding on {} task'.format('goal memory' if append_goal else 'no memory'
        ,featureCoding,all_tasks[task]),fontsize=16)
    plt.show()