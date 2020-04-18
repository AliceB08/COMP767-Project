import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from tqdm import tqdm
from watermaze import watermaze, RWMTask, DMPTask
from placeCellEncoding import PlaceCellEncoding

class Coordinate_ActorCritic():
    def __init__(self,env,numCells=None,gamma=1):
        '''
        Args
            env: environment (watermaze) object where planning is to be performed
            numCells: number of place cells for state space approximation
            gamma: discount factor
        '''
        self.env = env
        self.gamma = gamma
        self.numActions = len(self.env.direction)  # 8 actions for 8 directions of motion
        self.featureCode = PlaceCellEncoding(self.env,N=numCells,sigma=16)
        self.featureLength = len(self.getFeatureEncoding(self.env.position[:,0]))
        self.criticWeights = np.zeros((self.featureLength,1))
        self.XWeights = np.zeros((self.featureLength,1))
        self.YWeights = np.zeros((self.featureLength,1))
        self.actorWeights = np.zeros((self.featureLength,self.numActions))
        self.coord_action_weight = 0
        self.goal_coord_features = None
        
    def getFeatureEncoding(self,loc):
        # returns the feature encoding for an (x,y) location in space
        return self.featureCode.getEncoding(loc)
    
    def get_X_coord(self,features):
        # returns the agent's belief over its X-coordinate
        return np.sum(self.XWeights*features.reshape(-1,1))

    def get_Y_coord(self,features):
        # returns the agent's belief over its Y-coordinate
        return np.sum(self.YWeights*features.reshape(-1,1))

    def get_actor_output(self,features):
        # returns the actor's output (preferences) for each action
        return np.sum(self.actorWeights*features.reshape(-1,1),axis=0)

    def get_critic_output(self,features):
        # returns the critic's output (estimated value function)
        output = np.sum(self.criticWeights*features.reshape(-1,1))
        # checking for Nans while debugging
        if np.isnan(output):
            print('critic output',self.criticWeights.min(),self.criticWeights.max())
        return output
        
    def get_coord_action(self,features):
        # returns the action corresponding to shortest path to goal
        assert self.goal_coord_features is not None, 'Coordinate action cannot be specified until goal location is known!'
        goal_x = self.get_X_coord(self.goal_coord_features)
        goal_y = self.get_Y_coord(self.goal_coord_features)
        curr_x = self.get_X_coord(features)
        curr_y = self.get_Y_coord(features)
        desired_direction = np.mod(np.arctan2((goal_y-curr_y),(goal_x-curr_x)),2*np.pi)     # converting angle to [0,2*np.pi] from [-np.pi,np.pi]
        possible_dirs = np.array(list(self.env.direction.values()))
        action = np.argmin(np.abs(desired_direction-possible_dirs))     # choose the action closest to the desired action
        return action

    def get_action(self,loc):
        # choose an action using a stochastic policy, with prob proportional to softmax of actor outputs
        # returns 0-7 if one of the possible actions is chosen from the actor's weights, 8 if a_coord is chosen
        loc_features = self.getFeatureEncoding(loc)
        actor_output = self.get_actor_output(loc_features)
        if self.goal_coord_features is not None:
            actor_output = np.append(actor_output,self.coord_action_weight)
        actor_output -= actor_output.max()  # to avoid overflow
        action_prob = np.exp(actor_output)
        action_prob = action_prob/np.sum(action_prob)
        # checking for Nans while debugging
        if np.isnan(action_prob).any():
            print(loc_features.min(),loc_features.max(),self.actorWeights.min(),self.actorWeights.max(),actor_output,action_prob)
        action = np.random.choice(len(action_prob),1,p=action_prob)[0]
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

    def plot_X_estimates(self):
        '''
        Plots the X_coord estimates corresponding to each place cell, the color of the place cell corresponding to X_coord value
        '''
        # create the figure 
        fig = plt.figure('X estimates')
        ax = fig.gca()

        # create the pool perimeter
        pool_perimeter = plt.Circle((0, 0), self.env.radius, fill=False, color='b', ls='-')
        ax.add_artist(pool_perimeter)

        # create theplatform
        platform = plt.Circle(self.env.platform_location, self.env.platform_radius, fill=False, color='r', ls='-')
        ax.add_artist(platform)
        
        locs = self.featureCode.pos
        X_estimates = np.array([self.get_X_coord(self.getFeatureEncoding(l)) for l in locs])
        XPlot = plt.scatter(locs[:,0],locs[:,1],s=self.featureCode.sigma,c=X_estimates)
        ax.add_artist(XPlot)
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
        plt.title('X_coord estimates')
        plt.show(block=False)

    def plot_Y_estimates(self):
        '''
        Plots the Y_coord estimates corresponding to each place cell, the color of the place cell corresponding to Y_coord value
        '''
        # create the figure 
        fig = plt.figure('Y estimates')
        ax = fig.gca()

        # create the pool perimeter
        pool_perimeter = plt.Circle((0, 0), self.env.radius, fill=False, color='b', ls='-')
        ax.add_artist(pool_perimeter)

        # create theplatform
        platform = plt.Circle(self.env.platform_location, self.env.platform_radius, fill=False, color='r', ls='-')
        ax.add_artist(platform)
        
        locs = self.featureCode.pos
        Y_estimates = np.array([self.get_Y_coord(self.getFeatureEncoding(l)) for l in locs])
        YPlot = plt.scatter(locs[:,0],locs[:,1],s=self.featureCode.sigma,c=Y_estimates)
        ax.add_artist(YPlot)
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
        plt.title('Y_coord estimates')
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
        peak_action_preference = np.array([np.max(self.get_actor_output(self.getFeatureEncoding(l))) for l in locs])
        action_prob = np.array([1./(np.sum(np.exp(self.get_actor_output(self.getFeatureEncoding(l))-peak_action_preference[i]))
            +int(self.goal_coord_features is not None)*np.exp(self.coord_action_weight-peak_action_preference[i])) 
            for i,l in enumerate(locs)])
        list_actions = np.array(list(self.env.direction.values()))
        ActionPlot = plt.quiver(locs[:,0],locs[:,1],5*action_prob*np.cos(np.take(list_actions,action_values)),
            5*action_prob*np.sin(np.take(list_actions,action_values)))
        ax.add_artist(ActionPlot)
        if self.goal_coord_features is not None:
            # plot the preferred coordinate actions
            coord_action_values = np.array([self.get_coord_action(self.getFeatureEncoding(l)) for l in locs])
            coord_action_prob = np.array([np.exp(self.coord_action_weight-peak_action_preference[i])/
                (np.sum(np.exp(self.get_actor_output(self.getFeatureEncoding(l))-peak_action_preference[i]))+
                    np.exp(self.coord_action_weight-peak_action_preference[i])) 
                for i,l in enumerate(locs)])
            plt.quiver(locs[:,0],locs[:,1],5*coord_action_prob*np.cos(np.take(list_actions,coord_action_values))
                ,5*coord_action_prob*np.sin(np.take(list_actions,coord_action_values)),color='g')
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
            eligibility_trace = np.zeros((self.featureLength,1))
            # get current location
            curr_loc = self.env.position[:,self.env.t]
            curr_loc_encoding = self.getFeatureEncoding(curr_loc)
            curr_x = self.get_X_coord(curr_loc_encoding)
            curr_y = self.get_Y_coord(curr_loc_encoding)
            while(not self.env.timeup() and not self.env.atgoal()):
                # select an action from the actor network
                A = self.get_action(curr_loc)
                # check is A is actor's preferred action or a_coord
                if A<self.numActions:
                    coord_action_taken = False
                else:
                    coord_action_taken = True
                    A = self.get_coord_action(curr_loc_encoding)
                # move the rat (agent)
                self.env.move(A)
                # get next location
                next_loc = self.env.position[:,self.env.t]
                next_loc_encoding = self.getFeatureEncoding(next_loc)
                next_x = self.get_X_coord(next_loc_encoding)
                next_y = self.get_Y_coord(next_loc_encoding)
                # Linear function approx is used, hence weight derivative is the current feature encoding
                weight_derivative = curr_loc_encoding.reshape(-1,1)
                # calculate the eligibility trace
                eligibility_trace = lamda*self.gamma*eligibility_trace + weight_derivative
                # if goal is reached, set reward to 1 otherwise 0. Calculate TD-error accordingly
                if self.env.atgoal():
                    reward = 1
                    self.goal_coord_features = next_loc_encoding.copy()
                    TD_error = reward - self.get_critic_output(curr_loc_encoding)
                else:
                    reward = 0
                    TD_error = self.gamma*self.get_critic_output(next_loc_encoding) - self.get_critic_output(curr_loc_encoding)
                self.criticWeights += alpha*TD_error*eligibility_trace      # update critic weights
                if coord_action_taken:
                    self.coord_action_weight += eligibility_trace.max()*alpha*TD_error      # update a_coord weight
                # else:
                self.actorWeights[:,A] += (alpha*TD_error*eligibility_trace).reshape(-1)      # update actor weights
                # Update the X and Y weights according to the del_x and del_y, which are calculated using env stepsize and action taken
                self.XWeights += 0.1*alpha*(-self.env.stepsize*np.cos(self.env.direction[A])+next_x-curr_x)*eligibility_trace
                self.YWeights += 0.1*alpha*(-self.env.stepsize*np.sin(self.env.direction[A])+next_y-curr_y)*eligibility_trace
                # check for Nans while debugging
                if np.isnan(self.criticWeights).any():
                    print('critic weights',self.get_critic_output(next_loc_encoding),self.get_critic_output(curr_loc_encoding),TD_error,eligibility_trace.min(),eligibility_trace.max())
                if np.isnan(self.actorWeights[:,A]).any():
                    print(self.get_critic_output(next_loc_encoding),self.get_critic_output(curr_loc_encoding),TD_error,eligibility_trace.min(),eligibility_trace.max())
                if verbose:
                    # if verbose, for each action, plot the place cell activations
                    # If goal is reached, print the TD-error and the critic weights and actor preferences learned
                    self.featureCode.plotPlaceCellActivations(curr_loc)
                    if reward!=0:
                        print(TD_error,eligibility_trace.shape)
                        # print('Critic weights and updates are '+('equal!' if (self.criticWeights==alpha*eligibility_trace).all() else 'not equal!'))
                        self.plot_critic_weights()
                        self.plot_X_estimates()
                        self.plot_Y_estimates()
                        self.plot_actor_preferences()
                # copy the next state (loc) variables the current state (loc) variables for the next step
                curr_loc = next_loc.copy()
                curr_loc_encoding = next_loc_encoding.copy()
                curr_x = next_x.copy()
                curr_y = next_y.copy()
            time_taken[e] = self.env.t      # note the time taken by the agent to complete the episode
            if verbose:
                # if verbose, for certain episodes plot the path taken and the estimated value function so far
                if e%1==0:
                    print(e,self.env.t,self.env.atgoal())
                    self.env.plotpath()
                    self.plot_value_function()
        return time_taken
                
if __name__=='__main__':
    np.random.seed(10)
    days = 6
    sessions = 3
    episodes = 500
    # maze = watermaze(T=60)
    # maze = RWMTask(T=60,days=days)
    maze = DMPTask(T=60,days=days)
    maze.startposition()
    AC = Coordinate_ActorCritic(env=maze,numCells=493,gamma=0.9)
    # time_arr = AC.TD_lambda(alpha=0.002,lamda=0.,episodes=20,verbose=True)
    time_mean = []
    time_std = []
    for d in tqdm(range(1,1+days*sessions)):
        time_arr = AC.TD_lambda(alpha=0.001,lamda=0.9,day=1+(d-1)//sessions,episodes=episodes)
        # AC.plot_X_estimates()
        # AC.plot_Y_estimates()
        # AC.plot_actor_preferences()
        # AC.plot_value_function()
        tqdm.write("{} {} {}".format(time_arr.mean(),time_arr.std(),AC.coord_action_weight))
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
    plt.xticks(tick_arr,tick_label_arr)
    plt.ylabel('Steps')
    plt.xlabel('Days')
    plt.show()