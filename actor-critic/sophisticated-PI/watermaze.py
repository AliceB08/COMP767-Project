import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from tqdm.notebook import tqdm
####################################################################################################
# watermaze module
####################################################################################################
class watermaze(object):
    
    """
    This class defines a set of functions for simulating a rat moving in a water-maze.
    
    For the purposes of this assignment, you should be using the move function to 
    determine the next state of the environment at each time-step of the simulation.
    
    See the demo of its usage after the module code.
    """
    
    ####################################################################
    # the initialization function, measurements are in cm
    def __init__(self, pool_radius=60, platform_radius=10, platform_location=np.array([25,25]), 
                 stepsize=5.0, momentum=0.2, T=60):
        
        """
        The init function for the watermaze module.
        
        - The pool_radius argument specifies the radius of the pool.
        
        - The platform_radius argument specifies the radius of the platform.
        
        - The platform_location argument specifies the location of the platform centre.
        
        - The stepsize argument specifies how far the rat moves in one step.
        
        - The momentum argument specifies the ratio of old movement to new movement direction (i.e. 
        momentum = 0 means all new movement, momentum = 1 means all old movement, otherwise a mix.
        
        - The T argument is the maximum time for a trial in the pool.

        
        """
        
        # store the given info
        self.radius            = pool_radius
        self.platform_radius   = platform_radius
        self.platform_location = platform_location
        self.stepsize          = stepsize
        self.momentum          = momentum
        self.T                 = T
        
        # a dictionary for calculating directions
        self.direction = {
            0:  np.pi/2, #north
            1:  np.pi/4, #north-east
            2:  0, #east
            3:  7*np.pi/4, #south-east
            4:  3*np.pi/2, #south
            5:  5*np.pi/4, #south-west
            6:  np.pi, #west
            7:  3*np.pi/4, #north-west
        }
        
        # initialize the dynamic variables
        self.position = np.zeros((2,T))
        self.t        = 0
        self.prevdir  = np.zeros((2,))
        
    ####################################################################
    # for updating the rat's position in the pool
    def move(self, A, random_trajectory=False):
        """
        Updates the simulated rat's position in the water-maze environment by moving it in the 
        specified direction. 

        - The argument A is the last selected action, and must be an integer from 0-7, with 0 indicating N, 
        1 indicating NE, etc. 
        """
        trajectory = {}

        # check the A argument
        if (not np.isin(A, np.arange(8))):
            print('Error: The argument A must be an integer from 0-7, indicating which action was selected.')

        # determine the vector of direction of movement
        angle = self.direction[A]
        newdirection = np.array([np.cos(angle), np.sin(angle)])

        # add in momentum to reflect actual swimming dynamics (and normalize, then multiply by stepsize)
        direction = (1.0 - self.momentum)*newdirection + self.momentum*self.prevdir
        direction = direction/np.sqrt((direction**2).sum())
        direction = direction*self.stepsize
        if random_trajectory:
            direction = direction*np.mod(np.random.exponential(2.5),2.5)    # exponential distribution on velocity with max clipped of at 2*self.stepsize

        # update the position, prevent the rat from actually leaving the water-maze by having it "bounce" off the wall 
        [newposition, direction] = self.poolreflect(self.position[:, self.t] + direction)

        # if we're now at the very edge of the pool, move us in a little-bit
        if (np.linalg.norm(newposition) == self.radius):
            newposition = np.multiply(np.divide(newposition, np.linalg.norm(newposition)), (self.radius - 1))

        angular_velocity = np.arctan2(direction[1],direction[0])-np.arctan2(self.prevdir[1],self.prevdir[0])
        velocity = np.sum((newposition-self.position[:, self.t])**2)

        #fill the trajectory dictionary
        trajectory['prev_pos'] = self.position[:, self.t]
        trajectory['prev_hd'] = [np.arctan2(self.prevdir[1], self.prevdir[0])]
        trajectory['ego_vel'] = [velocity, math.sin(angular_velocity), math.cos(angular_velocity)]
        trajectory['target_pos'] = self.position[:, self.t + 1]
        trajectory['target_hd'] = np.arctan2(direction[1], direction[0])

        # update the position, time (and previous direction)
        self.position[:, self.t+1] = newposition
        self.t                    = self.t + 1
        self.prevdir              = direction

        return trajectory
        
    ####################################################################
    # for bouncing the rat off the wall of the pool
    def poolreflect(self, newposition):
        
        """
        The poolreflect function returns the point in space at which the rat will be located if it 
        tries to move from the current position to newposition but bumps off the wall of the pool. 
        If the rat would not bump into the wall, then it simply returns newposition. The function 
        also returns the direction the rat will be headed.
        """

        # determine if the newposition is outside the pool, if not, just return the new position
        if (np.linalg.norm(newposition) < self.radius):
            refposition  = newposition
            refdirection = newposition - self.position[:,self.t]

        else:

            # determine where the rat will hit the pool wall
            px = self.intercept(newposition)
            
            # get the tangent vector to this point by rotating -pi/2
            tx = np.asarray(np.matmul([[0, 1], [-1, 0]],px))

            # get the vector of the direction of movement
            dx = px - self.position[:,self.t]
            
            # get the angle between the direction of movement and the tangent vector
            theta = np.arccos(np.matmul((np.divide(tx,np.linalg.norm(tx))).transpose(),(np.divide(dx,np.linalg.norm(dx))))).item()

            # rotate the remaining direction of movement vector by 2*(pi - theta) to get the reflected direction
            ra = 2*(np.pi - theta)
            refdirection = np.asarray(np.matmul([[np.cos(ra), -np.sin(ra)], [np.sin(ra), np.cos(ra)]],(newposition - px)))

            # get the reflected position
            refposition = px + refdirection

        # make sure the new position is inside the pool
        if (np.linalg.norm(refposition) > self.radius):
            refposition = np.multiply((refposition/np.linalg.norm(refposition)),(self.radius - 1))

        return [refposition, refdirection]
    
    ####################################################################
    # for checking when/where the rat hits the edge of the pool
    def intercept(self,newposition):
        
        """
        The intercept function returns the point in space at which the rat will intercept with the pool wall 
        if it is moving from point P1 to point P2 in space, given the pool radius.
        """
        
        # for easy referencing, set p1 and p2
        p1 = self.position[:,self.t]
        p2 = newposition

        # calculate the terms used to find the point of intersection
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dr = np.sqrt(np.power(dx,2) + np.power(dy,2))
        D  = p1[0]*p2[1] - p2[0]*p1[1]
        sy = np.sign(dy)
        if (sy == 0):
            sy = 1.0
            
        # calculate the potential points of intersection
        pp1 = np.zeros((2,))
        pp2 = np.zeros((2,))

        pp1[0] = (D*dy + sy*dx*np.sqrt((np.power(self.radius,2))*(np.power(dr,2))-np.power(D,2)))/(np.power(dr,2))
        pp2[0] = (D*dy - sy*dx*np.sqrt((np.power(self.radius,2))*(np.power(dr,2))-np.power(D,2)))/(np.power(dr,2))
        pp1[1] = (-D*dx + np.absolute(dy)*np.sqrt((np.power(self.radius,2))*(np.power(dr,2))-np.power(D,2)))/(np.power(dr,2))
        pp2[1] = (-D*dx - np.absolute(dy)*np.sqrt((np.power(self.radius,2))*(np.power(dr,2))-np.power(D,2)))/(np.power(dr,2))

        # determine which intersection point is actually the right one (whichever is closer to p2)
        if np.linalg.norm(p2 - pp1) < np.linalg.norm(p2 - pp2):
            px = pp1

        else:
            px = pp2
        
        return px
    
    ####################################################################
    # sets the start position of the rat in the pool
    def startposition(self,random_trajectory=False):

        # select a random location from the main cardinal axes and calculate it's vector angle
        condition = 2*np.random.randint(0,4)
        # condition = 4
        angle = self.direction[condition]
        if random_trajectory:
            angle = np.random.uniform(0,2*np.pi)

        self.position[:,0] = np.asarray([np.cos(angle), np.sin(angle)]) * (self.radius - 1)
        
    ####################################################################
    # plot the most recent path of the rat through the pool
    def plotpath(self, live_plot=False):
        
        # create the figure 
        fig = plt.figure('path')
        ax = fig.gca()

        # create the pool perimeter
        pool_perimeter = plt.Circle((0, 0), self.radius, fill=False, color='b', ls='-')
        ax.add_artist(pool_perimeter)

        # create theplatform
        platform = plt.Circle(self.platform_location, self.platform_radius, fill=False, color='r', ls='-')
        ax.add_artist(platform)

        # plot the path
        plt.plot(self.position[0,0:self.t],self.position[1,0:self.t], color='k', ls='-')

        # plot the final location and starting location
        plt.plot(self.position[0,0],self.position[1,0],color='b', marker='o',markersize=4, markerfacecolor='b')
        plt.plot(self.position[0,self.t-1],self.position[1,self.t-1],color='r', marker='o',markersize=6, markerfacecolor='r')

        # adjust the axis
        ax.axis('equal')
        ax.set_xlim((-self.radius-50, self.radius+50))
        ax.set_ylim((-self.radius-50, self.radius+50))
        plt.xticks(np.arange(-self.radius, self.radius+20, step=20))
        plt.yticks(np.arange(-self.radius, self.radius+20, step=20))
        ax.set_xlabel('X Position (cm)')
        ax.set_ylabel('Y Position (cm)')

        # turn on the grid
        plt.grid(True)
        plt.tight_layout()

        # show the figure
        if live_plot:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            plt.show()
        
    ####################################################################
    # checks whether the time is up
    def timeup(self):
        
        """
        Returns true if the time for the trial is finished, false otherwise.
        """
        
        return self.t > (self.T - 2)
    
    ####################################################################
    # checks whether the rat has found the platform
    def atgoal(self):
        
        """
        Returns true if the rat is on the platform, false otherwise.
        """
        
        return np.sqrt(np.sum((self.position[:,self.t] - self.platform_location)**2)) <= (self.platform_radius + 1)

    def update_platform_location(self,day):
        pass

class RMWTask(watermaze):
    def __init__(self,pool_radius=60, platform_radius=10, platform_location=np.array([25,25]), 
        stepsize=5.0, momentum=0.2, T=60,days=10,alternate_platform_location=None):
        watermaze.__init__(self,pool_radius=pool_radius, platform_radius=platform_radius, platform_location=platform_location, 
            stepsize=stepsize, momentum=momentum, T=T)
        assert not (self.platform_location==0).all(), "Platform at the center!"
        self.days = days
        self.alternate_platform_location = alternate_platform_location
        self.platform_swtiched = False

    def update_platform_location(self,day):
        if day>0.7*self.days and not self.platform_swtiched:
            if self.alternate_platform_location is None or (self.alternate_platform_location==self.platform_location).all():
                self.alternate_platform_location = - self.platform_location
            self.platform_location = self.alternate_platform_location
            self.platform_swtiched = True

class DMPTask(watermaze):
    def __init__(self,pool_radius=60, platform_radius=10, platform_location=np.array([25,25]), 
        stepsize=5.0, momentum=0.2, T=60,days=10,alternate_platform_location_seq=None):
        watermaze.__init__(self,pool_radius=pool_radius, platform_radius=platform_radius, platform_location=platform_location, 
            stepsize=stepsize, momentum=momentum, T=T)
        assert not (self.platform_location==0).all(), "Platform at the center!"
        if alternate_platform_location_seq is not None:
            assert len(alternate_platform_location_seq)==days-1, "Alternate platform location sequences length {} doesn't match remaining days {}".format(
                len(alternate_platform_location_seq),days-1)
        self.days = days
        self.original_platform_location = self.platform_location
        self.alternate_platform_location_seq = alternate_platform_location_seq
        if alternate_platform_location_seq is None:
            # self.rot_angle_seq = np.random.permutation(np.linspace(1,self.days,self.days)*2*np.pi/self.days)
            self.rot_angle_seq = 2*np.pi*np.random.uniform(0,1,(self.days,))
        self.platform_swtiched = np.zeros((self.days,))

    def update_platform_location(self,day):
        if self.platform_swtiched[day-1]:
            return
        if self.alternate_platform_location_seq is not None:
            self.platform_location = self.alternate_platform_location_seq[day-2]
        else:
            angle = self.rot_angle_seq[day-1]
            new_location = np.array([int(np.cos(angle)*self.original_platform_location[0]-np.sin(angle)*self.original_platform_location[1])
                ,int(np.sin(angle)*self.original_platform_location[0]+np.cos(angle)*self.original_platform_location[1])]) 
            self.platform_location = new_location
            # print(180*angle/np.pi,self.platform_location)
        self.platform_swtiched[day-1] = True
