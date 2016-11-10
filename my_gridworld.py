from ipythonblocks import BlockGrid
import numpy as np
import time
from IPython import display
import matplotlib.pyplot as plt
import pandas as pd

class my_gridworld():
    
    def __init__(self):
        ### initialize grid, agent, obstacles, etc.,
        self.width = 6
        self.height = 6
        self.grid = BlockGrid(self.height,self.width, fill=(234, 123, 234))
        
        # decide on obstacle and goal locations
        self.obstacles = [[1,2],[3,4],[2,3],[2,1]]  # impenetrable obstacle locations                
        self.goal = [4,4]     # goal block
        self.player = [1,4]   # initial location player

        # enumerate states based on obstacle locations
        self.states = []
        for i in range(self.grid.height):
            for j in range(self.grid.width):
                block = [i,j]
                if block not in self.obstacles and block not in self.goal:
                    self.states.append(str(i) + str(j))
                    
        # initialize Q^* matrix
        self.Q_star = np.zeros((self.grid.width**2 - len(self.obstacles),5))

        # initialize action choices
        self.action_choices = [[-1,0],[0,-1],[1,0],[0,1],[0,0]]

    def color_grid(self):                            
        # remake + recolor grid
        self.grid = BlockGrid(self.width, self.height, fill=(234, 123, 234))
        
        # color obstacles
        for i in range(len(self.obstacles)):
            self.grid[self.obstacles[i][0],self.obstacles[i][1]].red = 0

        # make and color goal
        self.grid[self.goal[0],self.goal[1]].green = 255
        self.grid[self.goal[0],self.goal[1]].red = 0
        self.grid[self.goal[0],self.goal[1]].blue = 0
        
        # color player location
        self.grid[self.player[0],self.player[1]].green = 0
        self.grid[self.player[0],self.player[1]].red = 0
        self.grid[self.player[0],self.player[1]].blue = 0
        
        self.grid.show()

    # make rewards array - 
    def make_rewards(self):
        # create reward matrix
        R = -1*np.ones((5,5))
        R[goal[0],goal[1]] = 1000
        for i in range(len(obstacles)):
            R[obstacles[i][0],obstacles[i][1]] = 0    
        
    ## Q-learning function
    def qlearn(self,gamma):
        num_episodes = 1000
        num_complete = 0
        
        # loop over episodes, for each run simulation and update Q
        for n in range(num_episodes):
            # pick random initialization - make sure its not an obstacle or goal
            obstical_free = 0
            loc = 0
            while obstical_free == 0:
                loc = [np.random.randint(self.grid.width),np.random.randint(self.grid.height)]
                if loc not in self.obstacles:
                    obstical_free = 1

            # update Q matrix while loc != goal
            steps = 0
            max_steps = 200
            while steps < max_steps and loc != self.goal:    
                    
                # choose action - left = 0, right = 1, up = 2, down = 3, 4 = stay still
                k = np.random.randint(5)  
                loc2 = [sum(x) for x in zip(loc, self.action_choices[k])] 
                
                # check that new location within grid boundaries, if trying to go outside boundary -- either way can't move there!  So just place -1 for this location in Q
                if loc2[0] > self.grid.width-1 or loc2[0] < 0 or loc2[1] > self.grid.height-1 or loc2[1] < 0 or loc2 in self.obstacles:
                    # don't move location
                    ind_old = self.states.index(str(loc[0]) + str(loc[1]))
                    self.Q_star[ind_old,k] = -1
                else:
                    # we have a valid movement, if new state is goal set reward to 1000, otherwise set it to 0
                    r_k = 0
                    if loc2 == self.goal:
                        r_k = int(1000)

                    # update Q* matrix
                    ind_old = self.states.index(str(loc[0]) + str(loc[1]))
                    ind_new = self.states.index(str(loc2[0]) + str(loc2[1]))
                    self.Q_star[ind_old,k] = r_k + gamma*max(self.Q_star[ind_new,:])
                    
                    # update current location - one we just moved too
                    loc = loc2
                    
                # update counter
                steps+=1
        print 'q-learning process complete'
                
    # print out
    def show_qmat(self):        
        df = pd.DataFrame(self.Q_star,columns=['up','down','left','right','still'], index=self.states)
        print df.round(3) 
            
    # animate the player based on completed Q-learning cycle
    def animate_movement(self,loc):
        # show movement based on an initial 
        self.player = loc # initial agent location
        self.color_grid()
        time.sleep(0.3)
        display.clear_output(wait=True)

        # if you chose an invalid starting position, break out and try again
        if loc in self.obstacles or loc == self.goal or loc[0] > self.grid.width-1 or loc[0] < 0 or loc[1] > self.grid.height-1 or loc[1] < 0:
            print 'initialization is an obstacle or goal, initialize again'
        else:  
            # now use the learned Q* matrix to run from any (valid) initial point to the goal
            count = 0
            max_count = self.grid.width*self.grid.height
            while count < max_count:
                # find next state using max Q* value
                ind_old = self.states.index(str(self.player[0]) + str(self.player[1]))
                
                # find biggest value in Q* and determine block location
                action_ind = np.argmax(self.Q_star[ind_old,:])
                action = self.action_choices[action_ind]
                
                # move player to new location and recolor
                self.player = [sum(x) for x in zip(self.player,action)] 

                # clear current screen for next step
                self.color_grid()
                time.sleep(0.3)
                if self.player == self.goal:
                    break
                display.clear_output(wait=True)
                count+=1