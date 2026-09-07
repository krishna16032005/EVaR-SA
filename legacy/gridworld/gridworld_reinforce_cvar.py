# %% [markdown]
# # Imports

# %%
import numpy as np

from collections import deque

import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Gym
# import gym
# import gym_pygame
#import gymnasium as gym

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %%
class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=0)

    def act(self, state):
        state = torch.tensor([state], dtype=torch.float32).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

# %%
torch.autograd.set_detect_anomaly(True)

# %% [markdown]
# # GridWorld Environment

# %%
# https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/gridworld.py

class GridWorld(object):
    def __init__(self, m, n, obstacles, flag):
        """
        m: number of rows
        n: number of columns
        obstacles: list of (x,y) coordinates of obstacles
        """
     
        self.grid = np.zeros((m,n))
        self.m = m
        self.n = n
        self.stateSpace = [i for i in range(self.m*self.n)]
        self.obstacles = obstacles
        self.stateSpace.remove(self.m*self.n-1)
        self.stateSpacePlus = [i for i in range(self.m*self.n)]
        self.actionSpace = {0: -self.m, 1: self.m, 2: -1, 3: 1}
        self.possibleActions = [0, 1, 2, 3]
        self.addObstacles(obstacles)
        # if flag:
        #     self.screen, self.clock = main(m, n, obstacles, (0,0))
        self.agentPosition = 0
        
    def isTerminalState(self, state):
        return state in self.stateSpacePlus and state not in self.stateSpace
    
    def addObstacles(self, obstacles):
        for obstacle in obstacles:
            self.grid[obstacle[0], obstacle[1]] = -1
                
    def getAgentRowAndColumn(self):
        agentX = self.agentPosition // self.m
        agentY = self.agentPosition % self.n
        return agentX, agentY
    
    def setState(self, state):
        x, y = self.getAgentRowAndColumn()
        self.grid[x,y] = 0
        self.agentPosition = state
        x, y = self.getAgentRowAndColumn()
        self.grid[x,y] = 1
        
    def offGridMove(self, newState, oldState):
   
        if newState not in self.stateSpacePlus:     # if we move into a row not in the grid
            return True
        # if we're trying to wrap around to next row
        elif oldState % self.m == 0 and newState % self.m == self.m - 1:
            return True
        elif oldState % self.m == self.m - 1 and newState % self.m == 0:
            return True
        else:
            return False
        
    def step(self, action):
       
        resultingState = self.agentPosition + self.actionSpace[action]
        # if self.isTerminalState(resultingState):
        #     print("Terminal state reached!")
        reward = -1 if not self.isTerminalState(resultingState) else 0

        result_x = resultingState // self.m
        result_y = resultingState % self.n
        pos = (result_x, result_y)

        if pos in self.obstacles:
            reward  = -2

        if not self.offGridMove(resultingState, self.agentPosition) and pos not in self.obstacles:
            self.setState(resultingState)
            agentX, agentY = self.getAgentRowAndColumn()
            #print(agentX, agentY)
            return resultingState, reward, self.isTerminalState(resultingState), None
        else:
            return self.agentPosition, reward, self.isTerminalState(self.agentPosition), None
        
    def reset(self):
        self.agentPosition = 0
        self.grid = np.zeros((self.m,self.n))
        self.addObstacles(self.obstacles)
        return self.agentPosition
    
    # def render(self):
    #     pygame.init()
    #     font = pygame.font.Font(None, 20)
    #     screen = self.screen
    #     clock = self.clock
    #     rows = self.m
    #     cols = self.n
    #     obstacles = self.obstacles
    #     SPEED = 30
 
    #     agentX, agentY = self.agentPosition // self.m, self.agentPosition % self.n
    #     agent_pos = (agentX * BLOCK_SIZE, agentY * BLOCK_SIZE)
    
    #     x, y = agent_pos  # Extract the current agent position

    #     # print(agentX, agentY)
    #     # print(x, y)
    #     screen.fill(WHITE)

    #     # Draw grid lines
    #     draw_grid_lines(screen, rows, cols)

    #     # Draw obstacles
    #     draw_obstacles(screen, obstacles)

    #     # Draw agent and target
    #     pygame.draw.rect(screen, RED, pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE))
    #     pygame.draw.rect(screen, GREEN, pygame.Rect((rows-1) * BLOCK_SIZE, (cols-1)*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    #     # Flip the display
    #     pygame.display.flip()
    #     clock.tick(SPEED)
        #pygame.quit()  # Quit Pygame when the loop exits

    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)
    

        

# %%
n_rows = 4
n_columns = 4
obstacles = [(0, 1), (3, 1)]
# obstacles = [(1,1)]

env = GridWorld(n_rows, n_columns, obstacles, False)

# %%
import riskfolio as rp

# %%
def calculate_evar(returns, percentile=0.05, solver='CLARABEL'):
    neg_returns = np.array(-returns)
    res = rp.RiskFunctions.EVaR_Hist(neg_returns, percentile)
    return res

# %%
from math import e
def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque()
    scores = []

    # simulating episodes
    for i_episode in range(1, n_training_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()


        # for t in range(max_t):
        for _ in range(max_t):
        # while True:
            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         pygame.quit()
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            #env.render()
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))


        #pygame.quit()
        # Line 6 of pseudocode: calculate the return
        returns = deque()
        n_steps = len(rewards)

        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft( gamma*disc_return_t + rewards[t] )


        # standardization of the returns is employed to make training more stable
        # eps is the smallest representable float, which is added to the standard deviation of the returns to avoid numerical instabilities
        eps = np.finfo(np.float32).eps.item()

        # new_returns, new_log_probs = clip_returns_and_log_probs(1, returns, saved_log_probs)
        # new_returns = torch.tensor(new_returns)
        # new_returns = (new_returns - new_returns.mean()) / (new_returns.std() + eps)


        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # temp = np.percentile(returns, 99)
        temp = returns.clone().detach()
        
        
        var = np.percentile(temp, 90)
        values_above_var = []
        for val in temp:
            if(val >= var):
                values_above_var.append(val)
        cvar = np.mean(values_above_var)


        # policy loss for normal reinforce
        policy_loss = []
        cnt = 0
        # for log_prob, disc_return in zip(new_log_probs, new_returns):
        for log_prob, disc_return in zip(saved_log_probs, returns):
            if(disc_return >= cvar):
                policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat([loss.unsqueeze(0) for loss in policy_loss]).sum()
        
        # print(cnt, len(temp))

        # print(policy_loss)


        # Line 8: PyTorch prefers gradient descent
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}, Episode Score: {:.2f}'.format(i_episode, np.mean(scores_deque), sum(rewards)))
    # pygame.quit()
    return scores

# %%
gridworld_policy = Policy(s_size=1, a_size=4,h_size=32).to(device)
gridworld_optimizer = optim.Adam(gridworld_policy.parameters(), lr=1e-2)

# %%
# scores = reinforce(gridworld_policy, gridworld_optimizer, n_training_episodes=1500, max_t=200, gamma=0.99, print_every=100)

# %% [markdown]
# # Main

# %%
num_batches = 20
all_batches_scores = []

# for i in range(num_batches):


# %%
for i in range(num_batches):
    gridworld_policy = Policy(s_size=1, a_size=4,h_size=32).to(device)
    gridworld_optimizer = optim.Adam(gridworld_policy.parameters(), lr=1e-2)
    scores = reinforce(gridworld_policy, gridworld_optimizer, n_training_episodes=1, max_t=200, gamma=0.9, print_every=100)
    all_batches_scores.append(scores)
    print('-------------------------------------')

# %%
print(len(all_batches_scores))

# %% [markdown]
# # Saving results

# %%

import csv
import os

directory_path = './logs/cvar_reinforce/Grid_4_4/log_80/'
filename = '1.csv'

if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    

# %%
with open(directory_path + filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(all_batches_scores)

# %% [markdown]
# # Plotting results

# %%
data = []

# Reading the CSV file and storing the data into a 2D array
with open(directory_path + filename, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        row = [float(x) for x in row]
        data.append(row)
        

# %%
rewards = np.array(data)
mean_rewards = np.mean(rewards, axis=0)
min_rewards = np.min(rewards, axis=0)
max_rewards = np.max(rewards, axis=0)

plt.figure(figsize=(10, 6))

# Plot mean rewards as a dark line
plt.plot(mean_rewards, color='blue', label='Mean Rewards')

# Shade the area between min and max rewards
plt.fill_between(np.arange(len(mean_rewards)), min_rewards, max_rewards, color='blue', alpha=0.2, label='Min-Max Range')

# Adding labels and title
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Mean Rewards with evar_reinforce')
plt.legend()
plt.grid(False)

# Show plot
plt.savefig(directory_path + 'res.png')


