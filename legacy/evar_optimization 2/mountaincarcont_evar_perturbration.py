import gym
import numpy as np
from matplotlib import pyplot as plt
import sys
import csv
import os

sys.path.insert(1, '/spsa-main')
from spsa import _spsa, iterator

import riskfolio as rp

seed = 42
alpha_val = 0.9
print("alpha = ", alpha_val)
print("seed = ", seed)
np.random.seed(seed)

def calculate_evar(returns, percentile=0.05, solver='CLARABEL'):
    returns = np.array(returns)
    neg_returns = np.array(-returns)
    res = rp.RiskFunctions.EVaR_Hist(neg_returns, percentile)
    return res

def calculate_cvar(tau ,returns, alpha=0.9):
    sorted_returns = np.sort(returns)
    #print("<<< ", tau*alpha)
    VaR = sorted_returns[int(tau*alpha)]#np.percentile(sorted_returns, tau * (alpha))
    CVaR = np.mean(sorted_returns[sorted_returns > VaR])
    return VaR, CVaR

class GaussianPolicy:
    """
    A simple Gaussian policy for continuous actions.
    """
    def __init__(self, state_dim, weights = None):
        """
        Initializes the policy with random weights for mean and standard deviation.

        Args:
            state_dim: The dimensionality of the state space.
        """
        self.state_dim = state_dim
        self.mean_weight = np.array([None] * action_dim)
        self.std_weight = np.array([None] * action_dim)
        if weights is not None:

            for i in range(action_dim):
                self.mean_weight[i] = weights[i][0]
                self.std_weight[i] = weights[i][1]

        else:

            for i in range(action_dim):
                self.mean_weight[i] = np.random.rand(state_dim)
                self.std_weight[i] = np.random.rand(state_dim)


    def sample_action(self, state):
        """
        Samples an action from a Gaussian distribution based on the state.

        Args:
            state: A numpy array representing the current state.

        Returns:
            A sampled action from the Gaussian distribution.
        """
        mean = [None] * action_dim
        std = [None] * action_dim
        action = [None] * action_dim
        for i in range(action_dim):

            mean[i] = np.dot(state, self.mean_weight[i])
            std[i] = np.exp(np.dot(state, self.std_weight[i]))
            action[i] = np.clip(np.random.normal(mean[i], std[i]), -1.0, 1.0)

        return action

    def update_weights(self, reward, new_state, old_state, learning_rate, discount_factor):
        """
        Updates the mean and standard deviation weights based on reward.

        Args:
            reward: The reward received from the environment.
            new_state: The new state after taking an action.
            old_state: The previous state before taking an action.
            learning_rate: The learning rate for updating weights.
            discount_factor: The discount factor for future rewards.
        """
        td_error = reward - discount_factor * np.dot(new_state, self.mean_weight)
        # Update weights based on temporal difference error and state features
        self.mean_weight += learning_rate * td_error * old_state
        self.std_weight += learning_rate * td_error * old_state * (new_state - old_state)

    def update_parameters(self, new_weights):


        c = np.asarray(self.mean_weight)
        d = np.asarray(self.std_weight)

        for i in range(action_dim):
            assert new_weights[i][0].shape == self.mean_weight[i].shape, "Shape mismatch in update_parameters"
            assert new_weights[i][1].shape == self.std_weight[i].shape, "Shape mismatch in update_parameters"
            # assert new_weights[i].shape == self.std_weight.shape, "Shape mismatch in update_parameters"
            c[i] = new_weights[i][0]
            d[i] = new_weights[i][1]

        self.mean_weight = c
        self.std_weight = d


def objective_function(params : np.ndarray) -> float:
    global pvar
    global alpha_val
    c = np.asarray(policy_cont.mean_weight)
    d = np.asarray(policy_cont.std_weight)
    for i in range(action_dim):
        c[i] = params[i][0]
        d[i] = params[i][1]
    policy_cont.mean_weight = c
    policy_cont.std_weight = d

    alpha = alpha_val
    num_episodes = 200
    returns = []
    total_reward = []
    # Training loop (number of episodes)
    for _ in range(num_episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        R = 0
        while not terminated and not truncated:
            action = np.array(policy_cont.sample_action(state),dtype = "float32")

            # adding noise to action
            noise = [None] * action_dim
            for i in range(action_dim):
                noise[i] = np.random.normal(0, 1 - alpha)

            # print("noise - ", noise)
            # print("action - ", action)
            action = action + np.array(noise)
            action = np.clip(action, -1, 1)
            # print("action + noise - ", action)

            new_state, reward, terminated, truncated, _  = env.step(action)
            R = R + reward
            state = new_state

        returns.append(R)

    evar, _ = calculate_evar(returns, alpha)
    # print(evar)

    # var, cvar = calculate_cvar(num_episodes, returns, alpha)
    return evar

# Define hyperparameters
learning_rate = 0.01
discount_factor = 0.9
print("learning_rate - ", learning_rate)
print("discount_factor - ", discount_factor)

# Create the mountain car environment
# env = gym.make("MountainCarContinuous-v0",render_mode= "human")

env = gym.make('MountainCarContinuous-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]



params = None

# Create the Gaussian policy object
policy_cont = GaussianPolicy(state_dim, params)



pvar = False


def main():
    global pvar
    global policy
    f = objective_function
    x = []
    max_iterations = 1000
    for _ in range(action_dim):
        x.append([np.random.rand(state_dim), np.random.rand(state_dim)])

    objective_values = []
    directory_path = './tests/mountaincar/alpha_0_9/'
    filename = 'evar4.csv'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    try:
        cnt = 0
        for variables in iterator.maximize(f, x, lr=0.01, adam=True):
            print("Iteration [ ", cnt, " ]")
            pvar = True
            # print("weights : ", variables['x'])
            obj_val = f(variables['x'])
            print(" obj. val : ", obj_val)
            objective_values.append(obj_val)
            pvar = False
            policy_cont.update_parameters(variables['x'])
            cnt += 1
            sys.stdout.flush()
            if cnt==max_iterations:
                break

            if cnt % 100 == 0:
                with open(directory_path + filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(objective_values)
                    csvfile.flush()
                print(f"checkpoint for objective values, iteration: {cnt}. Saved at ", directory_path, filename)


    except KeyboardInterrupt:
        sys.exit()



    with open(directory_path + filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(objective_values)
    print("objective values stored in csv file - ", directory_path, filename)

if __name__ == "__main__":
    main()
