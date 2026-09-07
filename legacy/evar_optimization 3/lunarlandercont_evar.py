import gym
import numpy as np
from matplotlib import pyplot as plt
import sys
import csv
import os

sys.path.insert(1, '/spsa-main')
from spsa import _spsa, iterator

import riskfolio as rp


env = gym.make("LunarLanderContinuous-v2")
seed = [42, 37, 8, 67, 52]
alpha_val = 0.1
learning_rate = 0.01
discount_factor = 0.9
num_episodes = 50
max_iterations = 20
num_batches = 5
directory_path = './tests/lunarlander/alpha_0_1/'

print("Seed:", seed)
print("Alpha Value:", alpha_val)
print("Learning Rate:", learning_rate)
print("Discount Factor:", discount_factor)
print("Number of Episodes:", num_episodes)
print("Max Iterations:", max_iterations)
print("Number of Batches:", num_batches)
print("Directory Path", directory_path)


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
avg_return_values = []


def calculate_evar(returns, alpha=0.1, solver='CLARABEL'):
    returns = np.array(returns)
    neg_returns = np.array(-returns)
    res = rp.RiskFunctions.EVaR_Hist(neg_returns, alpha)
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
    global avg_return_values
    global pvar
    global alpha_val

    # print("inside objective")

    c = np.asarray(policy_cont.mean_weight)
    d = np.asarray(policy_cont.std_weight)
    for i in range(action_dim):
        c[i] = params[i][0]
        d[i] = params[i][1]
    policy_cont.mean_weight = c
    policy_cont.std_weight = d
    # alpha = 0.1
    returns = []
    total_reward = []
    # Training loop (number of episodes)
    for _ in range(num_episodes):
        state = env.reset()[0]

        steps = 0
        # total_reward = 0
        terminated = False
        truncated = False
        R = 0
        while not terminated and not truncated:
            # action = [policy_cont.sample_action(state)]
            action = np.array(policy_cont.sample_action(state),dtype = "float32")
            # adding noise to action
            noise = [None] * action_dim
            for i in range(action_dim):
                noise[i] = np.random.normal(0, alpha_val)

            action = action + np.array(noise)
            action = np.clip(action, -1, 1)

            new_state, reward, terminated, truncated, _  = env.step(action)
            # print(reward)
            R = R + reward
            state = new_state
            steps += 1

            if steps >= 500:
                break

        returns.append(R)
    # print("outside for")
    evar,_ = calculate_evar(returns, alpha_val)
    avg_return = np.mean(returns)
    avg_return_values.append(avg_return)
    # print("end of objective")
    return evar


params = None

# Create the Gaussian policy object
policy_cont = GaussianPolicy(state_dim, params)

pvar = False

def main():
    global pvar
    global policy_cont
    global avg_return_values

    for batch in range(num_batches):

        # np.random.seed(seed[batch])
        # env.seed(seed[batch])
        # env.action_space.seed(seed[batch])

        print(f"batch = {batch + 1}")

        policy_cont = GaussianPolicy(state_dim, params)

        objective_values = []
        avg_return_values = []

        evar_filename = f"evar_obj_{batch+1}.csv"
        returns_filename = f"evar_ret_{batch+1}.csv"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        f = objective_function
        x = []
        for _ in range(action_dim):
            x.append([np.random.rand(state_dim), np.random.rand(state_dim)])

        try:
            cnt = 0
            for variables in iterator.maximize(f, x, lr=0.01, adam=True):
                print("Iteration [ ", cnt, " ]")
                pvar = True

                obj_val = f(variables['x'])
                print(f"Obj. val : {obj_val}, Avg. Ret : {avg_return_values[cnt]}")

                objective_values.append(obj_val)
                # avg_return_values.append(avg_ret)

                pvar = False
                policy_cont.update_parameters(variables['x'])
                cnt += 1
                sys.stdout.flush()
                if cnt==max_iterations:
                    break
                # print("working")

                if cnt % 100 == 0:
                    with open(directory_path + evar_filename, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(objective_values)
                        csvfile.flush()
                    print(f"checkpoint for objective values, iteration: {cnt}. Saved at ", directory_path, evar_filename)
                if cnt % 100 == 0:
                    with open(directory_path + returns_filename, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(avg_return_values)
                        csvfile.flush()
                    print(f"checkpoint for avg return values, iteration: {cnt}. Saved at ", directory_path, returns_filename)


        except KeyboardInterrupt:
            sys.exit()



        with open(directory_path + evar_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(objective_values)
            csvfile.flush()
        print(f"checkpoint for objective values, iteration: {cnt}. Saved at ", directory_path, evar_filename)

        with open(directory_path + returns_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(avg_return_values)
            csvfile.flush()
        print(f"checkpoint for objective values, iteration: {cnt}. Saved at ", directory_path, returns_filename)




if __name__ == "__main__":
    main()
