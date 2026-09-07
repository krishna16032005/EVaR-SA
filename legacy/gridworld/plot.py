import csv
import os
import numpy as np
import matplotlib.pyplot as plt

directory_path = './logs/evar_reinforce/log1/'
filename = '1.csv'

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