import numpy as np

rewards = []
with open('rewards_dqn','r') as file:
    for line in file:
      rewards.append(float(line.replace('\n','')))

rewards = np.array(rewards)
print(np.mean(rewards))

length = []
with open('steps_dqn','r') as file:
    for line in file:
      length.append(float(line.replace('\n','')))

length = np.array(length)
print(np.mean(length))

'''
For random policy, we get:
mean of reward: 3416
mean of total steps: 2251

For dqnpolicy, we get:
mean of reward: 2899.0
mean of total steps: 2119.42

'''