import gym
import numpy as np
import DQL


env_to_use = 'Skiing-v0'

# game parameters
env = gym.make(env_to_use)


print(env.observation_space)


state_space = env.observation_space #Format: Box(250, 160, 3)
action_space = env.action_space #Format: Discrete(3)




'''

env.step() -> returns array (state,reward,done?,_info)

Action State:
action=0 -> going straight
action=1 -> going right
action=2 -> going left
'''

#We initialize our agent

agent = DQL.DQL_agent(state_space= state_space, action_space= action_space)


for ep in range(1):

    total_reward = 0
    steps_in_ep = 0

    # Initial state
    observation = env.reset() #Observation is array (250, 160, 3)
    observation = observation.reshape((1,250, 160,3)) #to create a batch with only one observation
    done=True
    action = agent.act(observation)
    #env.render()
    print(env.step(0))
    #Max number of rounds for one episode
    while(done is False):

        #state = observation[0]
        #action = agent.act(state)

        # take step
        #next_observation, reward, done, _info = env.step(0)
        print(env.step(0))
        env.render()


        total_reward += 1

        # add this to experience replay buffer
        #experience.append((observation, action, reward, next_observation,
                           # is next_observation a terminal state?
         #                  0.0 if done else 1.0))



        #observation = next_observation

