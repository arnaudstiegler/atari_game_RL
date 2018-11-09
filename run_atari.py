import gym
import numpy as np
import DQL
from utils import process_obs

env_to_use = 'Skiing-v0'

# game parameters
env = gym.make(env_to_use)



#state_space = env.observation_space #Format: Box(250, 160, 3)
#action_space = env.action_space #Format: Discrete(3)

state_space = 250,160,3
action_space = 3


'''

env.step() -> returns array (state,reward,done?,_info)

Action State:
action=0 -> going straight
action=1 -> going right
action=2 -> going left
'''

#We initialize our agent

agent = DQL.DQL_agent(state_space= state_space, action_space= action_space)

reward_list = []


for ep in range(1000):

    print(ep)

    total_reward = 0
    steps_in_ep = 0

    agent.reinitialize_agent()

    # Initial state
    obs = env.reset() #Observation is array (250, 160, 3)
    state = process_obs(obs)#to create a batch with only one observation
    done=False

    #Max number of rounds for one episode
    while(done is False):

        if(agent.initial_move):
            # If it is the first move, we can't store anything in the memory
            action = agent.act(state)
            new_state, reward, done, _info = env.step(action)
            agent.state = process_obs(new_state)
            env.render()
            agent.initial_move = False

        else:
            # take step
            action = agent.act(state)
            new_state,reward,done,_info = env.step(action)
            agent.state = process_obs(new_state)
            agent.add_to_memory(agent.state,agent.previous_state,action,reward,done)
            agent.experience_replay()

        #print(agent.Q.predict(agent.state)[0])

        #env.render()

        total_reward += reward
        steps_in_ep += 1
        agent.previous_state = agent.state

    reward_list.append(total_reward)

reward_list = np.array(reward_list)
np.savetxt("rewards",reward_list)
