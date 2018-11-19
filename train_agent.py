import gym
import numpy as np
import DQL
import timeit
from utils import process_obs
import skimage

env_to_use = 'Skiing-v0'

# game parameters
env = gym.make(env_to_use)
env._max_episode_steps = 1000


'''

env._max_episode_steps is set at 10000 which can lead to very long game.
The issue with that is that the agent gets stuck in suboptimal minimas (namely not moving)
To adress that, we will set it at 1000 

'''



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



#TODO: render into agent class
#TODO: Check all parameters for Network/Learning
#TODO: Check reward
#TODO: Check TD target

for ep in range(70):

    print("---- Currently running episode " +str(ep))
    start = timeit.default_timer()

    total_reward = 0
    steps_in_ep = 0



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
            #env.render()
            agent.initial_move = False



        elif(agent.observe_phase):
            #While we observe, we do not want to do replay_memory
            # take step
            action = agent.act(state)
            new_state, reward, done, _info = env.step(action)
            agent.state = process_obs(new_state)
            agent.add_to_memory(agent.state, agent.previous_state, action, reward, done)
            if(agent.time_steps > 5000):
                agent.observe_phase = False


        else:
            # take step
            action = agent.act(state)
            new_state,reward,done,_info = env.step(action)
            agent.state = process_obs(new_state)
            agent.add_to_memory(agent.state,agent.previous_state,action,reward,done)
            agent.experience_replay()


        #env.render()

        total_reward += reward
        steps_in_ep += 1
        agent.previous_state = agent.state

    reward_list.append(total_reward)

    #We backup the weights
    agent.Q.save_weights('dqn_1.h5')
    #We backup the rewards
    np.savetxt("rewards", reward_list)

    end = timeit.default_timer()
    print("Episode took " + str((end-start)) + " seconds")

reward_list = np.array(reward_list)
agent.Q.save_weights('dqn.h5')
np.savetxt("rewards",reward_list)
