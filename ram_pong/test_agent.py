import gym
import numpy as np
import DQL
import timeit
import time
from utils import normalize
from keras.models import load_model

env_to_use = 'Pong-ram-v4'

# game parameters
env = gym.make(env_to_use)
env.frameskip = 4 #We do the same action for the next 4 frames


#state_space = env.observation_space #Format: Box(250, 160, 3)
#action_space = env.action_space #Format: Discrete(3)
state_space = 128
action_space = 6


import gym
import numpy as np
import DQL
import timeit
import time
from keras.models import load_model



#We initialize our agent

agent = DQL.DQL_agent(state_space= state_space, action_space= action_space)
agent.Q = load_model('results/my_model.h5')
agent.epsilon=0.0
agent.explore = 1

reward_list = []
eps_length_list = []

for ep in range(100):

    print("---- Currently running episode " +str(ep))
    start = timeit.default_timer()

    total_reward = 0
    steps_in_ep = 0

    #agent.reinitialize_agent()

    s_t = env.reset()  # Observation is array (128)

    # In Keras, need to reshape
    s_t = np.apply_along_axis(normalize, 0, s_t)
    s_t = s_t.reshape(1, s_t.shape[0])  # 1*80*80*4

    done=False

    #Max number of rounds for one episode
    while(done is False):

        action = agent.act(s_t)
        new_state,reward,done,_info = env.step(action)
        new_state = np.apply_along_axis(normalize, 0, new_state)
        s_t1 = new_state.reshape(1, new_state.shape[0])
        agent.state = s_t1

        env.render()

        total_reward += reward
        steps_in_ep += 1
        s_t = s_t1
        agent.previous_state = s_t1

    print("total reward: " + str(total_reward))
    print("total number of steps: " + str(steps_in_ep))
    print("agent epsilon: " + str(agent.epsilon))
    reward_list.append(total_reward)
    eps_length_list.append(steps_in_ep)

    end = timeit.default_timer()
    print("Episode took " + str((end-start)) + " seconds")