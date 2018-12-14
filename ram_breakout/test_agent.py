from utils import normalize
import numpy as np
import DQL
import timeit
import time
from keras.models import load_model
import atari_wrapper

env_to_use = 'Breakout-ram-v4'
# game parameters
env = atari_wrapper.make_atari(env_to_use)
env = atari_wrapper.wrap_deepmind(env,episode_life=True, clip_rewards=False, frame_stack=False, scale=True)


#state_space = env.observation_space #Format: Box(250, 160, 3)
#action_space = env.action_space #Format: Discrete(3)
state_space = 128
action_space = 4



#We initialize our agent

agent = DQL.DQL_agent(state_space= state_space, action_space= action_space)
agent.Q = load_model('results/my_model.h5')
agent.epsilon=0.05
agent.explore = 1

reward_list = []
eps_length_list = []

for ep in range(100):

    print("---- Currently running episode " +str(ep))
    start = timeit.default_timer()

    total_reward = 0
    steps_in_ep = 0

    s_t = env.reset()  # Observation is array (128)

    # In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0])  # 1*80*80*4

    done=False

    #Max number of rounds for one episode
    while(done is False):
        time.sleep(0.1)


        action = agent.act(s_t)
        new_state,reward,done,_info = env.step(action)
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