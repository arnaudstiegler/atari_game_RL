import gym
import numpy as np
import DQL
import timeit
from keras.models import load_model
import atari_wrapper

env_to_use = 'BreakoutDeterministic-v4'

# game parameters
env = atari_wrapper.make_atari(env_to_use)
env = atari_wrapper.wrap_deepmind(env,episode_life=True, clip_rewards=False, frame_stack=True, scale=True)

state_space = env.observation_space #Format: Box(250, 160, 3)
action_space = env.action_space #Format: Discrete(3)

print(state_space)
print(action_space)

state_space = (4,84,84) # Using the DeepMind wrapper stacking
action_space = 4

#We initialize our agent

agent = DQL.DQL_agent(state_space= state_space, action_space= action_space)
agent.Q = load_model('results/my_model.h5')

agent.epsilon=1.0
agent.explore = 1

reward_list = []
eps_length_list = []

while(True):

    total_reward = 0
    steps_in_ep = 0

    #agent.check_learning(env, ep)

    done = False

    # Initial state
    s_t = np.array(env.reset()) #Observation is array (128)
    s_t = s_t
    start = timeit.default_timer()

    while(done is False):
        env.render()

        action = agent.act(s_t)
        new_state,reward,done,_info = env.step(action)
        s_t1 = np.array(new_state)

        agent.previous_state = s_t
        agent.state = s_t1


        total_reward += reward
        steps_in_ep += 1
        s_t = s_t1
        agent.previous_state = s_t1

    print("total reward: " + str(total_reward))
    print("total number of steps: " + str(steps_in_ep))
    print("agent epsilon: " + str(agent.epsilon))


    end = timeit.default_timer()
    print("Episode took " + str((end-start)) + " seconds")