import gym
import numpy as np
import DQL
import timeit
import time
from keras.models import load_model


env_to_use = 'CartPole-v1'

# game parameters
env = gym.make(env_to_use)
#env.frameskip = 5 #We do the same action for the next 5 frames



'''

env._max_episode_steps is set at 10000 which can lead to very long game.
The issue with that is that the agent gets stuck in suboptimal minimas (namely not moving)
To adress that, we will set it at 1000 

'''

state_space = 4
action_space = 2


'''

env.step() -> returns array (state,reward,done?,_info)

Action State for Time pilot
action=1 -> going straight
action=2 -> going up no fire
action=3 -> going right no fire
action=4 -> going left no fire
'''

#We initialize our agent

agent = DQL.DQL_agent(state_space= state_space, action_space= action_space)
agent.Q = load_model('results/my_model.h5')
#agent.Q.load_weights('results/dqn.h5')
agent.epsilon=0.05
agent.explore = 1

reward_list = []
eps_length_list = []

for ep in range(100):

    print("---- Currently running episode " +str(ep))
    start = timeit.default_timer()

    total_reward = 0
    steps_in_ep = 0

    #agent.reinitialize_agent()

    # Initial state
    s_t = env.reset() #Observation is array (128)

    s_t = s_t.reshape(1, s_t.shape[0])  #to have (1,128) for Keras


    #state = process_obs(obs)#to create a batch with only one observation
    done=False

    #Max number of rounds for one episode
    while(done is False):
        time.sleep(.01)
        if(agent.initial_move):
            # If it is the first move, we can't store anything in the memory

            action = agent.act(s_t)
            new_state, reward, done, _info = env.step(action)

            s_t1 = new_state.reshape(1, new_state.shape[0])

            agent.state = s_t1
            env.render()
            agent.initial_move = False



        else:
            # take step
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

