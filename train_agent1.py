import gym
import numpy as np
import DQL
import timeit
from utils import process_obs
import skimage

env_to_use = 'Breakout-v0'

# game parameters
env = gym.make(env_to_use)
#env._max_episode_steps = 1000
#print(env.action_space)



'''

env._max_episode_steps is set at 10000 which can lead to very long game.
The issue with that is that the agent gets stuck in suboptimal minimas (namely not moving)
To adress that, we will set it at 1000 

'''



state_space = env.observation_space #Format: Box(250, 160, 3)
action_space = env.action_space #Format: Discrete(3)

#print(state_space)
#print(action_space)

state_space = 250,160,3
action_space = 4


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
agent.Q.load_weights('breakout/dqn.h5')
reward_list = []
eps_length_list = []

#TODO: render into agent class
#TODO: Check all parameters for Network/Learning
#TODO: Check reward
#TODO: Check TD target
#TODO: add epochs count
#TODO: add performance testing after each epoch

ep = 0

while(True):

    print("---- Currently running episode " +str(ep))
    start = timeit.default_timer()

    print('epsilon value: ' + str(agent.epsilon))

    total_reward = 0
    steps_in_ep = 0

    #agent.reinitialize_agent()

    agent.check_learning(env, ep)  # Returns false if the check is not processed

    done = False

    # Initial state
    obs = env.reset() #Observation is array (250, 160, 3)
    '''
    x_t = skimage.color.rgb2gray(obs)
    x_t = skimage.transform.resize(x_t, (80, 80))
    x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))
    x_t = x_t / 255.0
    x_t = x_t.reshape(1, x_t.shape[0], x_t.shape[1])

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=1) #1*80*80*4
    '''

    x_t = skimage.color.rgb2gray(obs)
    x_t = skimage.transform.resize(x_t,(80,80))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

    x_t = x_t / 255.0

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    #print (s_t.shape)

    #In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4


    #Max number of rounds for one episode
    while(done is False):

        if(agent.initial_move):
            # If it is the first move, we can't store anything in the memory
            action = agent.act(s_t)
            new_state, reward, done, _info = env.step(action)
            '''
            x_t1 = skimage.color.rgb2gray(new_state)
            x_t1 = skimage.transform.resize(x_t1, (80, 80))
            x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
            x_t1 = x_t1 / 255.0
            x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
            s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)
            '''

            x_t1 = skimage.color.rgb2gray(new_state)
            x_t1 = skimage.transform.resize(x_t1, (80, 80))
            x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

            x_t1 = x_t1 / 255.0



            x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x80x80x1
            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

            agent.state = s_t1
            #env.render()
            agent.initial_move = False

        elif(agent.observe_phase):
            #While we observe, we do not want to do replay_memory
            # take step
            action = agent.act(s_t)
            new_state, reward, done, _info = env.step(action)
            '''
            x_t1 = skimage.color.rgb2gray(new_state)
            x_t1 = skimage.transform.resize(x_t1, (80, 80))
            x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
            x_t1 = x_t1 / 255.0
            x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
            s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)
            '''

            x_t1 = skimage.color.rgb2gray(new_state)
            x_t1 = skimage.transform.resize(x_t1, (80, 80))
            x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

            x_t1 = x_t1 / 255.0

            x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x80x80x1
            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

            agent.state = s_t1
            agent.add_to_memory(agent.state, agent.previous_state, action, reward, done)
            if (agent.time_steps > agent.observe_steps):
                agent.observe_phase = False
                print("--- END OBSERVE PHASE ---")

        else:
            # take step
            action = agent.act(s_t)
            new_state,reward,done,_info = env.step(action)
            if (_info['ale.lives'] < 5):
                done = True
            '''
            x_t1 = skimage.color.rgb2gray(new_state)
            x_t1 = skimage.transform.resize(x_t1, (80, 80))
            x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
            x_t1 = x_t1 / 255.0
            x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
            s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)
            '''

            x_t1 = skimage.color.rgb2gray(new_state)
            x_t1 = skimage.transform.resize(x_t1, (80, 80))
            x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

            x_t1 = x_t1 / 255.0

            x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x80x80x1
            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

            agent.state = s_t1
            agent.add_to_memory(agent.state,agent.previous_state,action,reward,done)
            agent.experience_replay()




        #env.render()

        total_reward += reward
        steps_in_ep += 1
        s_t = s_t1
        agent.previous_state = s_t1

    print("total reward: " + str(total_reward))
    print("total number of steps: " + str(steps_in_ep))
    print("agent epsilon: " + str(agent.epsilon))
    reward_list.append(total_reward)
    eps_length_list.append(steps_in_ep)

    #We backup the weights
    agent.Q.save_weights('breakout/dqn.h5')

    ep+=1

    end = timeit.default_timer()
    print("Episode took " + str((end-start)) + " seconds")
    print("Currently at time step: " + str(agent.time_steps))

agent.Q.save_weights('breakout/dqn.h5')

