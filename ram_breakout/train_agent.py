import gym
import numpy as np
from DQL import DQL_agent
import timeit
from keras.models import load_model
import keras
from ram_breakout.utils import normalize

env_to_use = 'Breakout-ram-v0'

# game parameters
env = gym.make(env_to_use)
env.frameskip = 4 #We do the same action for the next 5 frames



'''

env._max_episode_steps is set at 10000 which can lead to very long game.
The issue with that is that the agent gets stuck in suboptimal minimas (namely not moving)
To adress that, we will set it at 1000 

'''



#state_space = env.observation_space #Format: Box(250, 160, 3)
#action_space = env.action_space #Format: Discrete(3)

state_space = 128 # Using the ram input
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

agent = DQL_agent(state_space= state_space, action_space= action_space)
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

    agent.check_learning(env, ep)

    done = False

    # Initial state
    s_t = env.reset() #Observation is array (128)

    #In Keras, need to reshape
    s_t = np.apply_along_axis(normalize, 0, s_t)
    s_t = s_t.reshape(1, s_t.shape[0])  #1*80*80*4


    #Max number of rounds for one episode
    while(done is False):
        #env.render()

        #Pycharm refers to the base DQL model but when running it from the console, it uses /ram_breakout/DQL
        if(agent.time_steps % agent.update_target_Q == 0 and agent.time_steps !=0):
            # serialize model to JSON
            model_json = agent.Q.to_json()
            with open("results/model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            agent.Q.save_weights("results/model.h5")
            print("Saved model to disk")

            # load json and create model
            json_file = open('results/model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = keras.models.model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("results/model.h5")
            agent.Q_target = loaded_model
            print("Loaded model from disk")

            #print("update target network")
            #agent.target_Q = load_model('results/my_model.h5')

        if(agent.initial_move):
            # If it is the first move, we can't store anything in the memory
            action = agent.act(s_t)
            new_state, reward, done, _info = env.step(action)
            new_state = np.apply_along_axis(normalize, 0, new_state)
            s_t1 = new_state.reshape(1, new_state.shape[0])

            agent.state = s_t1
            agent.initial_move = False

        elif(agent.observe_phase):
            #While we observe, we do not want to do replay_memory
            # take step
            action = agent.act(s_t)
            new_state, reward, done, _info = env.step(action)
            new_state = np.apply_along_axis(normalize, 0, new_state)
            s_t1 = new_state.reshape(1, new_state.shape[0])

            agent.state = s_t1
            agent.add_to_memory(agent.state, agent.previous_state, action, reward, done)
            if (agent.time_steps > agent.observe_steps):
                agent.observe_phase = False
                print("--- END OBSERVE PHASE ---")

        else:
            # take step
            action = agent.act(s_t)
            new_state,reward,done,_info = env.step(action)
            new_state = np.apply_along_axis(normalize, 0, new_state)
            s_t1 = new_state.reshape(1, new_state.shape[0])
            agent.state = s_t1
            agent.add_to_memory(agent.state,agent.previous_state,action,reward,done)
            agent.experience_replay()

        if(agent.time_steps % agent.backup == 0):
            # We backup the weights
            agent.Q.save('results/my_model.h5')
            agent.Q.save_weights('results/dqn.h5')



        #env.render()

        total_reward += reward
        steps_in_ep += 1
        s_t = s_t1
        agent.previous_state = s_t1

    print("total reward: " + str(total_reward))
    print("total number of steps: " + str(steps_in_ep))
    reward_list.append(total_reward)
    eps_length_list.append(steps_in_ep)

    ep+=1


    end = timeit.default_timer()
    avg_timestep_s = float(steps_in_ep) / (end-start)
    print("Episode took " + str((end-start)) + " seconds")
    print("Average computation speed was: " + str(round(avg_timestep_s)) + " steps per second")
    print("Currently at time step: " + str(agent.time_steps))

agent.Q.save_weights('dqn.h5')

