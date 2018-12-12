import gym
from DQL import DQL_agent
import timeit
from utils import normalize
import numpy as np

env_to_use = 'Breakout-ram-v4'

# game parameters
env = gym.make(env_to_use)




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

    #print("---- Currently running episode " +str(ep))
    start = timeit.default_timer()

    #print('epsilon value: ' + str(agent.epsilon))

    total_reward = 0
    steps_in_ep = 0

    agent.check_learning(env, ep)

    done = False

    # Initial state
    s_t = env.reset() #Observation is array (128)

    #In Keras, need to reshape
    s_t = np.apply_along_axis(normalize, 0, s_t)
    s_t = s_t.reshape(1, s_t.shape[0])  # 1*80*80*4

    #We force the game to start directly
    s_t,a,b,c = env.step(1)

    s_t = np.apply_along_axis(normalize, 0, s_t)
    s_t = s_t.reshape(1, s_t.shape[0])

    #Max number of rounds for one episode
    while(done is False):
        #env.render()

        # FOR TRAINING, WE STOP EACH EPISODE AFTER ONE LIFE IS LOST
        if (env.env.ale.lives() < 5):
            break

        #Pycharm refers to the base DQL model but when running it from the console, it uses /ram_breakout/DQL
        if(agent.time_steps % agent.update_target_Q == 0 and agent.use_target):
            print("update target network")
            agent.target_Q.set_weights(agent.Q.get_weights())

        action = agent.act(s_t)
        new_state,reward,done,_info = env.step(action)
        s_t1 = new_state.reshape(1, new_state.shape[0])

        agent.previous_state = s_t
        agent.state = s_t1
        agent.add_to_memory(agent.previous_state,action,reward,agent.state,done)
        agent.experience_replay()

        if(agent.time_steps % agent.backup == 0):
            # We backup the model
            agent.Q.save('results/my_model.h5')

        total_reward += reward
        steps_in_ep += 1
        s_t = s_t1
        agent.previous_state = s_t1


    end = timeit.default_timer()
    avg_timestep_s = float(steps_in_ep) / (end-start)

    print("episode: {}, score = {}, time = {:0.2f}, epsilon = {}, timestep = {}".format(ep,total_reward,avg_timestep_s,agent.epsilon,agent.time_steps))
    reward_list.append(total_reward)
    eps_length_list.append(steps_in_ep)

    with open('results/epoch_rewards.txt','a') as file:
        file.write("{},{}".format(agent.time_steps,total_reward) + '\n')

    ep+=1

