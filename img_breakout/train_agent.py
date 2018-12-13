import gym
from DQL import DQL_agent
import timeit
import numpy as np
import atari_wrapper

env_to_use = 'BreakoutDeterministic-v4'

# game parameters
#env = gym.make(env_to_use)
env = atari_wrapper.make_atari(env_to_use)
env = atari_wrapper.wrap_deepmind(env,episode_life=True, clip_rewards=False, frame_stack=True, scale=True)
#env.frameskip = 5 #We do the same action for the next 4 frames
#env._max_episode_steps=1000

state_space = env.observation_space #Format: Box(250, 160, 3)
action_space = env.action_space #Format: Discrete(3)

print(state_space)
print(action_space)

state_space = (4,84,84) # Using the DeepMind wrapper stacking
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

    total_reward = 0
    steps_in_ep = 0

    #agent.check_learning(env, ep)

    done = False

    # Initial state
    s_t = np.array(env.reset()) #Observation is array (128)
    s_t = s_t
    start = timeit.default_timer()

    while(done is False):
        #env.render()

        #Pycharm refers to the base DQL model but when running it from the console, it uses /ram_breakout/DQL
        #if(agent.time_steps % agent.update_target_Q == 0 and agent.use_target):
            #print("update target network")
            #agent.target_Q.set_weights(agent.Q.get_weights())

        action = agent.act(s_t)
        new_state,reward,done,_info = env.step(1)
        s_t1 = np.array(new_state)

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

