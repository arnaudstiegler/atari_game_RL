import gym
import numpy as np
import agent_mountaincar
import timeit


env_to_use = 'CartPole-v1'

# game parameters
env = gym.make(env_to_use)
#env._max_episode_steps = 1000
#print(env.action_space)

'''

env._max_episode_steps is set at 10000 which can lead to very long game.
The issue with that is that the agent gets stuck in suboptimal minimas (namely not moving)
To adress that, we will set it at 1000 

'''



#state_space = env.observation_space #Format: Box(250, 160, 3)
#action_space = env.action_space #Format: Discrete(3)

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

agent = agent_mountaincar.mcar_agent(state_space= state_space, action_space= action_space)
agent.Q.load_weights('mountain_car/dqn.h5')
agent.epsilon=0.0
agent.explore = 1000000000

reward_list = []
eps_length_list = []


#TODO: render into agent class
#TODO: Check all parameters for Network/Learning
#TODO: Check reward
#TODO: Check TD target

for ep in range(100):

    agent.check_learning(env, ep)

    print("---- Currently running episode " +str(ep))
    start = timeit.default_timer()

    total_reward = 0
    steps_in_ep = 0

    #agent.reinitialize_agent()

    # Initial state
    s_t = env.reset() #Observation is array (250, 160, 3)



    #state = process_obs(obs)#to create a batch with only one observation
    done=False
    agent.observe_phase = False
    #Max number of rounds for one episode
    while(done is False):

        if(agent.initial_move):
            # If it is the first move, we can't store anything in the memory
            action = agent.act(s_t.reshape((1, agent.state_space)))
            s_t1, reward, done, _info = env.step(action)
            #env.render()
            agent.initial_move = False

        elif(agent.observe_phase):
            #While we observe, we do not want to do replay_memory
            # take step
            action = agent.act(s_t.reshape((1, agent.state_space)))
            s_t1, reward, done, _info = env.step(action)
            if(agent.number_steps_done > agent.observe_steps):
                agent.observe_phase = False

        else:
            # take step
            action = agent.act(s_t.reshape((1, agent.state_space)))
            s_t1, reward, done, _info = env.step(action)
            agent.state = s_t1.reshape((1, agent.state_space))
            #agent.add_to_memory(agent.state,agent.previous_state,action,reward,done)
            #agent.experience_replay()




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