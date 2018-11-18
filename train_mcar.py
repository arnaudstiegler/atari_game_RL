import gym

import agent_mountaincar
import timeit


env_to_use = 'MountainCar-v0'


env = gym.make(env_to_use)
'''

env._max_episode_steps is set at 10000 which can lead to very long game.
The issue with that is that the agent gets stuck in suboptimal minimas (namely not moving)
To adress that, we will set it at 1000 

'''



state_space = env.observation_space #Format: Box(250, 160, 3)
action_space = env.action_space #Format: Discrete(3)
print(action_space)
print(state_space)
state_space = 2
action_space = 3


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

    print('epsilon value: '+ str(agent.epsilon))
    total_reward = 0
    steps_in_ep = 0

    #agent.reinitialize_agent()

    # Initial state #Observation is array (250, 160, 3)


    agent.check_learning(env,ep) #Returns false if the check is not processed


    s_t = env.reset()
    done = False

    #s_t = s_t.reshape((2,1))
    #state = process_obs(obs)#to create a batch with only one observation

    #Max number of rounds for one episode
    while(done is False):

        if(agent.initial_move):
            # If it is the first move, we can't store anything in the memory
            action = agent.act(s_t.reshape((1,state_space)))
            s_t1, reward, done, _info = env.step(action)
            agent.state = s_t1.reshape((1,state_space))
            agent.initial_move = False

        elif(agent.observe_phase):
            #While we observe, we do not want to do replay_memory
            # take step
            action = agent.act(s_t.reshape((1,state_space)))
            s_t1, reward, done, _info = env.step(action)
            agent.state = s_t1.reshape((1,state_space))
            agent.add_to_memory(agent.state, agent.previous_state, action, reward, done)
            if(agent.time_steps > agent.observe_steps):
                agent.observe_phase = False
                print("--- END OBSERVE PHASE ---")

        else:
            # take step
            action = agent.act(s_t.reshape((1,state_space)))
            s_t1,reward,done,_info = env.step(action)
            agent.state = s_t1.reshape((1,state_space))
            agent.add_to_memory(agent.state,agent.previous_state,action,reward,done)
            agent.experience_replay()

        #env.render()


        total_reward += reward

        steps_in_ep += 1
        s_t = s_t1
        agent.previous_state = s_t1.reshape((1,state_space))

    #print("total reward: " + str(total_reward))
    #print("total number of steps: " + str(steps_in_ep))
    #print("agent epsilon: " + str(agent.epsilon))
    reward_list.append(total_reward)
    eps_length_list.append(steps_in_ep)

    #We backup the weights
    agent.Q.save_weights('mountain_car/dqn.h5')
    #We backup the rewards
    #np.savetxt("mountain_car/rewards", reward_list)
    #np.savetxt("mountain_car/steps", eps_length_list)

    end = timeit.default_timer()
    ep += 1
    print('number of steps: ' +str(steps_in_ep))
    print("Episode took " + str((end-start)) + " seconds")

agent.Q.save_weights('moutain_car/dqn.h5')

