from DQL import DQL_agent
import timeit
import atari_wrapper

env_to_use = 'Breakout-ram-v4'
# game parameters
env = atari_wrapper.make_atari(env_to_use)
env = atari_wrapper.wrap_deepmind(env,episode_life=True, clip_rewards=False, frame_stack=False, scale=True)


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


ep = 0

while(True):

    total_reward = 0
    steps_in_ep = 0

    agent.check_learning(env, ep)

    done = False

    # Initial state
    s_t = env.reset() #Observation is array (128)
    s_t = s_t.reshape(1, s_t.shape[0])


    start = timeit.default_timer()
    #Max number of rounds for one episode
    while(done is False):

        #Pycharm refers to the base DQL model but when running it from the console, it uses /ram_breakout/DQL
        if(agent.time_steps % agent.update_target_Q == 0 and agent.use_target):
            print("update target network")
            agent.target_Q.set_weights(agent.Q.get_weights())

        action = agent.act(s_t)

        new_state,reward,done,_info = env.step(action)
        s_t1 = new_state
        s_t1 = s_t1.reshape(1, s_t1.shape[0])

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

