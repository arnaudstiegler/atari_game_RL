from collections import deque
import numpy as np
import random
from keras.layers import Dense,Activation
from keras.models import Sequential
from keras.optimizers import Adam


class DQL_agent():

    def __init__(self,state_space,action_space):

        #Parameters of the environment
        self.state_space = state_space
        self.action_space = action_space

        #Learning parameters
        self.epsilon = 1.0
        #Number of time steps over which the agent will explore
        self.explore = 100000
        #Final value for epsilon (once exploration is finished)
        self.final_epsilon = 0.05

        self.epsilon_decay = (self.final_epsilon - self.epsilon)/(self.explore)
        self.gamma = 0.99

        #Memory replay parameters
        self.memory_size = 250000
        # Format of an experience is: (state,previous_state,action,reward)
        self.memory = deque([], self.memory_size)

        #Parameters for the CNN
        self.learning_rate_cnn = 0.001
        self.Q = self._build_model()
        self.use_target = True
        self.target_Q = self._build_model()
        #So that both networks start with the same weights
        self.target_Q.set_weights(self.Q.get_weights())

        #Parameters for the ongoing episode
        self.state = None
        self.previous_state = None
        self.action = None
        self.reward = None
        self.initial_move = True
        self.observe_phase = True
        self.observe_steps = 1 #Number of steps for observation (no learning)

        #Update the target network every ...
        self.update_target_Q = 1000
        #Max number of steps between two experience replays
        self.experience_nb_steps=1 #We update at each step
        #Size of a batch for experience replay
        self.experience_batch_size = 32
        #A counter of the number of steps since last experience replay
        self.time_steps = 0
        #Saving model
        self.backup = 5000

    def reinitialize_agent(self):
        #This function is actually useless
        self.initial_move = True
        self.time_steps = 0
        self.memory = deque([], self.memory_size)

    def _build_model(self):

        model = Sequential()
        model.add(Dense(128,input_dim=(self.state_space),activation='relu'))
        model.add(Dense(128,activation='relu'))
        model.add(Dense(self.action_space,activation='linear'))
        adam = Adam(lr=self.learning_rate_cnn)
        model.compile(loss='mse', optimizer=adam)

        return model


    def experience_replay(self):
        if (len(self.memory) > self.experience_batch_size):

            minibatch = random.sample(self.memory,self.experience_batch_size)

            state_batch = []
            target_batch = []

            for state,action,reward,next_state,done in minibatch:


                target = reward
                if not done:
                    target = reward + self.gamma * np.amax(self.target_Q.predict(next_state)[0])
                target_f = self.Q.predict(state)
                target_f[0][action] = target

                target_batch.append(target_f)
                state_batch.append(state)

            self.Q.fit(np.array(state_batch).reshape((self.experience_batch_size,self.state_space)),np.array(target_batch).reshape((self.experience_batch_size,self.action_space)), epochs=1, verbose=0)
            if self.epsilon > self.final_epsilon:
                self.epsilon += self.epsilon_decay



    def act(self,state):
        '''
                Implement Epsilon-Greedy policy

                Inputs:
                value: numpy ndarray
                        A vector of values of actions to choose from

                Outputs:
                action: int
                        Index of the chosen action
                '''

        # We have done an additional step
        self.time_steps += 1

        random_float = np.random.random()
        if (random_float < self.epsilon):
            return np.random.randint(self.action_space)
        else:
            # array of q(state,action) for all action
            return np.argmax(self.Q.predict(state)[0])





    def add_to_memory(self,state,action,reward,next_state,done):
        self.memory.append((state, action, reward, next_state, done))


    def check_learning(self,env,ep):
        if(ep % 100 == 0 ):
            print('---- CHECKING RESULTS ----')
            epsilon = self.epsilon
            timesteps = self.time_steps
            self.epsilon = 0.05

            rewards = []
            steps = []

            for it in range(20):

                s_t = env.reset()
                s_t,a,b,c= env.step(1) #Throwing the ball
                done=False
                total_reward = 0
                ep_steps = 0
                # In Keras, need to reshape
                self.state = s_t.reshape(1, s_t.shape[0])  # 1*80*80*4

                while(done==False):
                    action = self.act(self.state)
                    new_state, reward, done, _info = env.step(action)

                    self.state = new_state.reshape(1, new_state.shape[0])
                    total_reward += reward
                    ep_steps += 1


                steps.append(ep_steps)
                rewards.append(total_reward)

            with open('results/check_learning.txt','a') as file:
                file.write(str(ep) + "," + str(np.mean(rewards))+'\n')

            self.time_steps = timesteps
            self.epsilon = epsilon

            return True
        else:
            return False