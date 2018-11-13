from collections import deque
import numpy as np
import random
from keras.layers import Convolution2D,Flatten,Dense,Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import binary_crossentropy,mean_squared_error
from keras.initializers import random_normal


class DQL_agent():

    def __init__(self,state_space,action_space):

        #Parameters of the environment
        self.state_space = state_space
        self.action_space = action_space

        #Learning parameters
        self.epsilon = 0.5
        self.epsilon_decay = 0.99
        self.final_epsilon = 0.0001
        self.gamma = 0.95

        #Memory replay parameters
        self.memory_size = 10000
        # Format of an experience is: (state,previous_state,action,reward)
        self.D = deque([],self.memory_size)

        #Parameters for the CNN
        self.learning_rate_cnn = 0.001
        self.Q = self._build_model()

        #Parameters for the ongoing episode
        self.state = None
        self.previous_state = None
        self.action = None
        self.reward = None
        self.initial_move = True
        self.observe_phase = True

        #Max number of steps between two experience replays
        self.experience_nb_steps=1 #We update at each step
        #Size of a batch for experience replay
        self.experience_batch_size = 32
        #A counter of the number of steps since last experience replay
        self.number_steps_done = 0

    def reinitialize_agent(self):
        #This function is actually useless
        self.initial_move = True
        self.number_steps_done = 0
        self.D = deque([], self.memory_size)

    def _build_model(self):

        img_channels, img_rows, img_cols = 3, 250, 160

        init = random_normal(mean=0.0, stddev=0.05, seed=None)
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(img_rows, img_cols, img_channels,),kernel_initializer=init))  # 250*160*3
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same',kernel_initializer=init))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same',kernel_initializer=init))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(128,kernel_initializer=init))
        model.add(Activation('relu'))
        model.add(Dense(3,activation='linear',kernel_initializer=init))
        adam = Adam(lr=self.learning_rate_cnn)
        model.compile(loss='mse', optimizer=adam)
        return model


    def experience_replay(self):
        #if(self.batch_learning % self.experience_nb_steps==0 and self.batch_learning >= self.experience_batch_size):
        if(self.number_steps_done > self.experience_batch_size):

            batch = random.sample(self.D, self.experience_batch_size)

            for i in range(0, len(batch)):
                state_t = batch[i][1]
                state_t1 = batch[i][0]
                action = batch[i][2]
                reward = batch[i][3]
                done = batch[i][4]

                if(done):
                    target=reward
                else:
                    #When predicting, predict returns [[proba1,proba2,proba3]]
                    target = reward + self.gamma*np.amax(self.Q.predict(state_t1)[0])

                # When predicting, predict returns [[proba1,proba2,proba3]]
                target_f = self.Q.predict(state_t1)[0]
                target_f[action] = target
                target_f = np.array(target_f)
                self.Q.fit(state_t,target_f.reshape((1,3)), epochs=1, verbose=1)

            if(self.epsilon > self.final_epsilon):
                self.epsilon = self.epsilon*self.epsilon_decay



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

        q_values = self.Q.predict(state) #array of q(state,action) for all action

        random_float = np.random.random()
        if (random_float < self.epsilon):
            new_action = np.random.randint(self.action_space)
        else:
            new_action = np.argmax(q_values)

        #We have done an additional step
        self.number_steps_done += 1
        return new_action




    def add_to_memory(self,state,previous_state,action,reward,done):
        self.D.append([state,previous_state,action,reward,done])