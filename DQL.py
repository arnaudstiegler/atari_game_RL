from collections import deque
import numpy as np
from keras.layers import Convolution2D,Flatten,Dense,Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import binary_crossentropy,mean_squared_error
from keras.initializers import random_uniform

class DQL_agent():

    def __init__(self,state_space,action_space):

        #Parameters of the environment
        self.state_space = state_space
        self.action_space = action_space

        #Learning parameters
        self.epsilon = 0.1

        #Memory replay parameters
        self.memory_size = 1000
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
        self.batch_learning = 0

    def _build_model(self):

        #To define!!!

        '''
        model = Sequential()
        #keras.initializers.random_uniform(minval=-0.1, maxval=0.1, seed=None)
        img_channels, img_rows, img_cols = 3, 250, 160
        S = Input(shape=(img_rows, img_cols, img_channels,1), name='Input')
        h0 = Convolution2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu',
                           kernel_initializer='random_uniform', bias_initializer='random_uniform')(S)
        h1 = Convolution2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu',
                           kernel_initializer='random_uniform', bias_initializer='random_uniform')(h0)
        h2 = Flatten()(h1)
        h3 = Dense(256, activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform')(h2)
        P = Dense(1, name='o_P', activation='sigmoid', kernel_initializer='random_uniform',
                  bias_initializer='random_uniform')(h3)
        V = Dense(1, name='o_V', kernel_initializer='random_uniform', bias_initializer='random_uniform')(h3)

        model = Model(inputs=S, outputs=[P, V])
        rms = rmsprop(lr=self.learning_rate_cnn, rho=0.99, epsilon=0.1)
        model.compile(loss={'o_P': binary_crossentropy, 'o_V': mean_squared_error}, loss_weights={'o_P': 1., 'o_V': 0.5},
                      optimizer=rms)

        '''

        img_channels, img_rows, img_cols = 3, 250, 160

        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(img_rows, img_cols, img_channels,)))  # 250*160*3
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(3))
        adam = Adam(lr=self.learning_rate_cnn)
        model.compile(loss='mse', optimizer=adam)
        return model


    def experience_replay(self):
    #TODO: Implement experience replay
        return 0

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

        return new_action





        return 0



    def add_to_memory(self,state,previous_state,action,reward):
        self.D.append(state,previous_state,action,reward)