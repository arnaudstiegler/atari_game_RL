from collections import deque
import numpy as np
import random
from keras.layers import Convolution2D,Flatten,Dense,Activation,Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import binary_crossentropy,mean_squared_error
from keras.initializers import random_normal
import skimage

class DQL_agent():

    def __init__(self,state_space,action_space):

        #Parameters of the environment
        self.state_space = state_space
        self.action_space = action_space

        #Learning parameters
        self.epsilon = 1.0
        #Number of time steps over which the agent will explore
        self.explore = 250000
        #Final value for epsilon (once exploration is finished)
        self.final_epsilon = 0.001

        self.epsilon_decay = (self.final_epsilon - self.epsilon)/(self.explore)
        self.gamma = 0.95

        #Memory replay parameters
        self.memory_size = 100000
        # Format of an experience is: (state,previous_state,action,reward)
        self.D = deque([],self.memory_size)

        #Parameters for the CNN
        self.learning_rate_cnn = 0.0001
        self.Q = self._build_model()

        #Parameters for the ongoing episode
        self.state = None
        self.previous_state = None
        self.action = None
        self.reward = None
        self.initial_move = True
        self.observe_phase = True
        self.observe_steps = 1000 #Number of steps for observation (no learning)

        #Max number of steps between two experience replays
        self.experience_nb_steps=1 #We update at each step
        #Size of a batch for experience replay
        self.experience_batch_size = 32
        #A counter of the number of steps since last experience replay
        self.time_steps = 0

    def reinitialize_agent(self):
        #This function is actually useless
        self.initial_move = True
        self.time_steps = 0
        self.D = deque([], self.memory_size)

    def _build_model(self):

        img_channels, img_rows, img_cols = 4,80,80

        init = random_normal(mean=0.0, stddev=0.05, seed=None)
        model = Sequential()
        model.add(Convolution2D(16, 8, 8, subsample=(4,4), border_mode='same', kernel_initializer=init,input_shape=(img_channels,img_rows,img_cols)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='same', kernel_initializer=init))
        model.add(Activation('relu'))
        '''
        model.add(Convolution2D(64, 3, 3,strides=4, subsample=(1, 1), border_mode='same',kernel_initializer=init))
        model.add(Activation('relu'))
        '''
        model.add(Flatten())
        model.add(Dense(256,kernel_initializer=init))
        model.add(Activation('relu'))
        model.add(Dense(self.action_space,activation='linear',kernel_initializer=init))
        adam = Adam(lr=self.learning_rate_cnn)
        model.compile(loss='mse', optimizer=adam)
        return model


    def experience_replay(self):
        #if(self.batch_learning % self.experience_nb_steps==0 and self.batch_learning >= self.experience_batch_size):
        print(len(self.D))
        print(self.experience_batch_size+1)
        if(len(self.D) > self.experience_batch_size+1):

            batch = random.sample(self.D, self.experience_batch_size)

            for i in range(0, len(batch)):
                state_t = batch[i][1]
                state_t1 = batch[i][0]
                action = batch[i][2]
                reward = batch[i][3]
                done = batch[i][4]

                '''
                OLD VERSION
                
                if(done):
                    target=reward
                else:
                    #When predicting, predict returns [[proba1,proba2,proba3]]
                    target = reward + self.gamma*np.amax(self.Q.predict(state_t1)[0])

                # When predicting, predict returns [[proba1,proba2,proba3]]
                target_f = self.Q.predict(state_t1)[0]
                target_f[action] = target
                target_f = np.array(target_f)
                '''

                target = reward
                if not done:
                    Q_next = self.Q.predict(state_t1)[0]
                    target = (reward + self.gamma * np.amax(Q_next))

                target_f = self.Q.predict(state_t)
                target_f[0][action] = target

                self.Q.fit(state_t,target_f.reshape((1,self.action_space)), epochs=1, verbose=0)

            if(self.epsilon > self.final_epsilon):
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

        q_values = self.Q.predict(state) #array of q(state,action) for all action

        random_float = np.random.random()
        if (random_float < self.epsilon):
            new_action = np.random.randint(self.action_space)
        else:
            new_action = np.argmax(q_values)

        #We have done an additional step
        self.time_steps += 1
        return new_action




    def add_to_memory(self,state,previous_state,action,reward,done):
        self.D.append([state,previous_state,action,reward,done])


    def check_learning(self,env,ep):
        if(ep % 25 == 0 ):
            print('---- CHECKING RESULTS ----')
            epsilon = self.epsilon
            timesteps = self.time_steps
            self.epsilon = 0.05

            rewards = []

            for ep in range(20):

                obs = env.reset()
                done=False
                total_reward = 0

                x_t = skimage.color.rgb2gray(obs)
                x_t = skimage.transform.resize(x_t, (80, 80))
                x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))

                x_t = x_t.reshape(1, x_t.shape[0], x_t.shape[1])

                s_t = np.stack((x_t, x_t, x_t, x_t), axis=1)


                while(done==False):
                    # If it is the first move, we can't store anything in the memory
                    action = self.act(s_t)
                    new_state, reward, done, _info = env.step(action)
                    x_t1 = skimage.color.rgb2gray(new_state)
                    x_t1 = skimage.transform.resize(x_t1, (80, 80))
                    x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

                    x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
                    s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)
                    self.state = s_t1
                    total_reward += reward


                rewards.append(total_reward)
                print(total_reward)

            with open('breakout/epoch_rewards.txt','a') as file:
                file.write(str(np.mean(rewards))+'\n')

            self.time_steps = timesteps
            self.epsilon = epsilon

            return True
        else:
            return False