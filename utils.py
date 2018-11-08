

def process_obs(observation):
    #So that it has the correct format for Keras
    return observation.reshape((1,250, 160,3))