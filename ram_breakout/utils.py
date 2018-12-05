import skimage
import numpy as np
import tensorflow as tf
from keras import backend as K

def process_obs(observation):
    #So that it has the correct format for Keras
    x_t1 = skimage.color.rgb2gray(observation)
    x_t1 = skimage.transform.resize(x_t1, (80, 80))
    x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

    x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
    s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)
    return observation.reshape((1,250, 160,3))


def huber_loss(a, b, in_keras=True):
    error = a - b
    quadratic_term = error*error / 2
    linear_term = abs(error) - 1/2
    use_linear_term = (abs(error) > 1.0)
    if in_keras:
        # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
        use_linear_term = K.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term

def normalize(x):
    return x/256