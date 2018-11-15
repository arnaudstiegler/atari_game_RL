import skimage
import numpy as np

def process_obs(observation):
    #So that it has the correct format for Keras
    x_t1 = skimage.color.rgb2gray(observation)
    x_t1 = skimage.transform.resize(x_t1, (80, 80))
    x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

    x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
    s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)
    return observation.reshape((1,250, 160,3))

obs = np.load('obs.npy')