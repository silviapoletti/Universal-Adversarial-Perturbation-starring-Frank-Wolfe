from alexnet import *
import numpy as np

path = '../data/alexnet'

alex = AlexNet(path=path, load=False)

_,(x_test, y_test) = datasets.mnist.load_data()

# adversarial perturbation:
delta = np.load("report_perturbation_m15_T100.npy")
delta = np.tile(delta, 10000)
delta = delta.reshape(10000, 28, 28)
# perturbed data with delta:
perturbed_data = x_test+delta
perturbed_data = np.clip(perturbed_data,0,1)
# adding 3 channels for alexnet:
perturbed_data = tf.pad(perturbed_data, [[0, 0], [2, 2], [2, 2]])
perturbed_data = np.clip(perturbed_data,0,1)
perturbed_data = tf.expand_dims(perturbed_data, axis=3, name=None)
perturbed_data = tf.repeat(perturbed_data, 3, axis=3)

# gaussian perturbation:
noise = np.random.normal(0,0.3,784)
noise = np.tile(noise, 10000)
noise = noise.reshape(10000, 28, 28)
# perturbed data with noise:
noise_data = x_test+noise
noise_data = np.clip(noise_data,0,1)
# adding 3 channels for alexnet:
noise_data = tf.pad(noise_data, [[0, 0], [2, 2], [2, 2]])
noise_data = np.clip(noise_data,0,1)
noise_data = tf.expand_dims(noise_data, axis=3, name=None)
noise_data = tf.repeat(noise_data, 3, axis=3)

import matplotlib.pyplot as plt
plt.imshow(perturbed_data[0], cmap='Greys')
plt.show()
plt.imshow(noise_data[0], cmap='Greys')
plt.show()
alex.model.evaluate(x_test, y_test, verbose=1)

alex.model.evaluate(perturbed_data, y_test, verbose=1)

alex.model.evaluate(noise_data, y_test, verbose=1)