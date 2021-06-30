from lenet5 import *
import numpy as np

_, (test_x, test_y) = load_MNIST()

path = './data/lenet5'

lenet5 = LeNet5(path = path, load = False)
print(lenet5.model.summary())

print(lenet5.predict(np.array([test_x[0, :, :]]), np.array([test_y[0]])))

print(lenet5.predict(test_x[0:10, :, :], test_y[0:10]))

