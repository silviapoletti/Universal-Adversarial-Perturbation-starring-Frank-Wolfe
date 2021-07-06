from algorithms import *
from lenet5 import *
import numpy as np
import matplotlib.pyplot as plt
import utils

_, (test_x, test_y) = load_MNIST()

path = '../data/lenet5'

lenet5 = LeNet5(path=path, load=True)


# parameters:
epsilon = 0.25
d = 28*28
M = 10
m = 15
T = 100

lab = lenet5.predict(test_x)
indexes = lab == test_y
data = test_x[indexes]
labels = test_y[indexes] # 9826

data_per_classes = []
for label_class in range(0, 10):
    data_per_classes.append(data[labels == label_class][:100])

data_per_classes = np.array(data_per_classes)
data_workers = []

for offset in range(0, 100, M):
    image_worker = []
    for c in range(0, 10):
        image_worker.extend(data_per_classes[c, offset:offset+M, :, :, :])
    data_workers.append(image_worker)

data_workers = np.array(data_workers)
print(data_workers.shape)  # now all 10 workers have 100 images, 10 for each class.

y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.repeat(y, 10)

# DECENTRALIZED
delta = decentralized_stochastic_gradient_free_FW(data_workers, y, lenet5.negative_loss, m, T, M, epsilon, d)
print(delta)

np.save(f"../data/perturbations/decentralized_stoch/report_perturbation_m{m}_T{T}", delta[-1])