from algorithms import *
from lenet5 import *
import numpy as np
import matplotlib.pyplot as plt
import utils

_, (test_x, test_y) = load_MNIST()

path = './data/lenet5'

lenet5 = LeNet5(path=path, load=True)

#example of predictions
#print(lenet5.predict(np.array([test_x[0, :, :]]), np.array([test_y[0]])))
#print(lenet5.predict(test_x[0:10, :, :], test_y[0:10]))

epsilon = 0.25
d = 28*28

lab = lenet5.predict(test_x)
indexes = lab == test_y
data = test_x[indexes]
labels = test_y[indexes]

print(len(labels))  # 9826

# M = num workers; we consider a number of workers <=10
M = 10
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

m = 20
T = 20

# DECENTRALIZED
delta = decentralized_stochastic_gradient_free_FW(data_workers, y, lenet5.negative_loss, m, T, M, epsilon, d)
print(delta)

fig, ax = plt.subplots(figsize=(12, 12))
img = plt.imshow(delta[-1].reshape((28, 28)))
fig.colorbar(img, ax=ax, fraction=0.03, pad=0.05)
plt.savefig(f"./img/perturbation_{m}", bbox_inches="tight")
plt.show()

image = test_x[5].numpy().reshape(28, 28)
img_noise = image + delta[-1].reshape((28, 28))
fig, ax = plt.subplots(1, 2, figsize=(5, 5))
ax[0].imshow(image, cmap='Greys')
ax[1].imshow(img_noise, cmap='Greys')
plt.savefig(f"./img/image_perturbation_example_{m}", bbox_inches="tight")
plt.show()
print(lenet5.predict(np.array([img_noise.reshape(28, 28, 1)])))

# DECENTRALIZED VARIANCE REDUCED
# S_1 = 1000
# M = 10
# S_1' = 1000/M = 100
images = data[:1000]
targets = labels[:1000]
n = 5
S2 = 3
q = 5
S1 = len(images)

decentralized_variance_reduced_zo_FW(images, targets, lenet5.negative_loss, S2, T, M, n, epsilon, d, q, S1)