from algorithms import *
from lenet5 import *
import numpy as np
import matplotlib.pyplot as plt

def plot_perturbation(perturbation, m):
  fig, ax = plt.subplots(figsize=(12, 12))
  img = plt.imshow(perturbation.reshape((28, 28)))
  fig.colorbar(img, ax=ax, fraction=0.03, pad=0.05)
  plt.savefig(f"./perturbation_variance_{m}", bbox_inches="tight")
  plt.show()

def plot_perturbated_img(perturbation, m):
  image = test_x[5].numpy().reshape(28,28)
  img_noise = image + perturbation.reshape((28, 28))
  fig, ax = plt.subplots(1, 2,figsize=(5, 5))
  a = ax[0].imshow(image, cmap='Greys')
  b = ax[1].imshow(img_noise, cmap='Greys')
  fig.colorbar(a, ax=ax[0], fraction=0.03, pad=0.05)
  fig.colorbar(b, ax=ax[1], fraction=0.03, pad=0.05)
  plt.savefig(f"./image_perturbation_variance_example_{m}", bbox_inches="tight")
  plt.show()

_, (test_x, test_y) = load_MNIST()

path = './data/lenet5'

lenet5 = LeNet5(path=path, load=True)

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


images = data[:1000]
targets = labels[:1000]
n = 5
S2 = 3
q = 7
S1 = len(images)
M = 5
T = 20
epsilon = 0.25
d = 784


delta = decentralized_variance_reduced_zo_FW(images, targets, lenet5.negative_loss, S2, T, M, n, epsilon, d, q, S1)
np.save(f"./data/perturbation_variance_{M}_{T}", delta[-1])
delta = delta[-1]

perturbation = np.tile(delta, data.shape[0])
perturbation = perturbation.reshape((data.shape[0],28,28,1))
perturbated_data = data + perturbation
perturbated_data = np.clip(perturbated_data, 0., 1.)

loss = lenet5.model.evaluate(perturbated_data, labels)
plot_perturbation(delta, M)
plot_perturbated_img(delta, M)
#loss_history.append(loss)