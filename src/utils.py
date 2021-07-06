from lenet5 import *
import numpy as np
import matplotlib.pyplot as plt

def plot_perturbation(perturbation, file_path):
  fig, ax = plt.subplots(figsize=(12, 12))
  img = plt.imshow(perturbation.reshape((28, 28)))
  fig.colorbar(img, ax=ax, fraction=0.03, pad=0.05)
  plt.savefig(file_path, bbox_inches="tight")
  plt.show()

def plot_perturbated_img(perturbation, image_test, file_path):
  image = image_test.numpy().reshape(28,28)
  img_noise = image + perturbation.reshape((28, 28))
  fig, ax = plt.subplots(1, 2,figsize=(5, 5))
  a = ax[0].imshow(image, cmap='Greys')
  b = ax[1].imshow(img_noise, cmap='Greys')
  fig.colorbar(a, ax=ax[0], fraction=0.03, pad=0.05)
  fig.colorbar(b, ax=ax[1], fraction=0.03, pad=0.05)
  plt.savefig(file_path, bbox_inches="tight")
  plt.show()


def get_data (load=True):
    ########################################### DATA preparation ######################
    _, (test_x, test_y) = load_MNIST()

    path = '../data/lenet5'

    lenet5 = LeNet5(path=path, load=load)

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

    return data_workers, y, lenet5

