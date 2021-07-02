from algorithms import *
from lenet5 import *
import numpy as np

_, (test_x, test_y) = load_MNIST()

path = './data/lenet5'

lenet5 = LeNet5(path=path, load=True)

#print(lenet5.predict(np.array([test_x[0, :, :]]), np.array([test_y[0]])))
#print(lenet5.predict(test_x[0:10, :, :], test_y[0:10]))

epsilon = 0.25
d = 28*28

loss, lab = lenet5.predict(test_x, test_y)
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

print(np.array(data_workers).shape)  # now all 10 workers have 100 images, 10 for each class.

y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.repeat(y, 10)
print(y)

gradient_worker = np.zeros((M, d))
print(gradient_worker.shape)
# print(gradient_worker[0, :] != np.zeros((1, d)))

if np.array_equal(gradient_worker[0, :], np.zeros((1, d))):
    print("ok")


if gradient_worker[0, :] is np.zeros((1, d)):
    print("ok 2")

