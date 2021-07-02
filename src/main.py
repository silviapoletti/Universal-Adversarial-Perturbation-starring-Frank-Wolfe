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

data_0 = data[labels == 0][:100]
data_1 = data[labels == 1][:100]
data_2 = data[labels == 2][:100]
data_3 = data[labels == 3][:100]
data_4 = data[labels == 4][:100]
data_5 = data[labels == 5][:100]
data_6 = data[labels == 6][:100]
data_7 = data[labels == 7][:100]
data_8 = data[labels == 8][:100]
data_9 = data[labels == 9][:100]

M = 10
#for w_idx in range(1,M):
