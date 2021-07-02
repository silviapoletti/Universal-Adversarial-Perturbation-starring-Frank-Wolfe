from lenet5 import *
import numpy as np
import algorithms

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

M = 10
data_per_classes = []
for label_class in range(0, 10):
    data_per_classes.append(data[labels == label_class][:100])

data_per_classes = np.array(data_per_classes)
data_workers = []



#TODO: controllare che torni a me sembra di si, sarebbe piu bello generalizzare meglio facendo la divisone intera.
# Ad ogni 10 colonne formo un worker delle immagini di diverse label formo un worker
idx = 0
for j in range(0, 10):
    image_worker = []
    for i in range(0, 10):
        image_worker.extend(data_per_classes[:, idx, :, :, :])
        idx = idx + 1
    data_workers.append(image_worker)

print(np.array(data_workers).shape) # all 10 workers, now have 100 images each of 10 classes.
# I think it does work now.