from lenet5 import *
import numpy as np
import utils
from algorithms import *
import matplotlib.pyplot as plt

data_workers, y, lenet5 = utils.get_data()

M = 10  # workers
# adjacency matrix
A = np.array([
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 0, 1, 1, 1, 1]
])

# A = np.ones((M, M))  # fully connected graph
D = np.diag(np.sum(A, axis=0))
L = D - A
D_half = np.linalg.inv(D**(1/2)) # diagonal matrix
W = np.identity(M) - np.dot(D_half, np.dot(L, D_half))
J = np.full((M, M), 1/M)
norm = np.linalg.norm(W - np.full((M, M), 1/M))
print('norm: ', norm)
d = 784
m = 20
T = 10
epsilon = 0.25

delta_bar = distributed_zo_FW(A, M, d, data_workers, y, lenet5.negative_loss, epsilon, m, T)
np.save(f'../data/perturbations/distributed/distributed_delta_{T}_{m}', delta_bar)

utils.plot_perturbation(delta_bar[-1, :], f'../data/img/distributed/distributed_delta_{T}_{m}')
utils.plot_perturbated_img(delta_bar[-1, :], data_workers[1, 1], f'../data/img/distributed/perturbated_img_{T}_{m}')