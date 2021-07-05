from algorithms import *
from lenet5 import *
import numpy as np
import matplotlib.pyplot as plt

M = 5  # workers
# adjacency matrix
A = [
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 0, 0, 0]
]

A = np.ones((M, M)) - np.identity(M)  # fully connected graph

D = np.diag(np.sum(A, axis=0))
L = D - A
print(L)

sparsity = 0.1
W = np.identity(M) - sparsity * L
print(W)

norm = np.linalg.norm(W - np.full((M, M), 1/M))
print(norm)
