from algorithms import *
#from lenet5 import *
import numpy as np
import matplotlib.pyplot as plt

M = 5  # workers
# adjacency matrix
A = [
    [1, 1, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 0],
    [1, 1, 0, 0, 1]
]

#A = np.ones((M, M))  # fully connected graph
D = np.diag(np.sum(A, axis=0))
L = D - A
D_half = np.linalg.inv(D**(1/2)) # diagonal matrix
W = np.identity(M) - np.dot(D_half,np.dot(L,D_half))
J = np.full((M, M), 1/M)
norm = np.linalg.norm(W - np.full((M, M), 1/M))
print(norm)
