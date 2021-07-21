from algorithms import *
from utils import *
import numpy as np

data_workers, y, lenet5, data, labels,  test_x, test_y = get_data(dim=800)

# Parameters
T = 20
M = 5  # num workers; we consider a number of workers <=10
d = 28*28
epsilon = 0.25
S1 = len(data_workers[0])
S2 = 3
n = 5  # n=10
q = 5  # q=7, q=9

delta = decentralized_variance_reduced_zo_FW(data_workers, y, lenet5.negative_loss, T, M, d, epsilon, S1, S2, n, q)
np.save(f"../data/perturbations/variance_reduced/perturbation_variance_n{n}_q{q}", delta[-1])
