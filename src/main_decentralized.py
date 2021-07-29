from algorithms import *
from utils import *

data_workers, y, lenet5, data, labels,  test_x, test_y = get_data()

# parameters:
epsilon = 0.25
d = 28*28
M = 10
m = 15
T = 20  # T=50, T=100

delta = decentralized_stochastic_gradient_free_FW(data_workers, y, lenet5.negative_loss, T, M, d, epsilon, m, verbose=1)
np.save(f"../data/perturbations/decentralized_stoch/report_perturbation_m{m}_T{T}", delta[-1])
