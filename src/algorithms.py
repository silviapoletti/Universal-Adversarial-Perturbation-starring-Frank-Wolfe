import numpy as np

#TODO: what is F function?
#TODO: x_t is delta_t?
#TODO: does the worker sample the data points or is the same of the master x_t?
#TODO: x0 is the 0? which point is the starting point? is a feasible but which one is it?


def decentralized_stochastic_gradient_free_FW(data_workers, y, F, m, T, M, epsilon, d):
    """
    :param data: images
    :param y: labels
    :param F: loss function
    :param m: number of directions
    :param T: number of queries
    :param M: number of workers
    :return: universal perturbation
    """
    # starting point, x is the perturbation
    delta = np.zeros(d)  # starting point: delta_0
    delta_history = []
    gradient_worker = np.zeros((M, d))  # should hold workers' precedent g, handled by master.
    for t in range(0, T):
        ro = 4 / ((1 + d / m) ** (1 / 3) * (t + 8) ** (2 / 3))
        c = 2 * (m) ** (1 / 2) / (d ** (3 / 2) * (t + 8) ** (1 / 3))

        for w_idx in range(0, M):

            gradient_worker[w_idx, :] = decentralized_worker_job(data_workers[w_idx, :, :, :, :], y, d, ro, c, m, gradient_worker[w_idx, :], F, delta)
        #wait all workers computation
        g = np.average(gradient_worker, axis=0)
        v = - epsilon * np.sign(g)
        gamma = 2 / (t + 8)   # LMO
        delta = (1-gamma) * delta + gamma * v
        delta_history.append(delta)
        # send to all nodes
    return delta_history


def decentralized_worker_job(data, y, d, ro, c, m, g_prec, F, delta):
    """
    :param d: dimension
    :param ro:
    :param c:
    :param m: directions
    :param g: master g,  precedent
    :param F: loss function to minimize
    :return:
    """
    print("hey")
    g = np.zeros(d)

    # reshape:
    delta = np.tile(delta, 100)
    delta = delta.reshape((100,28,28,1))
    for i in range(0, m):
        z = np.random.normal(loc=0.0, scale=1.0, size=d)
        cz = c*z
        cz = np.tile(cz, 100)
        cz = cz.reshape((100, 28, 28, 1))
        g += 1/c * (F(data + delta + cz, y) - F(data + delta, y)) * z
    g = g/m

    if not np.array_equal(g_prec, np.zeros(d)):
        g = (1 - ro) * g_prec + ro * g

    return g
