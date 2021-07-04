import numpy as np

#TODO: x0 is the 0? which point is the starting point? is a feasible but which one is it?


def decentralized_stochastic_gradient_free_FW(data_workers, y, F, m, T, M, epsilon, d, tol=None):
    """
    :param data_workers: images. Each row contains the images for a single worker.
    :param y: labels
    :param F: loss function
    :param m: number of directions
    :param T: number of queries
    :param M: number of workers
    :param epsilon:
    :param d: image dimension
    :param tol: tolerance for duality gap
    :return: universal perturbation
    """
    # starting point, x is the perturbation
    delta = np.zeros(d)  # starting point: delta_0
    delta_history = []
    gradient_worker = np.zeros((M, d))  # should hold workers' precedent g, handled by master.
    for t in range(0, T):
        print("Iteration number ", t+1)
        ro = 4 / ((1 + d / m) ** (1 / 3) * (t + 8) ** (2 / 3))
        c = 2 * m ** (1 / 2) / (d ** (3 / 2) * (t + 8) ** (1 / 3))

        for w_idx in range(0, M):
            gradient_worker[w_idx, :] = decentralized_worker_job(data_workers[w_idx, :, :, :, :], y, F, m, d, ro, c, gradient_worker[w_idx, :], delta)
        # wait all workers computation
        g = np.average(gradient_worker, axis=0)
        v = - epsilon * np.sign(g)
        gamma = 2 / (t + 8)   # LMO
        delta = (1-gamma) * delta + gamma * v
        delta_history.append(delta)
        # send to all nodes

    return delta_history


def decentralized_worker_job(data, y, F, m, d, ro, c, g_prec, delta):
    """
    :param data: n images
    :param y: n labels
    :param F: loss function to minimize
    :param m: number of directions
    :param d: images dimension
    :param ro:
    :param c:
    :param g_prec: g computed by the same worker at the previous iteration, coming from the master node
    :param delta: perturbation
    :return: gradient
    """
    g = np.zeros(d)

    # reshape:
    delta = np.tile(delta, 100)
    delta = delta.reshape((100, 28, 28, 1))
    for i in range(0, m):
        z = np.random.normal(loc=0.0, scale=1.0, size=d)
        cz = c*z
        cz = np.tile(cz, 100)
        cz = cz.reshape((100, 28, 28, 1))
        # TODO: dobbiamo normalizzare le immagini perturbate?
        g += 1/c * (F(data + delta + cz, y) - F(data + delta, y)) * z
    g = g/m

    if not np.array_equal(g_prec, np.zeros(d)):
        g = (1 - ro) * g_prec + ro * g

    return g


def decentralized_variance_reduced_zo_FW(data_workers, y, F, S2, T, M, epsilon, d, tol=None):
    """
    :param data_workers: images. Each row contains the images for a single worker.
    :param y: labels
    :param F: loss function
    :param m: number of directions
    :param T: number of queries
    :param M: number of workers
    :param epsilon:
    :param d: image dimension
    :param tol: tolerance for duality gap
    :return: universal perturbation
    """
    # starting point, x is the perturbation
    delta = np.zeros(d)  # starting point: delta_0
    delta_history = []
    gradient_worker = np.zeros((M, d))  # should hold workers' precedent g, handled by master.
    for t in range(0, T):
        print("Iteration number ", t+1)
        ro = 4 / ((1 + d / m) ** (1 / 3) * (t + 8) ** (2 / 3))
        c = 2 * m ** (1 / 2) / (d ** (3 / 2) * (t + 8) ** (1 / 3))

        for w_idx in range(0, M):
            gradient_worker[w_idx, :] = decentralized_worker_job_variance_reduced(
                data_workers[w_idx, :, :, :, :], y, F, m, d, ro, c, gradient_worker[w_idx, :], delta)
        # wait all workers computation
        g = np.average(gradient_worker, axis=0)
        v = - epsilon * np.sign(g)
        gamma = 2 / (t + 8)   # LMO
        delta = (1-gamma) * delta + gamma * v
        delta_history.append(delta)
        # send to all nodes

    return delta_history


def decentralized_worker_job_variance_reduced(data, y, F, d, eta, g_prec, delta, t, q, S_1, S_2, n, M, delta_prec):
    """
    :param data: n images
    :param y: n labels
    :param F: loss function to minimize
    :param d: images dimension
    :param eta:
    :param g_prec: g computed by the same worker at the previous iteration, coming from the master node
    :param delta: perturbation
    :param t: current iteration
    :param q: period
    :param S_1:
    :param S_2:
    :param n: number of the loss function's components
    :param M: number of workers
    :param delta_prec: perturbation computed at the previous iteration, coming from the master node
    :return: gradient
    """
    # TODO: S_1_prime e S_2 in teoria sono due valori diversi, quindi utiliziamo un numero diverso di immagini nei due casi
    g = np.zeros(d)

    if (t % q) == 0:
        # KWSA
        S_1_prime = S_1/(M * d)

        for j in range(0, n):
            # sampling of S_1_prime images
            e = np.zeros(n)
            e[j] = 1
            eta_e = eta * e
            eta_e = np.tile(eta_e, S_1_prime)
            eta_e = eta_e.reshape((S_1_prime, 28, 28, 1))
            # TODO: dobbiamo normalizzare le immagini perturbate?
            g += 1 / eta * (F(data + delta + eta_e, y) - F(data + delta, y)) * e
        g = g / n  # TODO: non è di dimensione d ??

    else:
        # RDSA
        for j in range(0, S_2):
            z = np.random.normal(loc=0.0, scale=1.0, size=d)

            eta_z = eta * z
            eta_z = np.tile(eta_z, S_2)
            eta_z = eta_z.reshape((S_2, 28, 28, 1))
            # TODO: dobbiamo normalizzare le immagini perturbate?
            g += 1 / eta * ((F(data + delta + eta_z, y) - F(data + delta, y)) * z -
                            (F(data + delta_prec + eta_z, y) - F(data + delta_prec, y)))
        g = g / S_2
        g = g_prec + g

    return g
