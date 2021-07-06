import numpy as np
import utils
#TODO: x0 is the 0? which point is the starting point? is a feasible but which one is it?


def decentralized_stochastic_gradient_free_FW(data_workers, y, F, m, T, M, epsilon, d, tol=None):
    """
    :param data_workers: images. Each row contains the images for a single worker.
    :param y: labels
    :param F: loss function
    :param m: number of directions
    :param T: number of queries
    :param M: number of workers
    :param epsilon: tolerance
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
    :param ro: parameter linked to I-RDSA
    :param c: parameter linked to I-RDSA
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


def decentralized_variance_reduced_zo_FW(data_workers, y, F, S2, T, M, n, epsilon, d, q, S1, tol=None):
    """
    :param data_workers: images. Each row contains the images for a single worker.
    :param y: labels
    :param F: loss function
    :param S2:
    :param T: number of queries
    :param M: number of workers
    :param n: component functions, is the magic number
    :param epsilon: tolerance
    :param d: image dimension
    :param q: period
    :param S2: number of images
    :param tol: tolerance for duality gap
    :return: universal perturbation
    """
    # starting point, x is the perturbation
    delta = np.zeros(d)  # starting point: delta_0
    delta_history = [delta]

    gradient_worker = np.zeros((M, d))  # should hold workers' precedent g, handled by master.

    for t in range(0, T):
        print("Iteration number ", t+1)
        eta_RDSA = 2/(d**(3/2)*(t+8)**(1/3))
        eta_KWSA = 2/(d**(1/2)*(t+8)**(1/3))
        delta_prec = None if t == 0 else delta_history[-2]

        for w_idx in range(0, M):
            gradient_worker[w_idx, :] = decentralized_worker_job_variance_reduced(
                data_workers, y, t, F, d, eta_RDSA, eta_KWSA, gradient_worker[w_idx, :], delta, q, S1, S2, n, M, delta_prec)
        # wait all workers computation
        g = np.average(gradient_worker, axis=0)
        v = - epsilon * np.sign(g)
        gamma = 2 / (t + 8)   # LMO
        delta = (1-gamma) * delta + gamma * v
        delta_history.append(delta)
        # send to all nodes

    return delta_history


def decentralized_worker_job_variance_reduced(data, y, t, F, d, eta_RDSA, eta_KWSA, g_prec, delta, q, S1, S2, n, M, delta_prec):
    """
    :param data: n images
    :param y: n labels
    :param t: iteration
    :param F: loss function to minimize
    :param d: images dimension
    :param eta_RDSA: parameter linked to RDSA
    :param eta_KWSA: parameter linked to KWSA
    :param g_prec: g computed by the same worker at the previous iteration, coming from the master node
    :param delta: perturbation
    :param q: period
    :param S1: number of images
    :param S2:
    :param n: number of the loss function's components
    :param M: number of workers
    :param delta_prec: perturbation computed at the previous iteration, coming from the master node
    :return: gradient
    """
    # TODO: S1_prime e S2 in teoria sono due valori diversi, quindi utiliziamo un numero diverso di immagini nei due casi
    g = np.zeros(d)
    size = S1 // (n * M)

    # reshape:
    delta = np.tile(delta, size)
    delta = delta.reshape((size, 28, 28, 1))

    if delta_prec is not None:
        # reshape:
        delta_prec = np.tile(delta_prec, size)
        delta_prec = delta_prec.reshape((size, 28, 28, 1))

    if (t % q) == 0:
        # KWSA
        for k in range(d):
            e = np.zeros(d)
            e[k] = eta_KWSA
            # sampling of S1_prime images:
            sampling_index = np.random.randint(low=0, high=data.shape[0], size=size*n)
            sampling_images = np.take(data, sampling_index, axis=0)
            sampling_labels = y[sampling_index]
            e = np.tile(e, size)
            e = e.reshape((size, 28, 28, 1))
            verbose = 0
            for j in range(0, n):
                if j == n - 1:
                    verbose = 1
                g[k] += 1 / eta_KWSA * (F(sampling_images[j * size: (j + 1) * size, :, :, :] + delta + e,
                                          sampling_labels[j * size:(j + 1) * size], verbose=verbose)
                                        - F(sampling_images[j * size: (j + 1) * size, :, :, :] + delta,
                                            sampling_labels[j * size:(j + 1) * size]))
            g[k] = g[k] / n

    else:
        # RDSA
        z = np.random.normal(loc=0.0, scale=1.0, size=d)
        eta_z = eta_RDSA * z
        eta_z = np.tile(eta_z, size)
        eta_z = eta_z.reshape((size, 28, 28, 1))
        # sampling:
        sampling_index = np.random.randint(low=0, high=data.shape[0], size=size * n)
        sampling_components = np.random.randint(low=0, high=n, size=S2)
        sampling_images = np.take(data, sampling_index, axis=0)
        sampling_labels = y[sampling_index]
        verbose = 0
        for j in sampling_components:
            if j == sampling_components[-1]:
                verbose = 1
            g += 1 / eta_RDSA * ((F(sampling_images[j*size:(j+1)*size,:,:,:] + delta + eta_z, sampling_labels[j*size:(j+1)*size], verbose=verbose) -
                             F(sampling_images[j*size:(j+1)*size,:,:,:] + delta, sampling_labels[j*size:(j+1)*size])) * z -
                            (F(sampling_images[j*size:(j+1)*size,:,:,:] + delta_prec + eta_z, sampling_labels[j*size:(j+1)*size]) -
                             F(sampling_images[j*size:(j+1)*size,:,:,:] + delta_prec, sampling_labels[j*size:(j+1)*size])) * z)
        g = g / S2
        g = g_prec + g

    return g
def distributed_zo_FW(A, M, d, epsilon, T):
    """
    :param A: adjacency matrix
    :param M: number of workers
    :param d: dimension
    :param epsilon: tolerance
    :param T: number of queries
    :return:
    """
    D = np.diag(np.sum(A, axis=0))
    L = D - A
    D_half = np.linalg.inv(D ** (1 / 2))  # diagonal matrix
    W = np.identity(M) - np.dot(D_half, np.dot(L, D_half))
    x = np.random.uniform(low=-epsilon, high=epsilon, size=(M, d)) # initial matrix in which each row correspond to a worker initial point
    for i in range(0,M):
        distributed_zo_FW_worker_job_iterate(A[i,:],W[i,:],x)

def distributed_zo_FW_worker_job_iterate(A_row, W_row,x):
    neighbors_indices = np.nonzero(A_row)
    sum = np.zeros(x.shape[1])
    for i in neighbors_indices:
        sum += W_row[i] * x[i,:]
    return sum