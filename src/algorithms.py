
import numpy as np

#TODO: what is F function?
#TODO: x_t is delta_t?
#TODO: does the worker sample the data points or is the same of the master x_t?
#TODO: x0 is the 0? which point is the starting point? is a feasible but which one is it?

def decentralized_stochastic_gradient_free_FW(data_workers, F, m, T, M,epsilon,d):
    """
    :param data: images
    :param y: labels
    :param F: loss function
    :param m: number of directions
    :param T: number of queries
    :param M: number of workers
    :return: universal perturbation
    """
    #starting point, x is the perturbation
    delta = np.zeros(d) # starting point: delta_0
    delta_history = []
    g_worker = np.array(M) # should hold precedent g worker, handled by master.
    for t in range(0, T):
        for w_idx in range(0, M):
            g_worker[w_idx] = decentralized_worker_job(d, t, m, g_worker[w_idx], F, delta,y)

        #wait all workers computation
        g = np.average(g_worker)
        vt = - epsilon * np.sign(g)
        gamma = 2 / (t + 8)
        delta =  (1-gamma) * delta + gamma * vt
        delta_history.append(delta)
        # send to all nodes
        # what the fuck is x for worker and for master???? Worker should sample from the data points!!!??
    return delta, delta_history


def decentralized_worker_job(d, t, m, g, F, delta, y):
    '''
    :param d: dimension
    :param t:
    :param m: directions
    :param g: master g,  precedent
    :param F: loss function to minimize
    :return:
    '''
    z = np.random.normal(loc=0.0, scale=1.0, size=(d, d))

    ro = 4/((1+d/m)**(1/3) * (t+8)**(2/3))
    c = 2 * (m)**(1/2) / (d**(3/2) * (t+8)**(1/3))

    g_prec = g.copy()
    data_point = sample_data_point()

    g = 1/m * (np.sum((F(x + c * z, y) - F(data_point + delta, y))/ c) * z) # element wise product inside the summation, then i apply the sum
    g = (1 - ro) *  g_prec + ro * g
    return g

def sample_data_point():
    pass