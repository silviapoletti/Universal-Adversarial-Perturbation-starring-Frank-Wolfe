import polytope as pc
import pypoman
import numpy as np

epsilon = 0.25
# d = 28*28
d = 3
A = np.identity(d)
b = np.full(d, epsilon)
C = pc.Polytope(A, b)

vertices = pypoman.compute_polytope_vertices(A, b)

print(vertices)


#TODO: what is F function?
#TODO: x_t is delta_t?
#TODO: does the worker sample the data points or is the same of the master x_t?
#TODO: x0 is the 0? which point is the starting point? is a feasible but which one is it?
#TODO:
def decentralized_stochastic_gradient_free_FW(data, y, F, vertices, m, T, M):
    """
    :param x: images
    :param y: labels
    :param F: loss function
    :param C: polytope
    :param m: number of directions
    :param T: number of queries
    :param M: number of workers
    :return:
    """
    #starting point, x is the perturbation
    x = np.zeros(d) # x starting point: x_0
    loss_history = []
    g_worker = np.array(M) # should hold precedent g worker, handled by master.
    for t in range(0, T):
        for w_idx in range(0, M):
            g_worker[w_idx] = decentralized_worker_job(28*28, t, m, g_worker[w_idx], F, x,y)

        #wait all workers computation
        g = np.average(g_worker)
        vt = - epsilon * np.sign(g) + x0 # TODO: check x_orig
        gamma = 2 / (t + 8)
        x =  (1-gamma) * x + gamma * vt
        loss_history.append(x)
        # send to all nodes
        # what the fuck is x for worker and for master???? Worker should sample from the data points!!!??
    return x, loss_history


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