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

def decentralized_stochastic_gradient_free_FW(x, y, f, vertices, m, T):
    """
    :param x: images
    :param y: labels
    :param f: loss function
    :param C: polytope
    :param m: number of directions
    :param T: number of queries
    :return:
    """
    x0 = np.zeros(d)
    pass

