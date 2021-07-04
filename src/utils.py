
# S2 = n' * m
# n' = 2, 3,  4
# m = 18, 12, 9
# S2 = 36

# S2 = (2, 18) (3, 12) (4, 9)
###    1/3     1/3      1/3
def compatible_S2_pairs(card_s_2, n_components):
    '''

    :param card_s_2:
    :param n_components:
    :return: All possible pairs s.t i*j = card_s_2  excluding i=1 or i=n_components.
    '''
    result = []
    for i in range(2, n_components):
        if card_s_2 % i == 0:
            result.append((i, card_s_2/i))
    return result