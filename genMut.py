import numpy as np


def genMut(offs, mut_para, mut_rate, bounds_upper, bounds_lower):
    offs = offs.copy()
    N, n = offs.shape
    diff_x = bounds_upper - bounds_lower

    for i in range(N):
        for j in range(n):
            if np.random.rand() <= mut_rate:
                u = np.random.rand()
                if u > 0.5:
                    delta = 1 - (2 * (1 - u)) ** (1 / (mut_para + 1))
                else:
                    delta = (2 * u) ** (1 / (mut_para + 1)) - 1
                offs[i, j] = offs[i, j] + delta * diff_x[j]

    for i in range(n):
        lc = offs[:, i] > bounds_upper[i]
        offs[lc, i] = bounds_upper[i]
        lc = offs[:, i] < bounds_lower[i]
        offs[lc, i] = bounds_lower[i]

    return offs
