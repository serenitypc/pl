import numpy as np


def genCross(pop, N, cross_n, bounds_upper, bounds_lower):
    # 模拟二进制交叉
    num_gen = pop.shape[1]
    offs = np.zeros((N, num_gen))

    for i in range(N // 2):
        idx = np.argsort(np.random.rand(N))
        lc = idx[:2]
        u = np.random.rand()
        if u <= 0.5:
            b = (2 * u) ** (1 / (cross_n + 1))
        else:
            b = (1 / (2 - 2 * u)) ** (1 / (cross_n + 1))

        parent = pop[lc, :]
        c1 = 0.5 * (parent[0, :] + parent[1, :]) + 0.5 * b * (parent[0, :] - parent[1, :])
        c2 = 0.5 * (parent[0, :] + parent[1, :]) + 0.5 * b * (parent[1, :] - parent[0, :])

        lc = c1 > bounds_upper
        c1[lc] = bounds_upper[lc]
        lc = c2 > bounds_upper
        c2[lc] = bounds_upper[lc]
        lc = c1 < bounds_lower
        c1[lc] = bounds_lower[lc]
        lc = c2 < bounds_lower
        c2[lc] = bounds_lower[lc]

        offs[2 * i, :] = c1
        offs[2 * i + 1, :] = c2

    return offs
