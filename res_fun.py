import numpy as np


def res_fun(pop, V_state, inQ, numDay, sQ, A, Nmax):
    # 库容单位亿 m3，流量单位 m3/s。
    N, num_gen = pop.shape
    outQ = np.zeros((N, num_gen + 1))
    E_var = np.zeros((N, num_gen + 1))   # 实际记录各月出力 kW
    Q_surplus = np.zeros((N, num_gen + 1))

    # 加上起点、终点
    V = np.zeros((N, num_gen + 2))
    V[:, 0] = V_state[0]
    V[:, 1:-1] = pop
    V[:, -1] = V_state[1]

    # 水位—库容表
    V_table = np.array([6.62, 6.96, 7.34, 7.76, 8.20, 8.68, 9.20])
    Z_table = np.array([982, 984, 986, 988, 990, 992, 994])

    for k in range(num_gen + 1):
        # 水量平衡：下泄流量 = 来水 + 库容减少量 / 时段长度
        outQ[:, k] = inQ[k] + (V[:, k] - V[:, k + 1]) * 1e8 / (numDay[k] * 86400)

        # 平均库容 → 上游水位 → 尾水位 → 净水头
        V_mean = 0.5 * (V[:, k] + V[:, k + 1])
        Zup = np.interp(V_mean, V_table, Z_table)
        tempQ = np.maximum(outQ[:, k], 0)  # 负下泄方案随后用罚值淘汰
        Zdown = 920 + 0.03 * tempQ ** 0.8
        H_loss = 1                        # 水头损失，m
        H = Zup - Zdown - H_loss

        # 最大发电流量 80 m3/s，出力不超过装机容量
        Q_power = np.minimum(tempQ, 80)
        temp = A * Q_power * H
        lc = temp > Nmax
        temp[lc] = Nmax
        E_var[:, k] = temp
        Q_surplus[:, k] = tempQ - temp / (A * H)

    # 年发电量：月出力乘各月小时数，换算为亿kWh
    E = np.sum(E_var * numDay * 24, axis=1) / 1e8

    # 下泄须满足生态要求且不超过最大泄量 350 m3/s
    lc = np.any((outQ < sQ) | (outQ > 350), axis=1)
    E[lc] = -1e10

    return E, E_var, outQ, Q_surplus, pop
