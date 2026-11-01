import numpy as np
from genCross import genCross
from genMut import genMut
from res_fun import res_fun

np.random.seed(1)

# 1. 水库参数
Z0 = 982   # 死水位，m
V0 = 6.62  # 死库容，亿 m3
Zx = 994  # 正常蓄水位，m
Vx = 9.20  # 正常蓄水位对应库容，亿 m3
Zf = 990   # 汛限水位，m
Vf = 8.20  # 汛限库容，亿 m3
A = 8.5  # P = A * Q * H，出力单位 kW
Nmax = 42000  # 装机容量，kW
sQ = 5  # 最小下泄流量，m3/s

# 2. 5月至次年4月的来水和天数,5-9月为汛期
inQ = np.array([55, 100, 180, 130, 65, 40, 25, 20, 18, 24, 30, 80])
numDay = np.array([31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30])
V_state = np.array([Vf, Vf])      # 起点、终点库容，亿 m3

# 3. 遗传算法参数
M = 1  # 目标
N = 200  # 种群大小
gen = 500  # 进化次数
cross_n = 10  # 交叉参数
mut_rate = 0.05  # 变异概率
mut_para = 20  # 变异参数
num_gen = 11   # 每个个体含有11个基因，前11个月末库容，第12个月末固定
bounds_upper = np.array([Vf, Vf, Vf, Vf, Vx, Vx, Vx, Vx, Vx, Vx, Vx])
bounds_lower = V0 * np.ones(num_gen)

# 4. 初始化种群：一行一个方案，一列一个月末库容
pop = np.zeros((N, num_gen))
for i in range(num_gen):
    pop[:, i] = np.random.uniform(bounds_lower[i], bounds_upper[i], N)

pop[0, :] = Vf  # 加入一个已知可行的等库容方案
E_record = np.zeros(gen)  # 记录每一代的最优出力

# 5. 交叉、变异、计算、选择
for i in range(gen):
    offs_cross = genCross(pop, N, cross_n, bounds_upper, bounds_lower)
    offs = genMut(offs_cross, mut_para, mut_rate, bounds_upper, bounds_lower)
    new_pop = np.vstack((pop, offs))
    E, E_var, outQ, Q_surplus, new_pop = res_fun(
        new_pop, V_state, inQ, numDay, sQ, A, Nmax)
    lc = np.argsort(-E)
    pop = new_pop[lc[:N], :]
    E_record[i] = E[lc[0]]

# 6. 最优方案，结果直接留在变量里
E, E_var, outQ, Q_surplus, pop = res_fun(pop, V_state, inQ, numDay, sQ, A, Nmax)
lc = np.argmax(E)
E_best = E[lc]                   # 年发电量，亿 kWh
V_best = np.concatenate(([V_state[0]], pop[lc, :], [V_state[1]]))
outQ_best = outQ[lc, :]          # 各月下泄流量，m3/s
Q_surplus_best = Q_surplus[lc, :] # 各月弃水流量，m3/s
