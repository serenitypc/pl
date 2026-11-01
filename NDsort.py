import numpy as np


def NDsort(obj):
    # 所有目标都按最小化处理
    N = obj.shape[0]
    don_num = np.zeros(N, dtype=int)
    don_rank = np.zeros(N, dtype=int)
    don = []                        # don[i] 记录个体 i 支配的个体编号

    # 第 1 层：统计支配关系
    for i in range(N):
        temp = obj[i, :] - obj     # 第 i 个个体和全体逐行比较
        lc = np.all(temp <= 0, axis=1) & np.any(temp < 0, axis=1)
        don.append(np.where(lc)[0])
        lc = np.all(temp >= 0, axis=1) & np.any(temp > 0, axis=1)
        don_num[i] = np.sum(lc)

    re = [np.where(don_num == 0)[0].tolist()]
    don_rank[re[0]] = 1

    # 第 2 层至最后一层：扣除上一层对其他个体的支配
    current_rank = 2
    while True:
        temp1 = []
        for i in re[current_rank - 2]:
            for j in don[i]:
                don_num[j] -= 1
                if don_num[j] == 0:
                    temp1.append(j)
                    don_rank[j] = current_rank
        if len(temp1) == 0:
            break
        re.append(temp1)
        current_rank += 1

    return re, don_rank
