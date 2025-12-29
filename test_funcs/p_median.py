import numpy as np
from pulp import *

def p_median(t_mat, frac_j, p, N, K):
    facilities = np.arange(N)
    demand = np.arange(K)

    # declare facility variables
    X = LpVariable.dicts('X', (facilities), 0, 1, LpInteger)

    # declare demand variables
    Y = LpVariable.dicts('Y', (demand, facilities), 0, 1, LpInteger)

    # create the LP object, set up as a MINIMIZATION problem
    prob = LpProblem('P_Median', LpMinimize)

    # Objective
    prob += sum(sum(t_mat[i, j] * Y[i][j] for j in facilities) * frac_j[i] for i in demand)

    # set up constraints
    # This is same as prob += sum([X[j] for j in location]) == p
    prob += lpSum([X[j] for j in facilities]) == p  # total equal to p

    for i in demand: prob += sum(Y[i][j] for j in facilities) == 1  # y sum up to 1

    # y_ij <= x_j
    for i in demand:
        for j in facilities:
            prob += Y[i][j] <= X[j]

    # Solve it
    prob.solve()

    #  format output
    print(' ')
    print("Status:", LpStatus[prob.status])
    print(' ')
    print("Objective: ", value(prob.objective))
    print(' ')
    units = []
    edges = []
    for v in prob.variables():
        subV = v.name.split('_')
        if subV[0] == "X" and v.varValue == 1:  # X variables
            units.append(int(subV[1]))
        elif subV[0] == "Y" and v.varValue == 1:  # Y variables
            edges.append([int(subV[1]), int(subV[2])])
    units = np.sort(units)
    print(units)
    print(edges)
    return units


def evaluate_p_median(old_x, t_mat, frac_j):
    """
    Evaluate the objective value of a given solution x in the p-median problem.

    Args:
        x (array-like): Binary vector indicating open facilities (1 if open, 0 otherwise).
        t_mat (np.ndarray): K x N matrix of response times (t_ij).
        frac_j (np.ndarray): K-length array of weights for each demand point.

    Returns:
        float: Total weighted response time (objective value).
    """

    x = np.array(old_x)
    if x.ndim == 1:
        x = x.reshape(1, -1)  # 转换为一维解为二维形式（1行）

    num_solutions = x.shape[0]
    objectives = np.zeros(num_solutions)

    for k in range(num_solutions):
        open_facilities = np.where(x[k, :] == 1)[0]  # 当前解的开放设施索引

        # 为每个需求点分配最近的开放设施
        nearest_facilities = np.argmin(t_mat[:, open_facilities], axis=1)
        nearest_facilities = open_facilities[nearest_facilities]  # 映射回原始索引

        # 计算总加权响应时间
        objectives[k] = np.sum(frac_j * t_mat[np.arange(len(frac_j)), nearest_facilities])

    return objectives[0] if old_x.ndim == 1 else objectives