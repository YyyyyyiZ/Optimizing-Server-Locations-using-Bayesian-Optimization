from xml.sax import parseString
from collections import OrderedDict
from .base import TestFunction

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

import contextlib
import importlib

from .dispatch_main import *
from .alpha_hypercube import *
from .utils import *
from .GA import GA
from .BOCS.BOCS import BOCS
from .p_median import p_median

import time
import copy


def transfer_xval(binary_xval):
    unit = []
    index = 0
    for i in binary_xval:
        if i == 1:
            unit.append(index)
        index += 1
    return np.array(unit)


def binary_converter(ten_list, n_vars):
    binary_list = np.zeros(n_vars)
    for i in ten_list:
        binary_list[i] = 1
    return binary_list


def get_all_combinations(n_vars, sigma):
    # SAMPLE_MODELS: Function samples the binary models to
    # generate observations to train the statistical model

    # Generate matrix of zeros with ones along diagonals
    index_list = [i for i in range(n_vars)]
    comb = list(combinations(index_list, sigma))
    return comb


def get_data():
    Data = parameter(File_Name='Saint_Paul')
    N_1, N_2, K, Lambda_1, Lambda_2, Mu_1, Mu_2, N, pre_list_1, pre_list_2, frac_j_1, frac_j_2 = Data_to_Param(Data)
    #print(' K: ', K, ' Lambda_2: ', Lambda_2, ' Mu_2: ', Mu_2, ' N: ', N, ' frac_j_1: ', frac_j_1, ' frac_j_2: ', frac_j_2)
    Mu_1 = 47.17
    Mu_2 = 51.3
    distance = pd.read_csv("distance.csv")
    distance = distance.drop(distance.columns[0], axis=1)
    distance = distance / 1600

    def m_d(d, a, v_c):
        d_c = v_c ** 2 / (2 * a)
        acc = 2 * np.sqrt(d / a) * (d <= 2 * d_c)
        cru = (v_c / a + d / v_c) * (d > 2 * d_c)
        return acc + cru

    def c_d(d, b_0, b_1, b_2, a, v_c):
        return np.sqrt(b_0 * (b_2 + 1) + b_1 * (b_2 + 1) * m_d(d, a, v_c) + b_2 * (m_d(d, a, v_c) ** 2)) / m_d(d, a,
                                                                                                               v_c)

    a, v_c, b_0, b_1, b_2 = [1.04729917e-04, 9.98233204e-03, -7.13341809e+00, 4.16082726e+00, 1.39337617e-02]
    t_bar = m_d(distance, a, v_c) * np.exp(c_d(distance, b_0, b_1, b_2, a, v_c) ** 2 / 2) / 60  # convert it to minutes

    t_mat_1 = np.array(t_bar.iloc[:, [0, 1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
    stations_2 = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    t_mat_2 = np.array(t_bar.iloc[:, :])
    return K, N, Lambda_2, Mu_2, frac_j_2, t_mat_2


class MRT(TestFunction):
    """
    MRT Problem.

    """

    def __init__(self, max_iter, n_init, new_N, p, mu=5,
                 pmedian=0, ga=0, bocs=0, bocs_sub=0, opt=0, bocs_prior=0,
                 st_paul=1, random_seed=0):

        self.max_iter = max_iter
        self.n_init = n_init
        self.p = p
        self.new_N = new_N

        self.normalize = False
        # self.n_vertices = np.array([self.new_N] * self.p) # main里需要，但是好像后面的程序也没有用到。目前这个可能需要改一下
        self.n_vertices = np.array([2] * self.new_N)
        self.seed = random_seed
        self.config = self.n_vertices  # main里需要。其实就是一个array[x1 x2 ...]。
        self.dim = self.new_N
        self.categorical_dims = np.arange(self.dim)

        self.Lambda_2 = 84.68
        self.Mu_2 = mu
        self.K = 10 * 10

        self.random_hypercube = Two_State_Hypercube({'Lambda': self.Lambda_2, 'Mu': self.Mu_2, 'K': self.K})

        if st_paul:
            self.K, self.new_N, self.Lambda_2, self.Mu_2, self.frac_j_2, self.t_mat = get_data()
            # self.Mu_2 = 1000
            self.Mu_2 = 42.17
            self.Lambda_2 = 84.68
            # print(self.t_mat)
            print('self.t_mat.shape:', self.t_mat.shape)
            self.random_hypercube.Update_Parameters(N=self.p, Lambda=self.Lambda_2, Mu=self.Mu_2, K=self.K,
                                                    frac_j=self.frac_j_2)

            self.n_vertices = np.array([2] * self.new_N)
            self.config = self.n_vertices
            self.dim = self.new_N
            self.categorical_dims = np.arange(self.dim)
        else:
            self.random_hypercube.Update_Parameters(N=self.p)
            self.frac_j_2 = self.random_hypercube.Random_Fraction(seed=random_seed)
            # print('self.frac_j_2:', self.frac_j_2)
            # self.t_mat = self.random_hypercube.Random_Time_Mat(t_min = 5, t_max = 30, seed = random_seed)
            # print('self.t_mat:', self.t_mat)
            np.random.seed(random_seed)
            self.t_mat = np.random.uniform(low=5, high=30, size=(self.K, self.new_N))

        self.opt_f = self.opt_loc = -1
        self.p_median = self.p_median_loc = -1
        self.GA = self.GA_loc = self.GA_path = -1
        self.BOCS = self.BOCS_loc = -1
        self.BOCS_sub = self.BOCS_sub_loc = -1
        self.gp_prior = self.gp_prior_loc = -1
        self.BOCS_prior = self.BOCS_prior_loc = -1
        self.mfbo = self.mfbo_loc = -1

        if (pmedian):
            print('-' * 40 + "Running p-median solution" + '-' * 40)
            time_start = time.time()
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                unit = p_median(self.t_mat, self.frac_j_2, self.p, self.new_N, self.K)  # 为什么每次解出来的不一样？
                p_median_value = self.compute([unit], need_convert=False)[0]  # 这个在计算p-median的解在hypercube的MRT是多少
            print('-' * 80)
            print('p_median_value: ', p_median_value, ' unit:', unit)
            time_end = time.time()
            print('time cost: ', time_end - time_start, 's\n')
            self.p_median_loc = [unit]
            self.p_median = p_median_value
        if (ga):
            print('-' * 40 + "Running GA solution" + '-' * 40)
            time_start = time.time()
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                ga_sol = GA(self.new_N, self.p, self.K, self.Lambda_2, self.Mu_2, self.frac_j_2, self.t_mat)
            self.GA_loc = ga_sol.ga_unit
            self.GA = ga_sol.ga_opt
            self.GA_path = ga_sol.ga_path
            print('GA value: ', self.GA, ' unit:', self.GA_loc)
            time_end = time.time()
            print('time cost: ', time_end - time_start, 's\n')
        if (bocs):
            print('-' * 40 + "Running BOCS solution" + '-' * 40)
            time_start = time.time()
            # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            BOCS_sol = BOCS(self.max_iter + 20, self.n_init, self.new_N, self.p,
                            self.K, self.Lambda_2, self.Mu_2, self.frac_j_2, self.t_mat, acq="SDP-l1")
            self.BOCS_loc, self.BOCS = BOCS_sol.unit, BOCS_sol.opt
            print('BOCS value: ', self.BOCS, ' unit:', self.BOCS_loc)
            time_end = time.time()
            print('time cost: ', time_end - time_start, 's\n')
        if (bocs_sub):
            print('-' * 40 + "Running BOCS Submodular solution" + '-' * 40)
            time_start = time.time()
            # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            BOCS_sol = BOCS(self.max_iter + 20, self.n_init, self.new_N, self.p,
                            self.K, self.Lambda_2, self.Mu_2, self.frac_j_2, self.t_mat, acq="SDP-sub")
            self.BOCS_sub_loc, self.BOCS_sub = BOCS_sol.unit, BOCS_sol.opt
            print('BOCS Submodular value: ', np.min(self.BOCS_sub), ' unit:', self.BOCS_sub_loc)
            time_end = time.time()
            print('time cost: ', time_end - time_start, 's\n')

        if (bocs_prior):
            print('-' * 40 + "Running BOCS Submodular with Prior" + '-' * 40)
            time_start = time.time()
            BOCS_prior_sol = BOCS(self.max_iter + 20, self.n_init, self.new_N, self.p,
                                  self.K, self.Lambda_2, self.Mu_2, self.frac_j_2, self.t_mat, acq="SDP-sub",
                                  prior=True)
            self.BOCS_prior_loc, self.BOCS_prior = BOCS_prior_sol.unit, BOCS_prior_sol.opt
            print('BOCS Submodular  with Prior value: ', np.min(self.BOCS_prior), ' unit:', self.BOCS_prior_loc)
            time_end = time.time()
            print('time cost: ', time_end - time_start, 's\n')

        if (opt):
            print('-' * 40 + "Running Optimality by Enumeration" + '-' * 40)
            time_start = time.time()
            self.opt_f, self.opt_loc = self.optimal()
            print('opt value: ', self.opt_f)
            time_end = time.time()
            print('time cost: ', time_end - time_start, 's\n')

    def compute(self, x_val, normalize=False, need_convert=True, exact_sol=False):
        y_list = []
        for x in x_val:
            x = [int(i) for i in x]
            print('x: ', x)
            if need_convert:
                x = [i for i, j in enumerate(x) if j == 1]  # 这个是特意对于binary变成我们需要的[1,3,4,5,..]这样表示哪些地方有车
                print('transformed x: ', x)
                self.random_hypercube.Update_Parameters(N=len(x))  # 因为目前没有constraint，所以N的数量会变
            #x = transfer_xval(x)
            new_t_mat = np.array(self.t_mat[:, x])
            #print('new_t_mat.shape:', new_t_mat.shape)
            pre_list_sep = new_t_mat.argsort(axis=1)
            self.random_hypercube.Update_Parameters(t_mat=new_t_mat, pre_list=pre_list_sep)
            # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):  # 不要输出里面的东西
            #     if exact_sol:
            #         self.random_hypercube.Myopic_Policy(source = 't_mat')
            #         self.random_hypercube.Solve_Hypercube()
            #         MRT, _=self.random_hypercube.Get_MRT_Hypercube()
            #     else:
            #         rho_approx = self.random_hypercube.Larson_Approx()
            #         MRT_approx,_ = self.random_hypercube.Get_MRT_Approx()
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):  # 不要输出里面的东西
                rho_approx = self.random_hypercube.Larson_Approx()
                MRT_approx, MRT_approx_j = self.random_hypercube.Get_MRT_Approx()
                late_calls = self.random_hypercube.Get_LateCalls_Approx(threshold=10)
            #print(MRT_approx)
            #q_nj_approx = random_hypercube.q_nj

            if exact_sol:
                y_list.append(MRT)
            else:
                y_list.append(MRT_approx)

            # y_list.append(late_calls)

        return y_list

    def greedy_algo(self):
        fix = np.ones(self.new_N)
        for i in range(self.new_N - self.p):
            x_vals = []
            for j in range(self.new_N):
                if fix[j] == 0:
                    continue
                cur_fix = copy.deepcopy(fix)
                cur_fix[j] = 0
                x_vals.append(cur_fix)
            #print(x_vals)
            f_vals = self.compute(x_vals, need_convert=True)
            opt_index = np.argmax(f_vals)
            opt_loc = x_vals[opt_index]
            fix = opt_loc
        opt_f = f_vals[opt_index]
        return opt_f, opt_loc

    def optimal(self):
        # Compute minimum of objective function
        print('computing optimal solution...')
        time_start = time.time()
        x_vals = get_all_combinations(n_vars=self.new_N, sigma=self.p)
        #print(x_vals)
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            f_vals = self.compute(x_vals, need_convert=False)
        print('min(f_vals):', min(f_vals))
        opt_index = np.argmin(f_vals)
        #print(f_vals, opt_index)
        opt_f = f_vals[opt_index]
        opt_loc = x_vals[opt_index]
        time_end = time.time()
        print('opt_loc: ', opt_loc)
        print('opt_f: ', opt_f)
        print('optimal solution time cost: ', time_end - time_start, 's')
        return opt_f, opt_loc
