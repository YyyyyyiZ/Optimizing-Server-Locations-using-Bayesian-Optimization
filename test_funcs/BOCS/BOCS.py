# Author: Yi Zhang
# Date:   March 2025
# Source: Ricardo Baptista and Matthias Poloczek
#
# See LICENSE.md for copyright information
#

import sys
import contextlib
import cvxpy as cvx
from .LinReg import LinReg
from .sample_models import sample_models_constraints, sample_with_constraints, sample_initial_points

from ..dispatch_main import  *
from ..alpha_hypercube import *

from .BQPSubmodular import BQPSubmodular
from ..p_median import evaluate_p_median


class BOCS:
    def __init__(self, evalBudget, n_init, n_vars, fix_num, K, Lambda, Mu, frac_j, t_mat, acq='SDP-l1', prior=False):
        self.evalBudget = evalBudget
        self.n_init = n_init
        self.n_vars = n_vars
        self.fix_num = fix_num
        self.K = K
        self.Lambda = Lambda
        self.Mu = Mu
        self.frac_j = frac_j
        self.t_mat = t_mat
        self.prior = prior
        
        self.penalty = 1e-4
        self.acq = acq
        self.store_history = {}
        self.x_vals = None
        self.y_vals = None
        self.unit, self.opt = self.bocs()



    def fitness_evaluation(self, unit):
        unit_new = [i for i, j in enumerate(unit.tolist()) if j == 1]
        units_code = sum(2 ** np.array(unit_new))
        if units_code in self.store_history.keys():
            MRT = self.store_history[units_code]
        else:
            new_t_mat = np.array(self.t_mat[:, unit_new])
            new_pre_list = new_t_mat.argsort(axis=1)
            two_hc = Two_State_Hypercube({'Lambda': self.Lambda, 'Mu': self.Mu})
            two_hc.Update_Parameters(N=self.fix_num, K=self.K, pre_list=new_pre_list, frac_j=self.frac_j, t_mat=new_t_mat)
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                two_hc.Larson_Approx()
                # MRT, _ = two_hc.Get_MRT_Approx()
                MRT = two_hc.Get_LateCalls_Approx(threshold = 10)
            self.store_history[units_code] = MRT
        if self.prior:
            MRT -= evaluate_p_median(unit, self.t_mat, self.frac_j)

        return MRT


    def bocs(self):
        self.x_vals = sample_initial_points(self.n_init, self.n_vars, self.fix_num)
        self.y_vals = np.array([self.fitness_evaluation(units) for units in self.x_vals])

        # Train initial surrogate model
        LR = LinReg(self.n_vars, order=2)
        LR.train(self.x_vals, self.y_vals)

        n_iter = self.evalBudget - self.n_init
        model_iter = np.zeros((n_iter, self.n_vars))
        obj_iter = np.zeros(n_iter)

        A_final=None
        b_final=None
        mini=100
        for t in range(n_iter):
            # Draw alpha vector
            alpha_t = LR.alpha

            if self.acq == 'SA':
                SA_reruns = 5
                penalty = lambda x: self.penalty * np.sum(x, axis=1)
                # Setup statistical model objective for SA
                stat_model = lambda x: LR.surrogate_model(x, alpha_t) + penalty(x)

                SA_model = np.zeros((SA_reruns, self.n_vars))
                SA_obj = np.zeros(SA_reruns)

                for j in range(SA_reruns):
                    (optModel, objVals) = self.simulated_annealing(stat_model)
                    SA_model[j, :] = optModel[-1, :]
                    SA_obj[j] = objVals[-1]

                # Find optimal solution
                min_idx = np.argmin(SA_obj)
                x_new = SA_model[min_idx, :]

            # Run semidefinite relaxation for order 2 model with l1 loss
            if self.acq == 'SDP-l1':
                x_new, _ = self.sdp_relaxation(alpha_t)
            else:
                solver = BQPSubmodular(self.n_vars, self.fix_num)
                A, b = self.get_A_b(alpha_t, self.n_vars)
                x_new, _ = solver.solve(-b, -A)


            y_new = self.fitness_evaluation(x_new)
            if y_new<mini:
                A_final = A
                b_final=b

            self.x_vals = np.vstack((self.x_vals, x_new.reshape(1,-1)))
            self.y_vals = np.append(self.y_vals, y_new)

            # re-train linear model
            LR.train(self.x_vals, self.y_vals)

            model_iter[t, :] = x_new

            if self.prior:
                y_new += evaluate_p_median(x_new, self.t_mat, self.frac_j)

            print(f'Iteration {t+1}/{n_iter}. x_eval: {x_new} , y_eval: {y_new}')
            obj_iter[t] = y_new
        print(A_final,b_final)

        bocs_opt = np.min(obj_iter)
        bocs_eval = model_iter[np.argmin(bocs_opt), :]
        return bocs_eval, obj_iter

    def get_A_b(self, alpha, n_vars):
        # Extract vector of coefficients
        # b = alpha[1:n_vars+1] + self.penalty	# first-order term
        b = alpha[1:n_vars + 1]  # first-order term
        a = alpha[n_vars + 1:]  # second-order term

        # get indices for quadratic terms
        idx_prod = np.array(list(combinations(np.arange(n_vars), 2)))
        n_idx = idx_prod.shape[0]

        # check number of coefficients
        if a.size != n_idx:
            raise ValueError('Number of Coefficients does not match indices!')

        # Convert $a$ to matrix form
        A = np.zeros((n_vars, n_vars))
        for i in range(n_idx):
            A[idx_prod[i, 0], idx_prod[i, 1]] = a[i] / 2.
            A[idx_prod[i, 1], idx_prod[i, 0]] = a[i] / 2.

        return A, b

    def sdp_relaxation(self, alpha):
        """
        SDP_Relaxation: Function runs simulated annealing algorithm for optimizing binary functions.
        The function returns optimum models and min objective values found at each iteration
        """
        n_vars = self.n_vars
        A, b = self.get_A_b(alpha, n_vars)
    
        # Convert to standard form
        bt = b/2. + np.dot(A,np.ones(n_vars))/2.
        bt = bt.reshape((n_vars,1))
        At = np.vstack((np.append(A/4., bt/2.,axis=1),np.append(bt.T,2.)))
    
        # Run SDP relaxation
        X = cvx.Variable((n_vars+1, n_vars+1), PSD=True)
        obj = cvx.Minimize(cvx.trace(cvx.matmul(At,X)))
        constraints = [cvx.diag(X) == np.ones(n_vars+1)]
        prob = cvx.Problem(obj, constraints)
        prob.solve(solver=cvx.CVXOPT)
    
        # Extract vectors and compute Cholesky
        try:
            L = np.linalg.cholesky(X.value)
        except:		# add small identity matrix is X.value is numerically not PSD
            XpI = X.value + 1e-15*np.eye(n_vars+1)
            L = np.linalg.cholesky(XpI)
    
        # Repeat rounding for different vectors
        n_rand_vector = 100
    
        model_vect = np.zeros((n_vars,n_rand_vector))
        obj_vect   = np.zeros(n_rand_vector)
        valid_num = 0
        kk = 0

        while kk < n_rand_vector or valid_num < 1:
            # Generate a random cutting plane vector (uniformly distributed on the unit sphere - normalized vector)
            r = np.random.randn(n_vars+1) + 1
            r = r/np.linalg.norm(r)
            rank_index = np.argsort(np.dot(L.T,r)[:n_vars])
            y_soln = np.ones(n_vars)
            for i in range(n_vars - self.fix_num):
                cur_index = rank_index[i]
                y_soln[cur_index] = 0
    
            model_vect[:,kk] = y_soln

            if sum(model_vect[:,kk]) == self.fix_num:
                valid_num += 1
                obj_vect[kk] = np.dot(np.dot(model_vect[:,kk].T,A),model_vect[:,kk]) \
                    + np.dot(b,model_vect[:,kk])
            else:
                obj_vect[kk] = sys.maxsize
            kk += 1

        # Find optimal rounded solution
        opt_idx = np.argmin(obj_vect)
        model = model_vect[:,opt_idx]
        obj   = obj_vect[opt_idx]
        return model, obj

    def simulated_annealing(self, objective):
        # SIMULATED_ANNEALING: Function runs simulated annealing algorithm for
        # optimizing binary functions. The function returns optimum models and min
        # objective values found at each iteration

        # Extract inputs
        n_vars = self.n_vars
        n_iter = self.evalBudget

        # Declare vectors to save solutions
        model_iter = np.zeros((n_iter, n_vars))
        obj_iter = np.zeros(n_iter)

        # Set initial temperature and cooling schedule
        T = 1.
        cool = lambda T: .8 * T

        # Set initial condition and evaluate objective
        old_x = sample_models_constraints(1, n_vars, self.fix_num)
        old_obj = objective(old_x)

        # Set best_x and best_obj
        best_x = old_x
        best_obj = old_obj

        # Run simulated annealing
        for t in range(n_iter):

            # Decrease T according to cooling schedule
            T = cool(T)
            new_x = sample_with_constraints(old_x)

            # Evaluate objective function
            new_obj = objective(new_x)

            # Update current solution iterate
            if (new_obj < old_obj) or (np.random.rand() < np.exp((old_obj - new_obj) / T)):
                old_x = new_x
                old_obj = new_obj

            # Update best solution
            if new_obj < best_obj:
                best_x = new_x
                best_obj = new_obj

            # save solution
            model_iter[t, :] = best_x
            obj_iter[t] = best_obj

        return model_iter, obj_iter

