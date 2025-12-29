import os
import pandas as pd
import numpy as np
import random
import time
import math
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import comb
from itertools import permutations,combinations,chain
from block_prob import *
# from alpha_hypercube import *

################# Public Functions #################
def ErlangLoss(Lambda, Mu, N = None):
    # The Erlangloss Model is exactly the same as MMN0 and this returns the whole probability distribution
    # Solves the Erlang loss system
    if N == 0: # if there is 0 unit. The block probability is 1
    	return [1]
    if N is not None: # If there is a size N, we constitute the Lambda and Mu vectors manually 
        Lambda = np.ones(N)*Lambda
        Mu = Mu*(np.arange(N)+1)
    else: # if the Lambdas and Mus are already given in vector form then no need to do anything
        N = len(Lambda)
    LoM = [1] + [l/m for l,m in zip(Lambda, Mu)]
    Prod = [np.prod(LoM[0:i]) for i in range(1,N+2)]
    P_n = Prod/sum(Prod)
    return P_n

def Loss_f(a, n): # This is the function for loss probability that is used to calculate P_b
    return a**n/math.factorial(n)

def Get_Effective_Lambda(L, Mu, N):
    # Get the effective lambda that gives the desired offered load
    # Using Newton's Method
    rho_ = 0
    # rho = L * (1+L/(N*(N-L)))
    rho = 1 # this is not used. Just an initialization that will be replaced by rho_ in the iteration
    step = 0
    while np.abs(rho-rho_) > 0.001:
        rho = rho_
        l = rho * Mu
        B = ErlangLoss(l, Mu, N)[-1] # the loss probability for the loss system
        #print('B', B)
        rho_ = rho - (rho*(1-B) - L)/(1-B-(N-rho+rho*B)*B)
        # print(rho)
        step += 1
        if step > 100:
            break
    Lambda_eff = rho * Mu
    return Lambda_eff


# def Cal_Response_Time(N, pol, frac_j, time_mat):
#     # This is the average response time for each state s
#     RT_list = np.zeros(2**N)
#     for s in range(2**N-1): # The last state has value 0
#         pol_s = pol[s]
#         dis_times = [time_mat[pol_s[j],j] for j in range(len(pol_s))]
#         RT_list[s] = -np.dot(frac_j,dis_times)
#     return RT_list


# def Cal_Time_Over(N, pol, frac_j, time_mat, T):
#     # First calculate the expected reward for each state
#     TimeOver_list = np.zeros(2**N)
#     for s in range(2**N-1): # The last state has value 0
#         pol_s = pol[s]
#         #print(pol_s)
#         dis_timesover = [time_mat[pol_s[j],j]>T for j in range(len(pol_s))]
#         # print(frac,dis_times)
#         TimeOver_list[s] = np.dot(frac_j,dis_timesover)
#     return TimeOver_list
###################################################

#################      Class      #################

class Two_State_Hypercube():
    def __init__(self, data_dict = None):
        # initilize data stored in self.data_dict
        self.keys = ['N', 'K', 'Lambda', 'Mu', 'frac_j', 't_mat', 'pre_list', 'pol']
        self.rho_hyper, self.rho_approx, self.rho_simu = None, None, None # initialize the utilizations to be None 
        self.prob_dist = None
        self.data_dict = dict.fromkeys(self.keys, None) 
        if data_dict is not None:
            for k, v in data_dict.items():
                if k in self.keys:
                    self.data_dict[k] = v
        self.G = None # G for approximation
        self.Q = None # Q for approximation
        self.r = None # r for approximation
        self.P_b = None # This is to differentiate from 3-state case
        self.q_nj = None # This is the dispatch probability using Linear-alph algorithm. This is shared by both 2-state and 3-state algorithms. 

    def Update_Parameters(self, **kwargs): 
        # update any parameters passed through kwargs
        for k, v in kwargs.items(): 
            if k in self.keys:
                self.data_dict[k] = v
                if k == 'pre_list': # reset G if pre_list changes
                    self.G = None
    
    def Random_Pref(self, seed=9001):
        # random preference list
        random.seed(seed) # Set Random Seed
        N, K = self.data_dict['N'], self.data_dict['K']
        # Shuffle the IDs as each preference list
        pre_list = np.array([random.sample(list(range(N)),N) for _ in range(K)])
        self.Update_Parameters(pre_list = pre_list)
    
    def Random_Fraction(self, seed=9001):
        '''
            Obtain random frac. 
        '''
        np.random.seed(seed)
        K = self.data_dict['K']
        frac_j = np.random.random(size=K)
        frac_j /= sum(frac_j)
        self.data_dict['frac_j'] = frac_j
        return frac_j

    def Random_Time_Mat(self, t_min = 1, t_max = 10, seed = 9001):
        np.random.seed(seed)
        N, K = self.data_dict['N'], self.data_dict['K']
        t_mat = np.random.uniform(low=t_min, high=t_max, size=(K,N))
        self.Update_Parameters(t_mat = t_mat)
        return t_mat

    def Myopic_Policy(self, source='t_mat'):
        # Obtain the policy to dispatch the cloest available unit given the time matrix
        N, K = self.data_dict['N'], self.data_dict['K']
        if source == 't_mat':
            t_mat = self.data_dict['t_mat']
            pre_list = t_mat.argsort(axis=1)
            self.Update_Parameters(pre_list = pre_list)
        elif source == 'pre':
            pre_list = self.data_dict['pre_list']
        else:
            print('Wrong source!')
        policy = np.zeros([2**N, K],dtype=np.int)
        for s in range(2**N):
            for j in range(K):
                pre = pre_list[j]
                for n in range(N):
                    if not s >> pre[n] & 1: # n th choice is free
                        policy[s, j] = pre[n]
                        break
        self.data_dict['pol'] = policy

    def Cal_Trans(self):
        keys = ['N', 'K', 'Lambda', 'Mu', 'pol', 'frac_j']
        N, K, Lambda, Mu, pol, frac_j = [self.data_dict.get(key) for key in keys]
        # Calculate the 
        N_state = 2**N
        A = np.zeros([N_state, N_state])
        # Calculate upward transtition
        for s in range(N_state-1): # The last state will not transition to other states by a arrival
            pol_s = pol[s]
            for j in range(K):
                dis = pol_s[j]
                A[s, s+2**dis] += Lambda * frac_j[j]
        # Calculate downward transtition
        for s in range(1,N_state): # The first state will not transition
            bin_s = bin(s)
            len_bin = len(bin_s)
            i = 0
            while bin_s[len_bin-1-i] != 'b':
                if bin_s[len_bin-1-i] == '1':
                    A[s,s-2**i] = Mu
                i += 1
        return A

    def Solve_Hypercube(self, update_rho = True):
        keys = ['N', 'K', 'Lambda', 'Mu', 'pol', 'frac_j']
        N, K, Lambda, Mu, pol, frac_j = [self.data_dict.get(key) for key in keys]
        # Get the transition Matrix 
        A = self.Cal_Trans()
        # Solve for the linear systems
        transition = A.T - np.diag(A.T.sum(axis=0))
        #print (np.linalg.det(transition))
        #print ('Eigenvalues of Transition Matrix:',np.linalg.eig(transition)[0])
        transition[-1] = np.ones(2**N)
        b = np.zeros(2**N)
        b[-1] = 1
        start_time = time.time() # staring time
        prob_dist = np.linalg.solve(transition,b)
        print("------ Hypercube run %s seconds ------" % (time.time() - start_time))
        if update_rho: # store the utilizations
            statusmat = [("{0:0"+str(N)+"b}").format(i) for i in range(2**N)]
            busy = [[N-1-j for j in range(N) if i[j]=='1'] for i in statusmat]
            rho = [sum([prob_dist[j] for j in range(2**N) if i in busy[j]]) for i in range(N)]
            self.rho_hyper = rho
        self.prob_dist = prob_dist
        return prob_dist
    
    def Get_MRT_Hypercube(self): # Method 1 of getting response time as in Larson
        keys = ['N', 'K', 'pol', 'frac_j', 't_mat']
        N, K, pol, frac_j, t_mat = [self.data_dict.get(key) for key in keys]
        prob_dist = self.prob_dist 
        # This is the average response time for each state s
        q_nj = np.zeros([K, N])
        for n in range(N): # The last state has value 0
            q_nj[:,n] = frac_j * np.dot(prob_dist[:-1], pol[:-1,:]==n) # here we don't need last state so take :-1
        q_nj /= (1-prob_dist[-1])
        self.q_nj = q_nj # store these values in the class
        MRT = np.sum(q_nj*t_mat)
        MRT_j = np.sum(q_nj*t_mat,axis = 1)/np.sum(q_nj, axis=1)
        return MRT, MRT_j

    def Get_LateCalls_Hypercube(self, threshold): # Method 1 of getting response time as in Larson
        keys = ['N', 'K', 'pol', 'frac_j', 't_mat']
        N, K, pol, frac_j, t_mat = [self.data_dict.get(key) for key in keys]
        prob_dist = self.prob_dist 
        # This is the average response time for each state s
        q_nj = np.zeros([K, N])
        for n in range(N): # The last state has value 0
            q_nj[:,n] = frac_j * np.dot(prob_dist[:-1], pol[:-1,:]==n) # here we don't need last state so take :-1
        q_nj /= (1-prob_dist[-1])
        self.q_nj = q_nj # store these values in the class
        tau = 1.75
        late_calls_matrix = t_mat > threshold - tau
        late_calls = np.sum(q_nj*late_calls_matrix)
        return late_calls

    # For Approximation
    def Cal_P_n(self):
        keys = ['N', 'Lambda', 'Mu']
        N, Lambda, Mu = [self.data_dict.get(key) for key in keys]
        P_n = ErlangLoss(Lambda, Mu, N)
        return P_n

    def Cal_Q(self, P_n = None):
        keys = ['N', 'Lambda', 'Mu']
        N, Lambda, Mu = [self.data_dict.get(key) for key in keys]
        if self.G is None:
            self.G = [[np.where(self.data_dict['pre_list'][:,i] == j)[0] for i in range(N)] for j in range(N)] 
        Q = np.zeros(N)
        if P_n is None:
            P_n = self.Cal_P_n()
        N = len(P_n) - 1
        # r = Lambda/(Mu*N) * (1-P_n[-1]) # two ways of calculating r. This one is wrong for general cases where P_N is not from erlang loss
        r = np.dot(P_n,np.arange(N+1))/N

        for j in range(N):
            Q[j] = sum([math.factorial(k)/math.factorial(k-j) * math.factorial(N-j)/math.factorial(N)* (N-k)/(N-j) * P_n[k] for k in range(j,N)])/ (r**(j) * (1-r))
        self.Q = Q
        self.r = r
        return Q

    def Larson_Approx(self, epsilon=0.0001): # this already integrates the linear-alpha with two places that have alpha in the code
        keys = ['N', 'Lambda', 'Mu', 'frac_j', 'pre_list']
        N, Lambda, Mu, frac_j, pre_list = [self.data_dict.get(key) for key in keys]
        try:
            alpha = self.alpha
        except:
            print('Two state!')
            alpha = 0

        use_effective_lambda = True # This part is not in the paper now. 
        # Step 0: Initialization
        self.Cal_Q() # This calculates Q, r, and G
        r = self.r # average fraction of busy time for each unit in the system. Calculated in Cal_Q
        # print(r)
        if self.P_b is not None:
            # print('All pooled')
            r = Lambda/(N*Mu)*(1-self.P_b) # P_b is the block probability. The probability that all units are busy

        rho_i = np.full(N,r) # utilization of each unit i. Initialization
        rho_i_ = np.zeros(N) # temporary utilizations to store test value at each step
        n = 0
        # Step 1: Iteration
        start_time = time.time()
        while True:
            n += 1 # increase step by 1
            rho_total = (rho_i + (1-rho_i)*alpha)
            ######################
            # Use the effective lambda to get the most accurate P_n and Q for each iteration (This helps a lot)
            if use_effective_lambda:
                if self.P_b is not None: 
                    L = rho_total.sum()
                    Lambda_eff = Get_Effective_Lambda(L, Mu, N)
                    P_n = ErlangLoss(Lambda_eff, Mu, N)
                    self.Cal_Q(P_n) # when use this Q, the old Q is overwritten and does not work
            ######################
            for i in range(N): # for each unit 
                value = 1 # 
                for k in range(N): # for each order
                    prod_g_j = 0 # Product term for each sum term
                    for j in self.G[i][k]:
                        prod_g_j += Lambda*frac_j[j]*self.Q[k]* np.prod(rho_total[pre_list[j,:k]]) # if alpha = 0, this is just rho in paranthesis
                    value += (1/Mu) * prod_g_j  # There should be a 1/mu here. because we don't assume it to be 1. 
                rho_i_[i]= (1-((1-rho_i)*alpha)[i])*(1-1/value) # again when alpha = 0, this is 1 in the paranthesis
            # Step 2: Normalize. 
            # Here the normalizing factor only takes on 1 service because we assume adding the other does not change much. 
            Gamma = rho_i_.sum()/(r*N)  # r is only used here for normalization
            rho_i_ /= Gamma
            # Step 3: Convergence Test
            if abs(rho_i_ - rho_i).max() < epsilon:
                print ('Program stop in',n,'iterations in ', (time.time() - start_time), 'secs')
                # print(rho_i_)
                self.rho_approx = rho_i_
                return rho_i_
            else: # go to next step
                rho_i = np.array(rho_i_)
                rho_i_ = np.zeros(N)

    def Get_MRT_Approx(self): # Method 1 of getting response time as in Larson
        keys = ['N', 'K', 'pre_list', 'frac_j', 't_mat', 'Mu']
        N, K, pre_list, frac_j, t_mat, Mu = [self.data_dict.get(key) for key in keys]
        try: # if self.alpha exists, it is three state. rho is the total rho
            rho = self.rho_total_approx
            print('Three state! Rho total is:', rho)
        except: # two state. rho is the normal approximate rho
            print('Two state!')
            rho = self.rho_approx
        
        P_n = self.Cal_P_n()
        # self.Cal_Q(P_n) ### 
        print('average rho:', self.r, np.mean(rho))
        Q = self.Q
        # This is the average response time for each state s
        q_nj = np.zeros([K, N])
        for j in range(K):
            pre_j = pre_list[j]
            for n in range(N):
                q_nj[j,pre_j[n]] = Q[n]*np.prod(rho[pre_j[:n]])*(1-rho[pre_j[n]])
            #print(q_nj[j,:].sum())
            q_nj[j,:] *= (1-P_n[-1])/q_nj[j,:].sum() # normalization
            q_nj[j,:] *= frac_j[j]

        print('sum of q_nj', q_nj.sum())
        # q_nj /= (1-P_n[-1])
        q_nj /= q_nj.sum() # same as divide by (1-P_allbusy)
        self.q_nj = q_nj # store these values in the class
        # print('q_nj', q_nj)
        MRT_j = np.sum(q_nj*t_mat,axis = 1)/np.sum(q_nj, axis=1)
        MRT = np.sum(q_nj*t_mat)
        return MRT, MRT_j

    def Get_ambulance_util(self): # Method 1 of getting response time as in Larson
        keys = ['N', 'K', 'pre_list', 'frac_j', 't_mat', 'Mu']
        N, K, pre_list, frac_j, t_mat, Mu = [self.data_dict.get(key) for key in keys]
        try: # if self.alpha exists, it is three state. rho is the total rho
            rho = self.rho_total_approx
            print('Three state! Rho total is:', rho)
        except: # two state. rho is the normal approximate rho
            print('Two state!')
            rho = self.rho_approx
        
        P_n = self.Cal_P_n()
        # self.Cal_Q(P_n) ### 
        print('average rho:', self.r, np.mean(rho))
        Q = self.Q
        # This is the average response time for each state s
        q_nj = np.zeros([K, N])
        for j in range(K):
            pre_j = pre_list[j]
            for n in range(N):
                q_nj[j,pre_j[n]] = Q[n]*np.prod(rho[pre_j[:n]])*(1-rho[pre_j[n]])
            #print(q_nj[j,:].sum())
            q_nj[j,:] *= (1-P_n[-1])/q_nj[j,:].sum() # normalization
            q_nj[j,:] *= frac_j[j]

        print('sum of q_nj', q_nj.sum())
        # q_nj /= (1-P_n[-1])
        q_nj /= q_nj.sum() # same as divide by (1-P_allbusy)
        self.q_nj = q_nj # store these values in the class
        # print('q_nj', q_nj)
        MRT_j = np.sum(q_nj*t_mat,axis = 1)/np.sum(q_nj, axis=1)
        MRT = np.sum(q_nj*t_mat)
        return q_nj

    def Get_LateCalls_Approx(self, threshold): # Method 1 of getting response time as in Larson
        keys = ['N', 'K', 'pre_list', 'frac_j', 't_mat', 'Mu']
        N, K, pre_list, frac_j, t_mat, Mu = [self.data_dict.get(key) for key in keys]
        try: # if self.alpha exists, it is three state. rho is the total rho
            rho = self.rho_total_approx
            print('Three state! Rho total is:', rho)
        except: # two state. rho is the normal approximate rho
            print('Two state!')
            rho = self.rho_approx
        
        P_n = self.Cal_P_n()
        # self.Cal_Q(P_n) ### 
        print('average rho:', self.r, np.mean(rho))
        Q = self.Q
        # This is the average response time for each state s
        q_nj = np.zeros([K, N])
        for j in range(K):
            pre_j = pre_list[j]
            for n in range(N):
                q_nj[j,pre_j[n]] = Q[n]*np.prod(rho[pre_j[:n]])*(1-rho[pre_j[n]])
            #print(q_nj[j,:].sum())
            q_nj[j,:] *= (1-P_n[-1])/q_nj[j,:].sum() # normalization
            q_nj[j,:] *= frac_j[j]

        print('sum of q_nj', q_nj.sum())
        # q_nj /= (1-P_n[-1])
        q_nj /= q_nj.sum() # same as divide by (1-P_allbusy)
        self.q_nj = q_nj # store these values in the class
        # print('q_nj', q_nj)
        tau = 1.75
        late_calls_matrix = t_mat > threshold - tau
        late_calls = np.sum(q_nj*late_calls_matrix)
        return late_calls

    def Get_Percent_Pref_Dispatch(self):# get percentage of percentage responsed by most preferred unit
        pre_list = self.data_dict['pre_list']
        preferred_units_j = pre_list[:,0] # mmost preferred units for each node j
        percent_pref_response = np.sum(self.q_nj[np.arange(len(self.q_nj)), preferred_units_j]) # get the corresponsing fraction for each node and sum up 
        # print('percent_pref_response',percent_pref_response)
        return percent_pref_response

    def Simulator(self, uptill = 500, T = 0, seed=9001): 
        keys = ['N', 'K', 'Lambda', 'Mu', 'pol', 'frac_j', 'time_mat']
        N, K, Lambda, Mu, policy, frac_j, time_mat = [self.data_dict.get(key) for key in keys]
        if time_mat is None:
            time_mat = np.zeros([N,K])
        # list indicating call locations
        k_list = np.array(random.choices(population=list(range(K)),weights=frac_j, k=uptill))
        # list indicating first arrival units
        serve_list = list(np.zeros(uptill))
        # list of busy units
        busy_list = []
        # Num lost call, and over threshold
        num_lost = 0 
        num_over = 0
        # Steady State Distribtion
        steady_state = np.zeros(2**N)
        # Utilization list
        utilization_list = np.zeros(N) # utilization for each unit
        # Response time list for each unit
        response_time_list = np.zeros(uptill)
        # Initialization
        inter_arrival = np.random.exponential(1/Lambda,size=uptill) # The inter arrival times for all arrivals
        arrival = np.cumsum(inter_arrival)  # The arrival times for all arrivals
        # List of service times for each call
        service_time = np.random.exponential(1/Mu,size=uptill)
        # List of completion times for each call
        completion_time = arrival+service_time # This assumes that the travel time is included in the service time
        # List of service completion order
        completion_order = completion_time.argsort()
        arrival_cursor = 0 # cursor in the arrival list
        completion_cursor = 0 # cursor in the completion list
        ################# Start simulation ################
        time = 0  # Starting at time 0
        while (completion_cursor<=uptill-1):
            while (arrival_cursor <= uptill-1) and arrival[arrival_cursor] < completion_time[completion_order[completion_cursor]]: # while we see an arrival
                ## There is an arrival at this time
                # first add the time to the past state
                state = sum(2**np.array(busy_list)) # current state number
                steady_state[state] += arrival[arrival_cursor]-time # Update the time to stay at this state
                # steady_state = time_add(steady_state, busy_list, arrival[arrival_cursor]-time)
                # Update the time to this arrival
                time = arrival[arrival_cursor] 
                # Get preference list 
                k = k_list[arrival_cursor] # This shows the location of the call
                dis_unit = policy[state, k] # the unit to be dispatched

                if state < 2**N - 1: # if not last state
                    busy_list += [dis_unit]
                    serve_list[arrival_cursor] = dis_unit
                    utilization_list[dis_unit] += service_time[arrival_cursor] # Add service time to this unit
                    response_time = time_mat[dis_unit,k]
                    response_time_list[arrival_cursor] = response_time
                    if response_time > T:
                        num_over += 1
                else: 
                    serve_list[arrival_cursor] = -1 # no server
                    num_lost += 1
                    response_time_list[arrival_cursor] = np.nan
                # Go to next arrival
                arrival_cursor += 1 # Go to next arrival
            #### If this next arrival exceeds the previous , then we move to service completion
            ### Service completion
            state = sum(2**np.array(busy_list)) # current state number
            steady_state[state] += completion_time[completion_order[completion_cursor]]-time 
            # steady_state = time_add(steady_state, busy_list, completion_time[completion_order[completion_cursor]]-time)
            # Update the time to this arrival
            time = completion_time[completion_order[completion_cursor]]
            # If the previous call was served, take the unit out of the busy list, else pass
            try:
                busy_list.remove(serve_list[completion_order[completion_cursor]])
            except: # For those require 2 units
                pass
            #print('busy_list:', busy_list)
            completion_cursor += 1
        #print(serve_list)
        #print(arrival)
        prob_dist = steady_state/sum(steady_state)
        rho = utilization_list/time
        MRT = np.nanmean(response_time_list)
        frac_overT = num_over/(uptill-num_lost)
        self.rho_simu = rho
        return prob_dist, rho, MRT, frac_overT

######################### Three state public function ###########################
def SumOfProduct(arr, k): # calculates the sum product of all combanitions in arr given size k
    n = len(arr) 
    # Initialising all the values to 0 
    dp = [ [ 0 for x in range(n + 1)] for y in range(n + 1)] 
    # To store the answer for 
    # current value of k 
    cur_sum = 0
    # For k = 1, the answer will simply 
    # be the sum of all the elements 
    for i in range(1, n + 1): 
        dp[1][i] = arr[i - 1] 
        cur_sum += arr[i - 1] 
    # Filling the table in bottom up manner 
    for i in range(2 , k + 1): 
        # To store the elements of the current 
        # row so that we will be able to use this sum 
        # for subsequent values of k 
        temp_sum = 0
        for j in range( 1,  n + 1): 
            # We will subtract previously computed value 
            # so as to get the sum of elements from j + 1 
            # to n in the (i - 1)th row 
            cur_sum -= dp[i - 1][j] 
   
            dp[i][j] = arr[j - 1] * cur_sum 
            temp_sum += dp[i][j] 
        cur_sum = temp_sum 
    sumprod_vec = np.array(dp).sum(axis=1) 
    return sumprod_vec
#################################################################################

################################## Classes #####################################

class Three_State_Hypercube():
    def __init__(self, data_dict = None):
        # initilize data stored in self.data_dict
        self.keys_1 = ['N', 'N_1', 'N_2', 'K', 'Lambda_1', 'Mu_1', 'frac_j_1', 't_mat_1', 'pre_list_1']
        self.keys_2 = ['N', 'N_1', 'N_2', 'K', 'Lambda_2', 'Mu_2', 'frac_j_2', 't_mat_2', 'pre_list_2']
        self.data_dict_1 = dict.fromkeys(self.keys_1, None) 
        self.data_dict_2 = dict.fromkeys(self.keys_2, None) 
        if data_dict is not None:
            for k, v in data_dict.items():
                if k in self.keys_1:
                    self.data_dict_1[k] = v
                if k in self.keys_2:
                    self.data_dict_2[k] = v
        self.time_exact = None
        self.time_alphahypercube = None # MRT obtained from alpha hypercube
        self.time_linearalpha = None # MRT obtained from linear alpha
        self.prob_dist_3state = None # steadt state distribution when solved exactly by 3state hypercube
        self.pol_sub1 = None # policy for service 1 for exact 3 state system
        self.pol_sub2 = None # policy for service 2 for exact 3 state system

    def Update_Parameters(self, **kwargs): 
        # update any parameters passed through kwargs
        for k, v in kwargs.items(): 
            if k in self.keys_1:
                self.data_dict_1[k] = v
            if k in self.keys_2:
                self.data_dict_2[k] = v

    def Update_alpha(self, method, subsystem):
        keys = ['N', 'N_1', 'N_2']
        N, N_1, N_2 = [self.data_dict_1.get(key) for key in keys]
        N_sub1, N_sub2 =  N - N_2, N - N_1
        # Specify method
        if method in 'exact':
            rho_sub1, rho_sub2 = self.sub1.rho_hyper, self.sub2.rho_hyper
        elif method in 'approximation':
            rho_sub1, rho_sub2 = self.sub1.rho_approx, self.sub2.rho_approx
        else:
            print('Wrong method')
        # Specify subsystem 
        if subsystem == 1:
            rho = rho_sub1
            alpha = self.sub1.alpha
            N_sub_o = N_sub2
            N_me = N_1
            N_o = N_2
        elif subsystem == 2:
            rho = rho_sub2
            alpha = self.sub2.alpha
            N_sub_o = N_sub1
            N_me = N_2
            N_o = N_1
        else:
            print('Wrong subsystem!')
        alpha_ = np.array([0] * N_o + [rho[n+N_me-N_o]/(rho[n+N_me-N_o]+(1-rho[n+N_me-N_o])*(1-alpha[n+N_me-N_o])) for n in range(N_o,N_sub_o)])
        return alpha_

    def Solve_3state_Hypercube(self): # Now works for cases 2021.9.6
        '''
            :Data: Input data setting
            Output: Solve 3-state hypercube directly. 
        '''
        # Parameters
        keys = ['N', 'N_1', 'N_2', 'K']
        N, N_1, N_2, K = [self.data_dict_1.get(key) for key in keys]
        keys_sub = ['Lambda', 'Mu', 'frac_j', 'pre_list']
        Lambda_1, Mu_1, frac_j_1, pre_list_1 = [self.sub1.data_dict.get(key) for key in keys_sub]
        Lambda_2, Mu_2, frac_j_2, pre_list_2 = [self.sub2.data_dict.get(key) for key in keys_sub]
        N_sub1, N_sub2 =  N - N_2, N - N_1

        pre_list_1_3state = pre_list_1.copy() # make a copy of the preference list so that the original will not be modified
        pre_list_1_3state[pre_list_1_3state >= N_1] += N_2
        pre_list_2_3state = pre_list_2 + N_1

        start_time = time.time() # staring time
        # Initialize States for each subsystem
        Num_State = 2**(N_1+N_2) * 3**(N-N_1-N_2)
        pol_sub1, pol_sub2 = np.ones([Num_State, K],dtype=int)*-1, np.ones([Num_State, K],dtype=int)*-1

        statusmat_sep = [np.base_repr(i,base=2)[1:] for i in range(2**(N_1+N_2),2**(N_1+N_2+1))] # Pad a 1 in front to make the length and take out the 1
        statusmat_joint = [np.base_repr(i,base=3)[1:] for i in range(3**(N-N_1-N_2),2*3**(N-N_1-N_2))] # Pad a 1 in front to make the length and take out the 1
        statusmat = [y+x for y in statusmat_joint for x in statusmat_sep] # Get all the transitions
        # print(statusmat)
        transition = np.zeros([Num_State,Num_State]) # Initialize the transition matrix

        # Update upward transition rate
        for j in range(K): # Loop through every atom
            for n in range(Num_State): # Loop through states
                B_n = statusmat[n]
                # For EMS
                for i in pre_list_1_3state[j]: # find the one in the preference to add the rate to the state
                    if B_n[N-1-i] == '0': # find the first available unit N-1-i shows unit i in the binary representation
                        pol_sub1[n, j] = i # assign to policy
                        if i < N_1+N_2: # if it is a separate unit
                            m = n + 2**i # the state transit to 
                        else: # if it is a joint unit
                            m = n + 2**(N_1+N_2) * 3**(i-N_1-N_2) # the state transit to 
                        transition[m,n] += Lambda_1*frac_j_1[j]
                        break

                # For fire 
                for i in pre_list_2_3state[j]: # find the one in the preference to add the rate to the state
                    if B_n[N-1-i] == '0': # find the first available unit
                        pol_sub2[n, j] = i # assign to policy
                        if i < N_1+N_2: # if it is a separate unit
                            m = n + 2**i # the state transit to 
                        else: # if it is a joint unit
                            m = n + 2**(N_1+N_2) * 2*3**(i-N_1-N_2) # the state transit to 
                        transition[m,n] += Lambda_2*frac_j_2[j]
                        break


        # Upward downward transition rate
        for n in range(Num_State): # Loop through states 
            B_n = statusmat[n]
            for i in range(N_1): # Loop through every separate type-1
                if B_n[N-1-i] == '1':
                    m = n - 2**i
                    transition[m,n] = Mu_1
            for i in range(N_1,N_1+N_2): # Loop through every separate type-2
                if B_n[N-1-i] == '1':
                    m = n - 2**i
                    transition[m,n] = Mu_2
            for i in range(N_1+N_2,N): # Loop through every joint
                if B_n[N-1-i] == '1': # for type-1
                    m = n - 2**(N_1+N_2) * 3**(i-N_1-N_2) 
                    transition[m,n] = Mu_1
                elif B_n[N-1-i] == '2': # for type-2
                    m = n - 2**(N_1+N_2) * 2*3**(i-N_1-N_2) 
                    transition[m,n] = Mu_2

        # print(transition)
        self.pol_sub1 = pol_sub1
        self.pol_sub2 = pol_sub2
        # Set diagonal
        diag = np.diag(transition.sum(axis=0))
        transition -= diag
        # Solve for the steady state equation
        transition[-1] = np.ones(Num_State)
        # print(transition)
        b = np.zeros(Num_State)
        b[-1] = 1
        #prob_dist = np.linalg.solve(transition,b) # linear solve method
        transition_sparse = sparse.csc_matrix(transition)
        prob_dist = spsolve(transition_sparse,b) # sparse solve method
        self.prob_dist_3state = prob_dist
        print("------ %s seconds ------" % (time.time() - start_time))
        total_time = time.time() - start_time
        self.time_exact = total_time
        # print(prob_dist)
        # Get rho
        rho_1, rho_2 = np.zeros(N-N_2), np.zeros(N-N_1) # inialize 
        for n in range(Num_State):
            B_n = statusmat[n]
            #print(B_n)
            for i in range(N_1): # separate type-1
                if B_n[N-1-i] == '1':
                    rho_1[i] += prob_dist[n]
            for i in range(N_1, N_1+N_2): # separate type-2 
                if B_n[N-1-i] == '1':
                    rho_2[i-N_1] += prob_dist[n]
            for i in range(N_1+N_2,N): # joint unit
                if B_n[N-1-i] == '1':
                    rho_1[i-N_2] += prob_dist[n]  # serve type-1
                elif B_n[N-1-i] == '2':       # serve type-2
                    rho_2[i-N_1] += prob_dist[n]    
        return rho_1, rho_2

    def Get_MRT_3state(self):
        # Parameters
        keys = ['N', 'N_1', 'N_2', 'K']
        N, N_1, N_2, K = [self.data_dict_1.get(key) for key in keys]
        keys_sub = ['frac_j', 't_mat']
        frac_j_1, t_mat_1 = [self.sub1.data_dict.get(key) for key in keys_sub]
        frac_j_2, t_mat_2 = [self.sub2.data_dict.get(key) for key in keys_sub]
        N_sub1, N_sub2 =  N - N_2, N - N_1

        not_all_busy_states_1 = np.unique(np.where(self.pol_sub1 != -1)[0])
        not_all_busy_states_2 = np.unique(np.where(self.pol_sub2 != -1)[0])

        q_nj_1 = np.zeros([K, N_sub1])

        list_sub1 = np.arange(N_sub1)
        add_ind = np.zeros(N_sub1,dtype=int)
        add_ind[N_1:] = N_2
        list_sub1 = list_sub1 + add_ind
        for n in np.arange(N_sub1): # The last state has value 0
            q_nj_1[:,n] = frac_j_1 * np.dot(self.prob_dist_3state[not_all_busy_states_1], self.pol_sub1[not_all_busy_states_1,:]==list_sub1[n]) # here we don't need last state so take :-1
        print('sum of q_nj_1', q_nj_1.sum())
        q_nj_1 /= q_nj_1.sum() # Same as divide by (1-P_allbusy)
        MRT_1 = np.sum(q_nj_1*t_mat_1)

        q_nj_2 = np.zeros([K, N_sub2])
        list_sub2 = np.arange(N_sub2)+N_1
        for n in np.arange(N_sub2): # The last state has value 0
            q_nj_2[:,n] = frac_j_2 * np.dot(self.prob_dist_3state[not_all_busy_states_2], self.pol_sub2[not_all_busy_states_2,:]==list_sub2[n]) # here we don't need last state so take :-1
        print('sum of q_nj_2', q_nj_2.sum())
        q_nj_2 /= q_nj_2.sum() # Same as divide by (1-P_allbusy)
        MRT_2 = np.sum(q_nj_2*t_mat_2)
        return MRT_1, MRT_2

    def Creat_Two_Subsystems(self):
        self.sub1 = self.Subsystem(dict((key[:-2], value) for (key, value) in self.data_dict_1.items() if len(key)>3)) # take _1 off
        self.sub2 = self.Subsystem(dict((key[:-2], value) for (key, value) in self.data_dict_2.items() if len(key)>3)) # take _2 off
        N_sub1 = self.data_dict_1['N'] - self.data_dict_1['N_2']
        N_sub2 = self.data_dict_2['N'] - self.data_dict_2['N_1']
        K = self.data_dict_1['K']
        self.sub1.Update_Parameters(N = N_sub1, K = K)
        self.sub2.Update_Parameters(N = N_sub2, K = K)
        self.sub1.alpha = np.zeros(N_sub1)
        self.sub2.alpha = np.zeros(N_sub2)

    def Reset_Alpha(self):
        keys = ['N', 'N_1', 'N_2']
        N, N_1, N_2= [self.data_dict_1.get(key) for key in keys]
        N_sub1 = N - N_2
        N_sub2 = N - N_1
        self.sub1.alpha = np.zeros(N_sub1)
        self.sub2.alpha = np.zeros(N_sub2)

    def Alpha_Hypercube(self, epsilon = 0.0001):
        ite = 0
        run = True
        start_time = time.time()
        while run:
            ite += 1
            # subsystem 1
            self.sub1.Solve_Hypercube()
            alpha = self.Update_alpha(method='exact', subsystem=1)
            self.sub2.alpha = alpha
            # subsystem 2
            self.sub2.Solve_Hypercube()
            alpha = self.Update_alpha(method='exact', subsystem=2)
            if (max(abs(alpha - self.sub1.alpha)) < epsilon):
                run = False
            self.sub1.alpha = alpha
        print("------ Alpha Hypercube run %s seconds ------" % (time.time() - start_time))
        self.time_alphahypercube = time.time() - start_time
        print('Number of iteration:', ite)

    def Linear_Alpha(self, normalize = True, epsilon = 0.0001):
        if normalize: # if we normalize rho after each iteration. In this function, default is to normalize
            self.Cal_P_b() # calculate the block probability, which in turn gives average utility. This also initializes the intial alpha that is fast for computation
        ite = 0
        run = True
        start_time = time.time()
        while run:
            ite += 1
            self.sub1.Larson_Approx()
            alpha = self.Update_alpha(method='approx', subsystem=1)
            self.sub2.alpha = alpha
            print('rho_1',self.sub1.rho_approx)
            print('alpha_1',alpha)
            # subsystem 2
            self.sub2.Larson_Approx()
            alpha = self.Update_alpha(method='approx', subsystem=2)
            print('rho_2',self.sub2.rho_approx)
            print('alpha_2',alpha)
            if (max(abs(alpha - self.sub1.alpha)) < epsilon):
                run = False
            self.sub1.alpha = alpha
        print("------ Linear Alpha run %s seconds ------" % (time.time() - start_time))
        self.time_linearalpha = time.time() - start_time
        print('Number of iteration:', ite)
        self.ite = ite

    def Cal_P_b(self): # Calculate the block probability of the two subsystems. 
    # as a consequence, this might make linear-alpha slow. One alternative is to do the update as before without normalize like this
    # and then normalize after convergence. 
        keys = ['N', 'N_1', 'N_2']
        N, N_1, N_2 = [self.data_dict_1.get(key) for key in keys]
        keys_sub = ['Lambda', 'Mu']
        Lambda_1, Mu_1 = [self.sub1.data_dict.get(key) for key in keys_sub]
        Lambda_2, Mu_2 = [self.sub2.data_dict.get(key) for key in keys_sub]

        if N_1 == N_2 == 0: # all cross-trained
            P_b = ErlangLoss(Lambda_1+Lambda_2, (Lambda_1+Lambda_2)/(Lambda_1/Mu_1+Lambda_2/Mu_2), N)[-1] # blocking probability
            self.sub1.P_b, self.sub2.P_b = P_b, P_b # update the two blocking probabilities
            r_1, r_2 = Lambda_1/(N*Mu_1)*(1-P_b), Lambda_2/(N*Mu_2)*(1-P_b) # This is the average utilizations of each service
            self.sub1.alpha = np.ones(N) * r_2 / (1-r_1) # initialize alpha this way is faster
            self.sub2.alpha = np.ones(N) * r_1 / (1-r_2)
        else: # general case, This includes all cross-trained case but a little bit more complicated
            N_c = N - N_1 - N_2
            P_b1_1, P_b1_2 = P_b1(Lambda_1, Lambda_2, Mu_1, Mu_2, N_1, N_2, N)
            P_b2_1, P_b2_2 = P_b2(Lambda_1, Lambda_2, Mu_1, Mu_2, N_1, N_2, N)
            P_b_1, P_b_2 = (P_b1_1+P_b2_1)/2, (P_b1_2+P_b2_2)/2
            self.sub1.P_b, self.sub2.P_b = P_b_1, P_b_2 # update the two blocking probabilities
            r_1, r_2 = Lambda_1/((N-N_2)*Mu_1)*(1-P_b_1), Lambda_2/((N-N_1)*Mu_2)*(1-P_b_2) # This is the average utilizations of each service
            if r_1 + r_2 > 1: # This ensures that the intializes alpha to be less than 1
                r_1, r_2 = r_1/(r_1 + r_2), r_2//(r_1 + r_2) 
            self.sub1.alpha = np.concatenate((np.zeros(N_1), np.ones(N_c) * r_2 / (1-r_1))) # initialize alpha this way is faster. 0s for the separate units
            self.sub2.alpha = np.concatenate((np.zeros(N_1), np.ones(N_c) * r_1 / (1-r_2)))

    def Get_MRT_Approx_3state(self):
        keys = ['N', 'N_1', 'N_2']
        N, N_1, N_2 = [self.data_dict_1.get(key) for key in keys]
        rho_1_approx = self.sub1.rho_approx
        rho_2_approx = self.sub2.rho_approx

        self.sub1.rho_total_approx = rho_1_approx + np.append(np.zeros(N_1),rho_2_approx[N_2:])
        self.sub2.rho_total_approx = rho_2_approx + np.append(np.zeros(N_2),rho_1_approx[N_1:])

        MRT_1, MRT_1_j = self.sub1.Get_MRT_Approx()
        MRT_2, MRT_2_j = self.sub2.Get_MRT_Approx()
        return MRT_1, MRT_2, MRT_1_j, MRT_2_j

    def Simulator_3state(self, uptill = 500, T = 0, seed=9001):
        ############## load data ################
        np.random.seed(seed)
        keys = ['N', 'N_1', 'N_2', 'K']
        N, N_1, N_2, K = [self.data_dict_1.get(key) for key in keys]
        keys_sub = ['Lambda', 'Mu', 'frac_j', 'pre_list', 't_mat']
        Lambda_1, Mu_1, frac_j_1, pre_list_1, t_mat_1 = [self.sub1.data_dict.get(key) for key in keys_sub]
        Lambda_2, Mu_2, frac_j_2, pre_list_2, t_mat_2 = [self.sub2.data_dict.get(key) for key in keys_sub]
        Lambda_1_2, Lambda_2_2 = 0,0 # no double dispatch
        # change preference list to incorporate separate units
        pre_list_1_simu = pre_list_1.copy() # make a copy of the preference list so that the original will not be modified
        pre_list_1_simu[pre_list_1_simu >= N_1] += N_2
        pre_list_2_simu = pre_list_2 + N_1
        ##########################################

        start_time = time.time()
        # Initialization
        Lambda = Lambda_1+Lambda_2+Lambda_1_2+Lambda_2_2 # Total Arrival Rate
        inter_arrival = np.random.exponential(1/Lambda,size=uptill) # The inter arrival times for all arrivals
        arrival = np.cumsum(inter_arrival)  # The arrival times for all arrivals
        
        busy = [] # Set of busy units
        p = np.array([Lambda_1,Lambda_2,Lambda_1_2,Lambda_2_2])/Lambda # Call probabilities
        p = np.cumsum(p) # Cumulative probabilities
        num_lost = 0 # number of lost calls
        # Generate random calls following the above probabilities
        random_list = np.random.random(size=uptill)
        # Revise probabilities, mu_list and k_list
        arrival_1 = random_list<p[0] # arrival_1 is EMS
        arrival_3 = random_list>p[2] # arrival_2 is fire, arrival_3 is fire requries 2 dispatches

        # List of service rates mu
        mu_list = np.ones(uptill)*Mu_2
        mu_list[arrival_1] = Mu_1 # change back 
        # list indicating call locations
        k_list = np.array(random.choices(population=list(range(K)),weights=frac_j_2, k=uptill))
        k_list[arrival_1] = np.array(random.choices(population=list(range(K)),weights=frac_j_1, k=sum(arrival_1))) 
        # list of type of calls 1:EMS 2:fire_single 3:fire_multiple
        type_list = np.ones(uptill)*2
        type_list[arrival_1] = 1
        type_list[arrival_3] = 3

        ############## Initialization ##############
        # list indicating first arrival units
        serve_list = list(np.zeros(uptill))
        # list showing the rank of preferred units is dispatched
        dispatch_pre_list = np.zeros(uptill)
        # List of service times for each call
        service_time = np.random.exponential(1/mu_list,size=uptill)
        # List of completion times for each call
        completion_time = arrival+service_time
        # List of service completion order
        completion_order = completion_time.argsort()
        arrival_cursor = 0
        completion_cursor = 0
        # list of busy units
        busy_list = []
        #############################################
        # Total time
        total_time = completion_time[-1]

        while (completion_cursor<=uptill-1):
            while (arrival_cursor <= uptill-1) and arrival[arrival_cursor] < completion_time[completion_order[completion_cursor]]:
                # Get preference list 
                k = k_list[arrival_cursor]
                if type_list[arrival_cursor] == 1: 
                    pre = pre_list_1_simu[k]
                else:
                    pre = pre_list_2_simu[k]
                
                # Get serving units
                i = 0
                while i < len(pre) and (pre[i] in busy_list): # while there is an available unit
                    i += 1
                try: # If exceeds, then means all units are busy
                    busy_list += [pre[i]]   # update the busy units
                    serve_list[arrival_cursor] = [pre[i]]  # show the unit id that is serving
                    dispatch_pre_list[arrival_cursor] = i # show the rank of preference
                except: # if all units are busy, mark those as -1
                    serve_list[arrival_cursor] = -1
                    dispatch_pre_list[arrival_cursor] = -1
                    num_lost += 1
                
                # For dispatching 2 units
                if type_list[arrival_cursor] == 3 and i < len(pre):
                    while i < len(pre) and pre[i] in busy_list:
                        i += 1
                    try:
                        busy_list += [pre[i]]
                        #print(serve_list[arrival_cursor])
                        serve_list[arrival_cursor].append(pre[i])
                    except:
                        #serve_list[arrival_cursor] = -1
                        #num_lost += 1
                        pass
                
                # Go to next arrival
                arrival_cursor += 1

            # Service completion
            try:
                busy_list.remove(serve_list[completion_order[completion_cursor]])
            except: # For those require 2 units
                try:
                    busy_list.remove(serve_list[completion_order[completion_cursor]][0])
                    busy_list.remove(serve_list[completion_order[completion_cursor]][1])
                except:
                    pass
            completion_cursor += 1

        def calculate_utilization(type_list,k_list,serve_list,N,total_time,service_time):
            # List for utilizations
            utlization_list_1 = np.zeros(N)
            utlization_list_2 = np.zeros(N)
            for i in range(len(serve_list)):
                servers = serve_list[i]
                if servers != -1:
                    if type_list[i] == 1:
                        for server in servers:
                            utlization_list_1[server] += service_time[i]
                    else:
                        for server in servers:
                            utlization_list_2[server] += service_time[i]
            rho_1, rho_2 = utlization_list_1/total_time, utlization_list_2/total_time
            return rho_1, rho_2  # Utilizations

        def calculate_mean_response_time(type_list,k_list,serve_list,t_mat_1, t_mat_2, N_1, N_2):
            sub1_id = np.where(np.logical_and(type_list == 1, (np.array(serve_list) != -1).reshape(-1))) # for those that gets service
            sub2_id = np.where(np.logical_and(type_list != 1, (np.array(serve_list) != -1).reshape(-1)))

            sum_response_time_1 = 0
            sum_response_time_2 = 0
            # Get total response time for each type of call
            for i in sub1_id[0]:
                serve_unit = serve_list[i][0]
                if serve_unit >=  N_1:
                    serve_unit -= N_2
                sum_response_time_1 += t_mat_1[k_list[i],serve_unit]
            for i in sub2_id[0]:
                serve_unit = serve_list[i][0] - N_1
                sum_response_time_2 += t_mat_2[k_list[i],serve_unit]

            # Get mean response time 
            try:
                mean_response_time_1 = sum_response_time_1/len(sub1_id[0])
            except:
                mean_response_time_1 = 0
            try:
                mean_response_time_2 = sum_response_time_2/len(sub2_id[0])
            except:
                mean_response_time_2 = 0
            # Store in a text file
            return mean_response_time_1, mean_response_time_2

        MRT_1, MRT_2 = calculate_mean_response_time(type_list,k_list,serve_list, t_mat_1, t_mat_2, N_1, N_2)
        rho_1, rho_2 = calculate_utilization(type_list,k_list,serve_list,N,total_time,service_time)
        total_time = time.time() - start_time
        return rho_1, rho_2, MRT_1, MRT_2, total_time

    class Subsystem(Two_State_Hypercube): # This class belongs to the three-state class and is a children class of Two_state_hyper
        def __init__(self, data_dict = None):
            super().__init__(data_dict = data_dict)
            self.alpha = None # initilize alpha to be none. The alpha value for this subsystem. This alpha is intialized in the Cal_P_b function.
            self.rho_total_approx = None # total rho for this subsystem when calculated by approximation. rho_1+rho_2 for joint

        def Cal_Trans(self): # This contains alpha. Overwrites the original function in parental class
            # This is not the most efficient. Good enough for now. 
            keys = ['N', 'K', 'Lambda', 'Mu', 'pre_list', 'frac_j']
            N, K, Lambda, Mu, pre_list, frac_j = [self.data_dict.get(key) for key in keys]
            alpha = self.alpha

            Num_state = 2**N
            A = np.zeros([Num_state,Num_state]) # Initilize 

            ######################
            # Try the correction factor in alpha-hypercube. 
            # Have not implemented yet.
            # L = rho_total.sum()
            # Lambda_eff = Get_Effective_Lambda(L, Mu, N)
            # P_n = ErlangLoss(Lambda_eff, Mu, N)
            # self.Cal_Q(P_n)
            ######################

            # Calculate upward transition
            for s in range(Num_state - 1):
                for j in range(K):
                    pre_list_j = pre_list[j]
                    alpha_prod = 1
                    for k in range(N): # find states in which kth preferred unit is free
                        unit = pre_list_j[k]
                        if not s & (1 << unit): # if it is free
                            s_ = s ^ (1 << unit) # The state it transitions to
                            A[s, s_] += Lambda * frac_j[j] * alpha_prod * (1-alpha[unit])
                            alpha_prod *= alpha[unit]
                            A[s_, s] = Mu
            return A

        def Cal_P_n(self): # this overwrites the P_n in the parental class
            '''
                :Lambda_v, Mu_v, alpha: Inputs
                "Output: P_n
                This one assumes every combination is with equal probability. Getting the steady state probability P_n
            '''
            keys = ['N', 'Lambda', 'Mu']
            N, Lambda, Mu = [self.data_dict.get(key) for key in keys]
            # alpha = self.alpha

            Lambda_BD = np.ones(N)*Lambda # Lambdas of the birth and death chain
            sumprod_vec = SumOfProduct(self.alpha, N) # this calculates all the combinations of sum of product of alpha's in one run. Significantly reduce TIME. 
            for i in range(N):
                num_comb = comb(N,i+1) # number of totoal combinations when in total i+1 units busy
                Lambda_BD[N-i-1] = Lambda/num_comb*(num_comb-sumprod_vec[i+1]) # Using the equal probability assmption, this is equation lambda(k) in Thm 1
                # When there is no much difference of lambdas, we just assume it is Lambda rather than do massive computation
                if Lambda - Lambda_BD[N-i-1] < 0.001:
                    break
            #print('Lambda_BD',Lambda_BD)
            Mu_BD = Mu*(np.array(range(N))+1)
            P_n = ErlangLoss(Lambda_BD, Mu_BD)
            return P_n

        def Get_MRT_Hypercube(self): # overwrites the one for 2-class cases. Much more complicated because cannot directly use pol
            keys = ['N', 'K', 'pre_list', 'frac_j','t_mat','Mu']
            N, K, pre_list, frac_j, t_mat, Mu = [self.data_dict.get(key) for key in keys]
            alpha = self.alpha
            prob_dist = self.prob_dist
            Num_state = 2**N
            
            q_nj = np.zeros([K, N]) # probability of assigning unit i to node j
            for s in range(Num_state-1):
                ###################
                # alpha_vec = alpha[[i for i in range(N) if not s & (1 << i)]]
                # L = np.array(alpha_vec).sum()
                # Lambda_eff = Get_Effective_Lambda(L, Mu, len(alpha_vec))
                # # print(Lambda_eff)
                # P_n = ErlangLoss(Lambda_eff, Mu, len(alpha_vec))
                # # print(P_n)
                # Q = self.Cal_Q(P_n)
                ###################
                for j in range(K):
                    pre_list_j = pre_list[j]
                    alpha_prod = 1
                    # i = 0
                    for k in range(N): # find states in which kth preferred unit is free
                        unit = pre_list_j[k]
                        if not s & (1 << unit): # if it is free
                            # print(Q[i])
                            q_nj[j,unit] += frac_j[j] * alpha_prod * (1-alpha[unit]) * prob_dist[s]
                            alpha_prod *= alpha[unit] # this is to capture states that has 1 there so it will not be dispatched
                            # i += 1
            print('sum of q_nj', q_nj.sum())
            q_nj /= q_nj.sum() # this is the same as divide by (1-P_allbusy)
            MRT = np.sum(q_nj*t_mat)
            MRT_j = np.sum(q_nj*t_mat,axis = 1)/np.sum(q_nj, axis=1)
            return MRT, MRT_j

