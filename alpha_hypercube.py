#############################################################################################
####################### PACKAGES ############################################################
import numpy as np
import pandas as pd
import math
from scipy import sparse
from scipy.sparse.linalg import spsolve
import random
import time
import pickle
from scipy.special import comb
import time
# import pysnooper
from itertools import permutations,combinations,chain
import scipy.special
#import matplotlib.pyplot as plt
import itertools

#############################################################################################
####################### UNIVERSAL FUNCTIONS #################################################
def parameter(File_Name = '0_2', Lambda= None, Mu = None, e = None, f = None, Pre_L_1 = None, Pre_L_2 = None, sort = True):
	'''
		:File_Name: File that stored the parameter for the runs
		:Lambda, Mu, e, f, Pre_L_1, Pre_L_2: the date of this collection
		Output: Data
	'''
	file = open(File_Name+'.pkl', 'rb')
	Data = pickle.load(file)
	if Lambda != None:
		Data['Lambda_1'], Data['Lambda_2'] = Lambda
	if Mu != None:
		Data['Mu_1'], Data['Mu_2'] = Mu
	if e != None:
		Data['e'] = np.array(e)
	if f != None:
		Data['f'] = np.array(f)
	if np.any(Pre_L_1 != None):
		Data['Pre_L_1'] = Pre_L_1
	if np.any(Pre_L_2 != None):
		Data['Pre_L_2'] = Pre_L_2
	N, N_1, N_2= Data['N'], Data['N_1'], Data['N_2']
	if sort: # If want to perform argsort
		Data['Pre_L_1'] = Data['Pre_L_1'].argsort().argsort()
		Data['Pre_L_2'] = Data['Pre_L_2'].argsort().argsort()
	file.close()
	return Data

def Random_Data(N,N_1,N_2,K,Lambda_1,Lambda_2,Mu_1,Mu_2,seed=9001):
	'''
		:N,N_1,N_2,K: Inputs
		Output: Data
		Obtain a random Data set
	'''
	# Set Random Seed
	Data = {'K': K,
	 'Lambda_1': Lambda_1,
	 'Lambda_2': Lambda_2,
	 'Mu_1': Mu_1,
	 'Mu_2': Mu_2,
	 'N': N,
	 'N_1': N_1,
	 'N_2': N_2}
	Data['e'], Data['f'] = Random_Fraction(K,seed=seed)
	Data['Pre_L_1'],Data['Pre_L_2'] = Random_Pref(N,N_1,N_2,K,seed=seed)
	return Data

def Random_Pref(N,N_1,N_2,K,seed=9001):
	'''
		:N,N_1,N_2,K: Inputs
		Output: Pre_L_1, Pre_L_2
		Obtain a random preference list given inputs. 
	'''
	# Set Random Seed
	random.seed(seed)
	# ID of separate units
	Sep_med = list(range(1,N_1+1))
	Sep_fire = list(range(N_1+1,N_1+N_2+1))
	# Get ID of each subsystem
	Pre_L_1 = [[x for x in range(1,N+1) if x not in Sep_fire]]*K
	Pre_L_2 = [[x for x in range(1,N+1) if x not in Sep_med]]*K
	# Shuffle the IDs as each preference list
	Pre_L_1 = [random.sample(x,len(x)) for x in Pre_L_1]
	Pre_L_2 = [random.sample(x,len(x)) for x in Pre_L_2]
	return np.array(Pre_L_1)-1, np.array(Pre_L_2)-1

def Random_Fraction(K,seed=9001):
	'''
		Obtain random e and f. 
	'''
	np.random.seed(seed)
	e = np.random.random(size=K)
	e = e/sum(e)
	f = np.random.random(size=K)
	f = f/sum(f)
	return e,f

def Random_Response_Time(N, K, MaxT = 20, seed=9001):
	np.random.seed(seed)
	time_mat = np.random.rand(N,K)*MaxT
	return time_mat

def jacobi_method(A, b):
	'''
		Jacobi method for Solve A^-1 b. 
	'''
	D = np.diag(A)
	R = A - np.diagflat(D)
	x = np.zeros(len(b))
	x_new = np.ones(len(b))
	while(max(np.abs(x-x_new)>0.00001)):
		x = np.copy(x_new)
		x_new = (b - np.dot(R,x))/ D
	return x_new

def powerset(iterable):
	'''
		Generates the powerset of the given list
	'''
	"powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
	s = list(iterable)
	return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def get_pref(N_1, N_2, K=71, distance_file = "distance.csv", list_1=[], list_2=[], list_D = []):
	'''
		:N_1, N_2, K: Number of units and number of nodes
		:distance_file: File that stores the distances pairs of units and nodes
		:list_1, list_2, list_D: Indicate separate units and joint units. If empty, then follow the rule of 1,2,D based on N_1 and N_2
		Output: Preference Lists 1 and 2
		Get the preference list given the setting and distance csv file
	'''
	distance = pd.read_csv(distance_file)
	distance = distance.drop(distance.columns[0], axis=1)
	Pre_1 = []
	Pre_2 = []
	# If don't specify which for which, then just assume first several is. 
	if len(list_1) == len(list_2) == len(list_D) == 0:
		for i in range(K):
			order = np.array(distance.iloc[i,:]).argsort()+1
			Pre_1.append(list(order[(order < N_1+1) | (order > N_1+N_2)]))
			Pre_2.append(list(order[(order > N_1)]))
	# If sepecifies, then follows the specified ones. 
	else:
		assert len(list_1) == N_1
		assert len(list_2) == N_2
		units_1 = list_1 + list_D
		units_2 = list_2 + list_D
		for i in range(K):
			order = np.array(distance.iloc[i,:]).argsort()+1
			Pre_1.append(list(order[[o in units_1 for o in order]]))
			Pre_2.append(list(order[[o in units_2 for o in order]]))
	Pre_1 = np.array(Pre_1).argsort().argsort()
	Pre_2 = np.array(Pre_2).argsort().argsort()
	return np.array(Pre_1), np.array(Pre_2)

def plot_rho(N, rho_e, rho_f):
	color_list = ['3282b8', 'c06c84',  'f8b195', '52de97', 'ffba5a','3282b8', 'c06c84',  'f8b195', '52de97', 'ffba5a']
	fig = plt.figure(figsize=(14,4))
	ax = plt.gca()
	for j in range(len(rho_e)):
		for i in range(N):
			subseq = [rho_e[j][i],1-rho_f[j][i]]
			plt.plot(subseq, np.zeros(len(subseq)) - i, color='#'+color_list[j], marker='o',linestyle='', alpha=0.2)
	ax.set_yticklabels('')
	ax.set_yticks(-np.arange(0, N), minor=True)
	ax.set_yticklabels([i for i in range(N)], minor=True)
	ax.set_xlim([0,1])
	ax.set_ylim([-N, 1])
	plt.tight_layout()
	plt.show()

def close_policy(time_mat):
	# Obtain the policy to dispatch the cloest available unit given the time matrix
	N,J = time_mat.shape
	policy = np.zeros([2**N, J],dtype=np.int)
	pre_list = time_mat.argsort(axis=0).T
	for s in range(2**N):
		for j in range(J):
			pre = pre_list[j]
			for n in range(N):
				if not s >> pre[n] & 1: # n th choice is free
					policy[s, j] = pre[n]
					break
	return policy

def Cal_trans(N, K, Lambda, Mu, frac, pol):
	# Calculate the 
	A = np.zeros([2**N, 2**N])
	# Calculate upward transtition
	for s in range(2**N-1): # The last state will not transition to other states by a arrival
		pol_s = pol[s]
		for j in range(K):
			dis = pol_s[j]
			A[s, s+2**dis] += Lambda * frac[j]
	# Calculate downward transtition
	for s in range(1,2**N): # The first state will not transition
		bin_s = bin(s)
		len_bin = len(bin_s)
		i = 0
		while bin_s[len_bin-1-i] != 'b':
			if bin_s[len_bin-1-i] == '1':
				A[s,s-2**i] = Mu
			i += 1
	return A

def Cal_Expected_Reward(N, A, Lambda, pol, frac, time_mat, r_type = 'T', T = 0):
	# First calculate the expected reward for each state
	R_list = np.zeros(2**N)
	for s in range(2**N-1): # The last state has value 0
		pol_s = pol[s]
		#print(pol_s)
		dis_times = [time_mat[pol_s[j],j] for j in range(len(pol_s))]
		if r_type == 'T':
			R_list[s] = -Lambda/A[s].sum()*np.dot(frac,dis_times)
		elif r_type == 'F':
			R_list[s] = -Lambda/A[s].sum()*np.dot(frac,np.array(dis_times)>T)
		else:
			print('Wrong reward type!')
		# Since the reward of a service completion is 0, we don't capture that for a state. 
		# Therefore, this expected reward is the probability making a transition up and the corresponding reward for that transition, which is the expected response time
	return R_list

def Cal_State_Value(N, A, R_list):
	# Then calculate the value of each state V(s)
	V_list = np.zeros(2**N)
	B = np.zeros([2**N,2**N])
	for s in range(0, 2**N):
		B[s] = A[s]/A[s].sum() # transition probabilities to all other states
	# Setting V[0] = 0
	B_rev = B- np.diag(np.ones(2**N))
	B_rev[:, 0] = -1 # Assuming state V[0] to be 0
	# Solve the linear system to get the value list 
	V_list = np.linalg.solve(B_rev,-R_list)
	# V_list = np.dot(np.linalg.inv(B),-R_list)
	g = V_list[0] # V[0] is the g in this V_list. g is half of the average reward in out case
	V_list[0] = 0 # substitute back let V(0)=0 be the value for state 0
	return V_list, g, B

def Get_optimal_policy(N, K, Lambda, Mu, time_mat,frac, r_type='T', T=0):
	# Obtain the optimal policy
	pol = close_policy(time_mat) # 
	pol_next = pol.copy()
	g_ = 0
	# Get idle list for each state, which is the action space 
	statusmat = [("{0:0"+str(N)+"b}").format(i) for i in range(2**N)]
	idle_list = [np.array([N-1-j for j in range(N) if i[j]=='0']) for i in statusmat]
	# Improving policy 
	while True:
		A = Cal_trans(N, K, Lambda, Mu, frac, pol) # Get transition matrix between states
		R_list = Cal_Expected_Reward(N, A, Lambda, pol, frac, time_mat,r_type, T) # Get the expected reward for each state
		V_list, g, B = Cal_State_Value(N, A, R_list) # Solve for the steady state value for each state
		# Start improve the policy
		for s in range(2**N-1):
			idle = idle_list[s]
			if len(idle) > 1: # If no choice is needed than no need to improve
				for j in range(K):
					# This is the value to pick
					if r_type == 'T':
						value_list = -time_mat[:,j][idle] + V_list[s+2**idle] # response time + value from list
					else:
						value_list = -1 * (time_mat[:,j][idle]>T) + V_list[s+2**idle] # percentage over threshold
					# print(s,j,value_list)
					u_id = value_list.argmax() # chooce the action gives the highest reward
					dis = idle[u_id]
					pol_next[s][j] = dis # Update the policy
		#print(R_list)
		# print(g)
		if (pol_next == pol).all(): # If converge 
			break
		elif np.abs(g_-g)<1e-8:
			# print(pol_next, pol)
			break
		else: # continue
			pol = pol_next.copy()
			g_ = g
	return pol, A # return the policy and the optimal transition matrix, and the reward

def Cal_Response_Time(N, A, Lambda, pol, frac, time_mat):
	# This is the average response time for each state s
	RT_list = np.zeros(2**N)
	for s in range(2**N-1): # The last state has value 0
		pol_s = pol[s]
		dis_times = [time_mat[pol_s[j],j] for j in range(len(pol_s))]
		RT_list[s] = -np.dot(frac,dis_times)
	return RT_list


def Cal_Time_Over(N, A, Lambda, pol, frac, time_mat, T):
	# First calculate the expected reward for each state
	TimeOver_list = np.zeros(2**N)
	for s in range(2**N-1): # The last state has value 0
		pol_s = pol[s]
		#print(pol_s)
		dis_timesover = [time_mat[pol_s[j],j]>T for j in range(len(pol_s))]
		# print(frac,dis_times)
		TimeOver_list[s] = np.dot(frac,dis_timesover)
	return TimeOver_list

def Solve_Hypercube(N, A, Lambda, pol, frac, time_mat, T = 0):
	#Solve for the  
	transition = A.T - np.diag(A.T.sum(axis=0))
	#print (transition)
	#print (np.linalg.det(transition))
	#print ('Eigenvalues of Transition Matrix:',np.linalg.eig(transition)[0])
	transition[-1] = np.ones(2**N)
	b = np.zeros(2**N)
	b[-1] = 1
	prob_dist = np.linalg.solve(transition,b)
	#print(prob_dist)
	TO = 1 # fraction of calls over a threshold 
	RT_list = Cal_Response_Time(N, A, Lambda, pol, frac, time_mat)
	if T > 0:
		Timeover_list = Cal_Time_Over(N, A, Lambda, pol, frac, time_mat, T)
		TO = np.dot(prob_dist[:-1]/prob_dist[:-1].sum(),Timeover_list[:-1])
	#print(RT_list)
	MRT = np.dot(prob_dist[:-1]/prob_dist[:-1].sum(),-RT_list[:-1]) # Remove those served outside of the system
	#print(MRT)
	return prob_dist, MRT, TO
#############################################################################################
####################### 3-STATE HYPERCUBE #####################################################

def three_state_hypercube(Data):
	'''
		:Data: Input data setting
		Output: 
		Solve 3-state hypercube directly. Only work for all joint now
	'''
	# Parameters
	N, N_1, N_2, K = Data['N'], Data['N_1'], Data['N_2'], Data['K']
	Lambda_1, Lambda_2, Mu_1, Mu_2 = Data['Lambda_1'], Data['Lambda_2'], Data['Mu_1'], Data['Mu_2']
	e,f,Pre_L_1,Pre_L_2 = Data['e'], Data['f'], Data['Pre_L_1'], Data['Pre_L_2']
	start_time = time.time() # staring time
	# Number of units for each subsystem
	N_med = N - N_2
	N_fire = N - N_1
	# Initialize States for each subsystem
	Num_State = 2**(N_1+N_2) * 3**(N-N_1-N_2)
	statusmat_sep = [np.base_repr(i,base=2)[1:] for i in range(2**(N_1+N_2),2**(N_1+N_2+1))] # Pad a 1 in front to make the length and take out the 1
	statusmat_joint = [np.base_repr(i,base=3)[1:] for i in range(3**(N-N_1-N_2),2*3**(N-N_1-N_2))] # Pad a 1 in front to make the length and take out the 1
	statusmat = [y+x for y in statusmat_joint for x in statusmat_sep] # Get all the transitions
	transition = np.zeros([Num_State,Num_State]) # Initialize the transition matrix
	# Update upward transition rate
	for j in range(K): # Loop through every atom
		for n in range(Num_State): # Loop through states
			B_n = statusmat[n]
			# For EMS
			for i in Pre_L_1[j]: # find the one in the preference to add the rate to the state
				if B_n[N-1-i] == '0': # find the first available unit N-1-i shows unit i in the binary representation
					if i < N_1+N_2: # if it is a separate unit
						m = n + 2**i # the state transit to 
					else: # if it is a joint unit
						m = n + 2**(N_1+N_2) * 3**(i-N_1-N_2) # the state transit to 
					transition[m,n] += Lambda_1*e[j]
					break
			# For fire 
			for i in Pre_L_2[j]: # find the one in the preference to add the rate to the state
				if B_n[N-1-i] == '0': # find the first available unit
					if i < N_1+N_2: # if it is a separate unit
						m = n + 2**i # the state transit to 
					else: # if it is a joint unit
						m = n + 2**(N_1+N_2) * 2*3**(i-N_1-N_2) # the state transit to 
					transition[m,n] += Lambda_2*f[j]
					break
	# Upward downward transition rate
	for n in range(Num_State): # Loop through states 
		B_n = statusmat[n]
		for i in range(N): # Loop through every element in state
			if B_n[N-1-i] == '1':
				if i < N_1+N_2:
					m = n - 2**i
				else:
					m = n - 2**(N_1+N_2) * 3**(i-N_1-N_2)
				transition[m,n] = Mu_1
			elif B_n[N-1-i] == '2':
				if i < N_1+N_2:
					m = n - 2**i
				else:
					m = n - 2**(N_1+N_2) * 2*3**(i-N_1-N_2)
				transition[m,n] = Mu_2
	# print(1)
	# Set diagonal
	diag = np.diag(transition.sum(axis=0))
	transition -= diag
	# Solve for the steady state equation
	transition[-1] = np.ones(Num_State)
	b = np.zeros(Num_State)
	b[-1] = 1
	#prob_dist = np.linalg.solve(transition,b) # linear solve method
	#prob_dist = jacobi_method(transition,b) # Jacobi method
	transition_sparse = sparse.csc_matrix(transition)
	prob_dist = spsolve(transition_sparse,b) # sparse solve method
	#print(prob_dist) 
	print("------ %s seconds ------" % (time.time() - start_time))
	total_time = time.time() - start_time
	# Get rho
	rho_e, rho_f = np.zeros(N-N_2), np.zeros(N-N_1)
	Sep_med = list(range(N_1))
	Sep_fire = list(range(N_1,N_1+N_2))
	Med = [x for x in range(N) if x not in Sep_fire]
	Fire = [x for x in range(N) if x not in Sep_med]
	for n in range(Num_State):
		B_n = statusmat[n]
		for i in range(N-N_2):
			if B_n[N-1-Med[i]] == '1':
				rho_e[i] += prob_dist[n]
		for i in range(N-N_1):
			if B_n[N-1-Fire[i]] == '2':
				rho_f[i] += prob_dist[n]
	# print(rho_e, rho_f)
	return rho_e, rho_f, total_time

#############################################################################################
####################### ALPHA HYPERCUBE #####################################################
def alpha_hypercube(Data):
	'''
		:Data: Input data setting
		Output: rho_med, rho_fire, alpha_e, alpha_f, prob_dist_e, prob_dist_f, J, iteration
		alpha hypercube algorithm. Obtain utiliations, alphas, probability distributions, Jacbian matrix, and number of Iterations 
	'''
	# Parameters
	N, N_1, N_2, K = Data['N'], Data['N_1'], Data['N_2'], Data['K']
	Lambda_1, Lambda_2, Mu_1, Mu_2 = Data['Lambda_1'], Data['Lambda_2'], Data['Mu_1'], Data['Mu_2']
	e,f,Pre_L_1,Pre_L_2 = Data['e'], Data['f'], Data['Pre_L_1'], Data['Pre_L_2']
	start_time = time.time() # staring time
	# Number of units for each subsystem
	N_med = N - N_2
	N_fire = N - N_1
	# Initialize States for each subsystem
	State_med, statusmat_med, busy_e=State_Initialization(N_med)
	State_fire, statusmat_fire, busy_f=State_Initialization(N_fire)
	# Initialize alpha_f
	alpha_f_old = np.ones(N_med)
	alpha_f = np.zeros(N_med)
	# Start Iteration
	iteration = 0
	J_e_list = [] # Convergence factor J
	J_f_list = [] # Convergence factor J
	rho_med_list, rho_fire_list = [], []
	alpha_med_list, alpha_fire_list = [], []
	while (max(abs(alpha_f_old-alpha_f)) > 0.00001):
		iteration += 1 # Ite +1
		##############################
		# Start Med subsystem
		# Calculate steady state distribution. 
		prob_dist_e,transition_e = Get_SteadyState_Dist(N_med, Mu_1, statusmat_med, State_med, alpha_f, busy_e, Pre_L_1, e, Lambda_1)
		rho_med = [sum([prob_dist_e[j] for j in range(State_med) if i in busy_e[j]]) for i in range(N_med)]
		J_e = convergence(transition_e, rho_med, N_med,alpha_f,statusmat_med,busy_e,Lambda_1,Pre_L_1, e)  # check convergence
		print('J_e', J_e)
		# print('max rho sum of J_e', max(J_e.sum(axis=1)))
		# print('eigen value of J_e',np.linalg.eig(J_e)[0])
		# Get alpha_e
		alpha_e = np.zeros(N_fire)
		for k in range(N_2,N_fire):
			alpha_e[k] = rho_med[k+N_1-N_2]/(rho_med[k+N_1-N_2]+(1-rho_med[k+N_1-N_2])*(1-alpha_f[k+N_1-N_2]))
		##############################
		# Start Fire subsystem
		# Calculate steady state distribution. 
		prob_dist_f,transition_f = Get_SteadyState_Dist(N_fire, Mu_2, statusmat_fire, State_fire, alpha_e, busy_f, Pre_L_2, f, Lambda_2)
		rho_fire = [sum([prob_dist_f[j] for j in range(State_fire) if i in busy_f[j]]) for i in range(N_fire)]
		J_f = convergence(transition_f, rho_fire, N_fire,alpha_e,statusmat_fire,busy_f,Lambda_2,Pre_L_2, f)  # Check convergence
		print('J_f', J_f)
		#print('max rho sum of J_f',max(J_f.sum(axis=1)))
		#print('eigen value of J_f',np.linalg.eig(J_f)[0])
		
		alpha_f_old = alpha_f # Update alpha for last step
		# Get alpha_f
		alpha_f = np.zeros(N_med)
		for k in range(N_1,N_med):
			alpha_f[k] = rho_fire[k+N_2-N_1]/(rho_fire[k+N_2-N_1]+(1-rho_fire[k+N_2-N_1])*(1-alpha_e[k+N_2-N_1]))
		# J = max(J, max(np.array(J_e))*max(np.array(J_f)))
		try:
			J_e_list.append(max(J_e.sum(axis=1)))
			J_f_list.append(max(J_f.sum(axis=1)))
		except:
			pass
		rho_med_list += [rho_med]
		rho_fire_list += [rho_fire]
		alpha_med_list += [alpha_e]
		alpha_fire_list += [alpha_f]
		print('rho_med:',rho_med)
		print('rho_fire:',rho_fire)
		print('alpha_e:',alpha_e)
		print('alpha_f:',alpha_f)
		print('Total Rho_e:',sum(rho_med))
		print('Total Rho_f:',sum(rho_fire))
		print('-------------------------')
	print("------ %s seconds ------" % (time.time() - start_time))
	print("------ %s iteration ------" % (iteration))
	print('rho_med:',['%.4f' % x for x in rho_med])
	print('rho_fire:',['%.4f' % x for x in rho_fire])
	print('alpha_e:',alpha_e)
	print('alpha_f:',alpha_f)
	J = [J_e_list]+[J_f_list]
	print(prob_dist_e)
	#plot_rho(N_med, rho_med_list, rho_fire_list)
	#plot_rho(N_med, alpha_med_list, alpha_fire_list)
	return rho_med, rho_fire, alpha_e, alpha_f, prob_dist_e, prob_dist_f, J, iteration

def alpha_hypercube_new(Data):
	'''
		:Data: Input data setting
		Output: rho_med, rho_fire, alpha_e, alpha_f, prob_dist_e, prob_dist_f, J, iteration
		New alpha hypercube algorithm. Do not update alpha at each step
	'''
	# Parameters
	N, N_1, N_2, K = Data['N'], Data['N_1'], Data['N_2'], Data['K']
	Lambda_1, Lambda_2, Mu_1, Mu_2 = Data['Lambda_1'], Data['Lambda_2'], Data['Mu_1'], Data['Mu_2']
	e,f,Pre_L_1,Pre_L_2 = Data['e'], Data['f'], Data['Pre_L_1'], Data['Pre_L_2']
	start_time = time.time() # staring time
	# Number of units for each subsystem
	N_med = N - N_2
	N_fire = N - N_1
	# Initialize States for each subsystem
	State_med, statusmat_med, busy_e=State_Initialization(N_med)
	State_fire, statusmat_fire, busy_f=State_Initialization(N_fire)
	# Initialize alpha_f, alpha_e
	alpha_f_old = np.ones(N_med)
	alpha_f = np.zeros(N_med)
	alpha_e_old = np.ones(N_fire)
	alpha_e = np.zeros(N_fire)
	# Start Iteration
	iteration = 0
	J = 0
	rho_med_list, rho_fire_list = [], []
	alpha_med_list, alpha_fire_list = [], []
	while (max(abs(alpha_f_old-alpha_f)) > 0.00001 or max(abs(alpha_e_old-alpha_e)) > 0.00001):
		iteration += 1 # Ite +1
		##############################
		# Start Med subsystem
		# Calculate steady state distribution. 
		prob_dist_e,transition_e = Get_SteadyState_Dist(N_med, Mu_1, statusmat_med, State_med, alpha_f, busy_e, Pre_L_1, e, Lambda_1)
		rho_med = [sum([prob_dist_e[j] for j in range(State_med) if i in busy_e[j]]) for i in range(N_med)]
		#J_e = convergence(transition_e, rho_med, N_med,alpha_f,statusmat_med,busy_e,Lambda_1,Pre_L_1, e)  # check convergence
		##############################
		# Start Fire subsystem
		# Calculate steady state distribution. 
		prob_dist_f,transition_f = Get_SteadyState_Dist(N_fire, Mu_2, statusmat_fire, State_fire, alpha_e, busy_f, Pre_L_2, f, Lambda_2)
		rho_fire = [sum([prob_dist_f[j] for j in range(State_fire) if i in busy_f[j]]) for i in range(N_fire)]
		#J_f = convergence(transition_f, rho_fire, N_fire,alpha_e,statusmat_fire,busy_f,Lambda_2,Pre_L_2, f)  # Check convergence
		##############################
		# Update alpha
		alpha_e_old = alpha_e
		alpha_f_old = alpha_f # Update alpha for last step
		# Get alpha_e
		alpha_e = np.zeros(N_fire)
		for k in range(N_2,N_fire):
			alpha_e[k] = rho_med[k+N_1-N_2]/(rho_med[k+N_1-N_2]+(1-rho_med[k+N_1-N_2])*(1-alpha_f_old[k+N_1-N_2]))
		# Get alpha_f
		alpha_f = np.zeros(N_med)
		for k in range(N_1,N_med):
			alpha_f[k] = rho_fire[k+N_2-N_1]/(rho_fire[k+N_2-N_1]+(1-rho_fire[k+N_2-N_1])*(1-alpha_e_old[k+N_2-N_1]))
		#J = max(J, max(np.array(J_e))*max(np.array(J_f)))
		rho_med_list += [rho_med]
		rho_fire_list += [rho_fire]
		alpha_med_list += [alpha_e]
		alpha_fire_list += [alpha_f]
		print('rho_med:',rho_med)
		print('rho_fire:',rho_fire)
		print('alpha_e:',alpha_e)
		print('alpha_f:',alpha_f)
		print('Total Rho_e:',sum(rho_med))
		print('Total Rho_f:',sum(rho_fire))
		print('-------------------------')
	print("------ %s seconds ------" % (time.time() - start_time))
	print("------ %s iteration ------" % (iteration))
	print('rho_med:',['%.4f' % x for x in rho_med])
	print('rho_fire:',['%.4f' % x for x in rho_fire])
	print('alpha_e:',alpha_e)
	print('alpha_f:',alpha_f)
	#plot_rho(N_med, rho_med_list, rho_fire_list)
	#plot_rho(N_med, alpha_med_list, alpha_fire_list)
	return rho_med, rho_fire, alpha_e, alpha_f, prob_dist_e, prob_dist_f, J, iteration

# def alpha_hypercube_multiple(Data): # see matlab code
# 	### Not finished yet. Need to revise Get_SteadyState_Dist to include the transition of 2. the function:transup needs to be revised.
# 	N, N_1, N_2, K = Data['N'], Data['N_1'], Data['N_2'], Data['K']
# 	Lambda_1, Lambda_2,Lambda_1_2,Lambda_2_2, Mu_1, Mu_2 = Data['Lambda_1'], Data['Lambda_2'], Data['Lambda_1_2'], Data['Lambda_2_2'], Data['Mu_1'], Data['Mu_2']
# 	e,f,Pre_L_1,Pre_L_2 = Data['e'], Data['f'], Data['Pre_L_1'], Data['Pre_L_2']
# 	## Initilization
# 	start_time = time.time()
# 	# Get number of units for each subsystem
# 	N_med = N - N_2
# 	N_fire = N - N_1
# 	State_med, statusmat_med, busy_e=State_Initialization(N_med)
# 	State_fire, statusmat_fire, busy_f=State_Initialization(N_fire)
# 	# Initialize alpha_e
# 	alpha_f_old = np.ones(N_med)
# 	alpha_f = np.zeros(N_med)
# 	# Start Iteration
# 	iteration = 0
# 	while (max(abs(alpha_f_old-alpha_f)) > 0.00001):
# 		iteration += 1
# 		# Next is to calculate the steady state distribution. 
# 		prob_dist_e,transition_e = Get_SteadyState_Dist(N_med, Mu_1, statusmat_med, State_med, alpha_f, busy_e, Pre_L_1, e, Lambda_1)
# 		# Get rho
# 		rho_med = [sum([prob_dist_e[j] for j in range(State_med) if i in busy_e[j]]) for i in range(N_med)]
# 		# Get alpha
# 		alpha_e = np.zeros(N_fire)
# 		for k in range(N_2,N_fire):
# 			alpha_e[k] = rho_med[k+N_1-N_2]/(rho_med[k+N_1-N_2]+(1-rho_med[k+N_1-N_2])*(1-alpha_f[k+N_1-N_2]))
# 		# End of medical subsystem
# 		##############################
# 		# Start Fire subsystem
# 		# Next is to calculate the steady state distribution. 
# 		prob_dist_f,transition_f = Get_SteadyState_Dist(N_fire, Mu_2, statusmat_fire, State_fire, alpha_e, busy_f, Pre_L_2, f, Lambda_2)
# 		# Get rho
# 		rho_fire = [sum([prob_dist_f[j] for j in range(State_fire) if i in busy_f[j]]) for i in range(N_fire)]
# 		# Get alpha
# 		alpha_f_old = alpha_f
# 		alpha_f = np.zeros(N_med)
# 		for k in range(N_1,N_med):
# 			alpha_f[k] = rho_fire[k+N_2-N_1]/(rho_fire[k+N_2-N_1]+(1-rho_fire[k+N_2-N_1])*(1-alpha_e[k+N_2-N_1]))
# 		print('rho_med:',rho_med)
# 		print('rho_fire:',rho_fire)
# 		print('alpha_e:',alpha_e)
# 		print('alpha_f:',alpha_f)
# 		print('Total Rho_e:',sum(rho_med))
# 		print('Total Rho_f:',sum(rho_fire))
# 		print('-------------------------')
# 	print("------ %s seconds ------" % (time.time() - start_time))
# 	print("------ %s iteration ------" % (iteration))
# 	print('rho_med:',['%.4f' % x for x in rho_med])
# 	print('rho_fire:',['%.4f' % x for x in rho_fire])
# 	print('alpha_e:',alpha_e)
# 	print('alpha_f:',alpha_f)
# 	return rho_med, rho_fire, alpha_e, alpha_f, prob_dist_e, prob_dist_f, iteration

#@pysnooper.snoop()
def State_Initialization(N_sub):
	'''
		:Data: Input data setting
		Output: State_sub,statusmat_sub,busy_sub
		Initialize states
	'''
	# Number of states for each subsystem
	State_sub = 2 ** N_sub
	# binary status matrix for medical and fire
	statusmat_sub = [("{0:0"+str(N_sub)+"b}").format(i) for i in range(State_sub)]
	# busy units ID for each state
	busy_sub = [[N_sub-1-j for j in range(N_sub) if i[j]=='1'] for i in statusmat_sub]
	return State_sub,statusmat_sub,busy_sub

def Get_SteadyState_Dist(N_sub, Mu, statusmat_sub, State_sub, alpha, busy_list, Pre, prob, Lambda_v):
	'''
		:N_sub, Mu, statusmat_sub, State_sub, alpha, busy_list, Pre, prob, Lambda_v: Inputs
		Output: prob_dist, transition
		Get the statedy state distribution and Matrix A
	'''
	# np.set_printoptions(precision=3,suppress=True)
	transup = Calculate_transup(N_sub,alpha,statusmat_sub,busy_list,Lambda_v,Pre, prob)
	transdown = Calculate_transdown(N_sub,Mu,busy_list)
	# Inflow
	inflow = (transup+transdown).T
	# Outflow
	outflow = inflow.sum(axis=0)
	outflow_diag = np.diag(outflow)
	#Solve it 
	transition = inflow - outflow_diag
	#print (transition)
	#print (np.linalg.det(transition))
	#print ('Eigenvalues of Transition Matrix:',np.linalg.eig(transition)[0])
	transition[-1] = np.ones(State_sub)
	b = np.zeros(State_sub)
	b[-1] = 1
	# Use sparse solve
	start_time = time.time()
	transition_sparse = sparse.csc_matrix(transition)
	prob_dist = spsolve(transition_sparse,b)
	#prob_dist = jacobi_method(transition,b)
	#prob_dist = np.linalg.solve(transition,b)
	print("------ %s seconds ------" % (time.time() - start_time))
	return prob_dist, transition

def convergence(A, rho, N_sub,alpha,statusmat_sub,busy_list,Lambda_v,Pre, prob):
	'''
		:A, rho, N_sub,alpha,statusmat_sub,busy_list,Lambda_v,Pre, prob: Inputs. prob is e or f. 
		Output: J  the jacobian matrix
		A function that helps check the convergence condition
	'''
	# column vector b
	b = np.zeros(2**N_sub).T
	b[-1] = 1
	# A /= Lambda_v # it gets the  same steady state distribution if A is divided by Lambda
	# A[-1,:] = 1
	A_inv = np.linalg.inv(A)
	p_j = A_inv[:,-1]
	############ Some print functions to check ##############
	#print ('A', A)
	#print('A^-1',A_inv)
	#print('1 norm A_inv\n', np.sum(np.abs(A_inv),axis=0))
	#print('tr(A^-1)\n', np.matrix.trace(A_inv))
	# print('alpha:',alpha)
	# print('rho:',rho)
	# print('column sum of A^-1\n',np.linalg.inv(A).sum(axis=0))
	# print('p_j', p_j)
	# print('eig(A^-1)\n',np.linalg.eig(np.linalg.inv(A))[0])
	#########################################################
	J = np.zeros([N_sub,N_sub]) # The Jacobian matrix J for subsystem y. This is J_1 or J_2.
	J_list = []
	for i in range(N_sub):
		# print('unit:', i)
		# Obtain e_i
		e_i = np.zeros(2**N_sub)
		b_i = [n for n in range(2**N_sub) if i in busy_list[n]]
		e_i[b_i] = 1
		M_i = np.zeros([2**N_sub,2**N_sub])
		M_i[-1,:] = e_i-1
		# Obtain A_i
		Ai = np.append(A[:-1], [e_i], axis=0) # This is to replace the last row of A by e_i
		Ai_inv = np.linalg.inv(Ai)
		############ Some print functions to check ##############
		# print('e_i\n', e_i)
		# print('M_i\n', M_i)
		# print('(e_i-1)^T A^-1\n', (e_i-1).dot(A_inv))
		#print('e_i^T A^-1\n', e_i.dot(A_inv))
		# print('eig(A)\n',np.linalg.eig(A)[0])
		# print('Ai\n',Ai)
		# print('A Ai^-1\n',np.matmul(A,np.linalg.inv(Ai)))
		# print('Ai^-1\n',Ai_inv)
		# print('eig(Ai^-1)\n',np.linalg.eig(Ai_inv)[0])
		# print('column sum of Ai^-1\n',np.linalg.inv(Ai).sum(axis=0))
		# v = np.linalg.inv(Ai).sum(axis=0)[:,np.newaxis] # column sum of Ai^-1
		# print('Ai^-1 - A^-1\n',Ai_inv-A_inv)
		# print('A^-1 M_i', A_inv.dot(M_i))
		#print('eig(Ai^-1 - A^-1)\n',np.linalg.eig(Ai_inv-A_inv)[0])
		#print('(A - Ai)\n', A-Ai)
		#print('Ai^-1 (A - Ai)\n', Ai_inv.dot(A-Ai))
		#print('eig(Ai^-1 (A - Ai)\n',np.linalg.eig(Ai_inv.dot(A-Ai))[0])
		# print('M_i A^-1', M_i.dot(A_inv))
		# print('A^-1 M_i A^-1\n', A_inv.dot(M_i).dot(A_inv))
		c_i = (M_i.dot(A_inv))[-1,:]
		#print('c_i', c_i)
		# print('row sum of Ai^-1-A^-1\n',(np.linalg.inv(Ai)-np.linalg.inv(A)).sum(axis=1))
		# print('(Ai-rhoiA)\n',Ai-rho[i]*A)
		# print('(Ai^-1-A^-1)A ^T\n',np.matmul(np.linalg.inv(Ai)-np.linalg.inv(A),A).T)
		# print('(Ai^-1 -A^-1) A\n',(Ai_inv-A_inv).dot(A))
		#########################################################
		J_norm = 0 # This is the value for |J| which checks for the converegence of the algorithm
		sum_dA = np.zeros([2**N_sub,2**N_sub])
		for j in range(N_sub): # Here j is the m in the paper
			#print('differentiate over unit:', j)
			dA_j = A_prime(N_sub,alpha,statusmat_sub,busy_list,Lambda_v,Pre, prob,j)
			# sum_dA += dA_j # The sum of matrices dA_j over all units j
			tr_j=np.matrix.trace(np.matmul(Ai_inv-A_inv,dA_j)) # Calculate the trace for the term
			drho_ij = rho[i]*tr_j  # This is drho_ij as calculated in the paper
			# print('trace_j', tr_j)
			# print('drho_ij', drho_ij)
			############ Some print functions to check ##############
			#print('dA_j\n',dA_j)
			# print('eig(dA_j)\n',np.linalg.eig(dA_j)[0])
			# print('trace of dA_j\n',np.matrix.trace(dA_j))
			#print('Pk dA_j\n',np.linalg.inv(A)[:,-1]*dA_j)
			#print('row sum of Pk dA\n',(np.linalg.inv(A)[:,-1]*dA_j).sum(axis=1))
			#w = (np.linalg.inv(A)[:,-1]*dA).sum(axis=1)[:,np.newaxis] #row sum of Pk dA_j
			#print('v wT',np.matmul(v.T,w))
			# print('A^-1 dA_j\n',np.matmul(A_inv,dA_j))
			# print('e_i^T A^-1 dA_j\n',e_i.dot(np.matmul(A_inv,dA_j)))
			# print('(e_i-1)^T A^-1 dA_j\n',(e_i-1).dot(np.matmul(A_inv,dA_j)))
			# print('(Ai^-1 -A^-1) dA_j\n',(Ai_inv-A_inv).dot(dA_j))
			# sprint('(e_i-1)^T A^-1 dA_j\n',np.matmul(A_inv,dA_j))
			# print('1/rho_i A^-1 b\n', np.mat(A_inv.dot(b)).T.dot(np.mat(e_i))/rho[i])
			#print('(A - Ai) A^-1 dA_j\n',(A-Ai).dot(np.matmul(np.linalg.inv(A),dA_j)))
			#print('eig [(A - Ai) A^-1 dA_j]\n',np.linalg.eig((A-Ai).dot(np.matmul(np.linalg.inv(A),dA_j)))[0])
			#print('(Ai^-1 -A^-1) dA_j\n',np.matmul(np.linalg.inv(Ai)-np.linalg.inv(A),dA_j))
			#print('eig((Ai^-1 -A^-1) dA_j)\n',np.linalg.eig(np.matmul(Ai_inv-A_inv,dA_j))[0])
			#print('rho_i*tr_j:',rho[i]*tr_j) # 2020.1.6 Tested that this is equal to drho_ij
			#print('M_i A^-1 dA_j\n', M_i.dot(A_inv).dot(dA_j))
			# print('sum of c_i dA_j',[sum([c_i[i]*dA_j[i,j] for i in range(2**N_sub)]) for j in range(2**N_sub)])
			#print('drho_ij another method:', -sum([p_j[j]*sum([c_i[i]*dA_j[i,j] for i in range(2**N_sub)]) for j in range(2**N_sub)]))
			# print('drho_ij:',-e_i.dot(A_inv).dot(dA_j).dot(A_inv).dot(b))
			# print('(Ai^-1 - A^-1) A\n', (Ai_inv - A_inv).dot(A))
			#print('tr((Ai^-1 - A^-1) dA_j)\n', np.matrix.trace((Ai_inv - A_inv).dot(dA_j)))
			# print('eiven value of (Ai^-1 - A^-1) dA_j\n', np.linalg.eig((Ai_inv - A_inv).dot(dA_j))[0])
			# print('tr(A^-1 M_i A^-1 dA_j)', np.matrix.trace(A_inv.dot(M_i).dot(A_inv).dot(dA_j)))
			#print('A^-1 dA_j A^-1\n',A_inv.dot(dA_j).dot(A_inv))
			#print('column sum of A^-1 dA_j A^-1\n',A_inv.dot(dA_j).dot(A_inv).sum(axis=0))
			#########################################################
			if j == i:
				# print('j==i')
				J_ij = (drho_ij*(1-alpha[i])+rho[i]*(1-rho[i]))/(rho[i]+(1-rho[i])*(1-alpha[i]))**2
				#print('J_ij diagonal',J_ij)
			else:
				J_ij = drho_ij*(1-alpha[i])/(rho[i]+(1-rho[i])*(1-alpha[i]))**2
			J[i,j] = J_ij
			#J_i = np.abs(numerator)
			#print('J_ij',J_ij)
			#J_norm += J_i
		#print('sum_dA\n',sum_dA)
		#print('A^-1 sum_dA A^-1\n',e_i.dot(A_inv).dot(sum_dA).dot(A_inv).dot(b))
		#print('Sum J_norm', J_norm
		#J_list.append(J_norm)
	return J

def A_prime(N,alpha,statusmat,busy_list,Lambda_v,Pre, prob,j):
	'''
		:N,alpha,statusmat,busy_list,Lambda_v,Pre, prob,j: Inputs
		Output: dA
		Obtain the derivative of Matrix A
	'''
	K = len(Pre)
	all_set = list(range(N))
	dA = np.zeros([2**N,2**N])
	for m in range(2**N):
		if j not in busy_list[m]:
			free_list = list(set(all_set)-set(busy_list[m])-set([j]))
			dA[m,m] = Lambda_v*np.prod(alpha[free_list])
			for n in range(m+1,2**N-1):
				d = np.log2(m^n)
				if int(d) == d:
					free_list = list(set(all_set)-set(busy_list[n]))
					if d == j:
						dA[n,m] = -Lambda_v*sum([prob[k]*np.prod(alpha[list(set(free_list).intersection(set(Pre[k][:np.where(Pre[k] == d)[0][0]])))]) for k in range(K)])
					else:
						dA[n,m] = Lambda_v*(1-alpha[int(d)])*sum([prob[k]*(len(alpha[list(set(free_list).intersection(set(Pre[k][:np.where(Pre[k] == d)[0][0]])))])>0)*(np.where(Pre[k] == j)[0][0]<np.where(Pre[k] == d)[0][0])*np.prod(alpha[list((set(free_list)-set([j])).intersection(set(Pre[k][:np.where(Pre[k] == d)[0][0]])))]) for k in range(K)])
	return dA

def response_time(Pre, steady_dist, alpha, frac_call, units, distance_file="distance.csv", distance = None):
	# This is an inefficient way of obtaining response time. Don't have to improve now. 
	# For multiple dispatch, this is to calculate the response time for the first responding unit. 
	
	## This is wrong!! Need to divide by (1-P_allbusy). It works fine if all busy is near 0, but just wrong. 
	diff_len = len(units) - len(alpha) # If the length is different, pad it with 0
	if diff_len > 0:
		alpha = np.pad(alpha_f, (diff_len,0), 'constant', constant_values=(0))
	elif diff_len < 0:
		alpha = alpha[diff_len:]
	Pre = Pre.argsort().argsort()  # Makes it continuous
	# print(Pre)

	if distance is None:
		distance = pd.read_csv(distance_file)
		distance = distance.drop(distance.columns[0], axis=1)
		distance = distance/1600
		distance[distance > 0.7712735431536174] = (distance[distance > 0.7712735431536174]*111.51113889304331+86.005591195132666)/60
		distance[distance < 0.7712735431536174] = 195.86302790816589*np.sqrt(distance[distance < 0.7712735431536174])/60

	K = len(Pre)
	response_time = np.zeros(K)

	statusmat = [("{0:0"+str(len(units))+"b}").format(i) for i in range(len(steady_dist))]
	for r in range(K):
		response = 0
		res_time = distance.iloc[r,units]
		for i in range(len(steady_dist)):
			carry = 1
			p = steady_dist[i]
			for j in range(len(units)):
				index = Pre[r,j]
				if statusmat[i][len(units)-1-index] == '0':
					response += p*carry*(1-alpha[index])*res_time[index]
					carry = carry*alpha[index]
					if carry == 0:
						break
		response_time[r] = response
	return np.dot(response_time,frac_call), response_time

#############################################################################################
####################### LINEAR ALPHA ########################################################
def alpha_approximation(Data):
	'''
		:Data: Inputs
		Output: rho_i_e, rho_i_f, alpha_e, alpha_f
		Approximation method for my algorithm
	'''
	N, N_1, N_2, K = Data['N'], Data['N_1'], Data['N_2'], Data['K']
	Lambda_1, Lambda_2, Mu_1, Mu_2 = Data['Lambda_1'], Data['Lambda_2'], Data['Mu_1'], Data['Mu_2']
	e,f,Pre_L_1,Pre_L_2 = Data['e'], Data['f'], Data['Pre_L_1'], Data['Pre_L_2']
	# Initialization
	# Lambda for each node
	Lambda_1_j = Lambda_1 * e
	Lambda_2_j = Lambda_2 * f
	# Get number of units for each subsystem
	N_med = N - N_2
	N_fire = N - N_1
	# Number of states for each subsystem
	State_med = 2 ** N_med
	State_fire = 2 ** N_fire
	# Calculate average Rho
	Rho_med = Lambda_1/(Mu_1*N_med) 
	Rho_fire = Lambda_2/(Mu_2*N_fire) 
	# Calculate ErlangLoss steady state probabilities
	P_k_e = ErlangLoss(Lambda_1, Mu_1, N_med)
	P_k_f = ErlangLoss(Lambda_2, Mu_2, N_fire)
	# Initialization r and rho
	r_e = Rho_med*(1-P_k_e[-1]) # r in the paper. average utilization over all servers
	rho_i_e = np.full(N_med,r_e) # utilization of each unit i
	r_f = Rho_fire*(1-P_k_f[-1]) # r in the paper. average utilization over all servers
	rho_i_f = np.full(N_fire,r_f) # utilization of each unit i
	# Calculate Q
	Q_e = Q_Calculation(N = N_med, P_k = P_k_e, Rho = r_e)
	Q_f = Q_Calculation(N = N_fire, P_k = P_k_f, Rho = r_f)
	# Preference List Calculate G
	G_e = [[np.where(Pre_L_1[:,i] == j)[0] for i in range(N_med)] for j in range(N_med)] 
	G_f = [[np.where(Pre_L_2[:,i] == j)[0] for i in range(N_fire)] for j in range(N_fire)]
	# Get matrix
	# State_med,statusmat_med,busy_med=State_Initialization(N_med)
	# transdown_med = Calculate_transdown(N_med,Mu_1,busy_med)
	# State_fire,statusmat_fire,busy_fire=State_Initialization(N_fire)
	# transdown_fire = Calculate_transdown(N_fire,Mu_2,busy_fire)
	# Start calculating time 
	start_time = time.time()
	# Initialize alpha_e
	alpha_f_old = np.ones(N_med)
	alpha_f = np.zeros(N_med)
	alpha_e = np.zeros(N_fire)
	# Larson Approximation for Medical Subsystem
	new = True
	while (max(abs(alpha_f_old-alpha_f)) > 0.0001):
		# Medical Subsystem
		# Start Approximation
		if new:
			#P_k_e,Lambda,_ = P_withbreak_pref_new(Lambda_1, Mu_1, alpha_f,Pre_L_1,e,statusmat_med,busy_med,transdown_med)
			#P_k_e,Lambda,_ = P_withbreak_rand_new(Lambda_1, Mu_1, alpha_f,statusmat_med,busy_med,transdown_med)
			P_k_e,Lambda = P_equal_prob(Lambda_1, Mu_1, alpha_f)
			r_e = np.dot(P_k_e,list(range(N_med+1)))/N_med
			Q_e = Q_Calculation(N = N_med, P_k = P_k_e, Rho = r_e)
		rho_i_e = Larson_Approx(N_med, Mu_1, Q_e, Pre_L_1,G_e,Lambda_1_j, r_e, rho_i_e, alpha_f, alpha_other=np.concatenate((np.zeros(N_1), alpha_e[-(N-N_1-N_2):]), axis=None), epsilon=0.000033)
		# Get alpha_f
		alpha_e = np.zeros(N_fire)
		for k in range(N_2,N_fire):
			alpha_e[k] = rho_i_e[k+N_1-N_2]/(rho_i_e[k+N_1-N_2]+(1-rho_i_e[k+N_1-N_2])*(1-alpha_f[k+N_1-N_2]))
		##############################
		# Start Fire subsystem
		if new:
			#P_k_f,Lambda,_ = P_withbreak_pref_new(Lambda_2, Mu_2, alpha_e,Pre_L_2,f,statusmat_fire,busy_fire,transdown_fire)
			#P_k_f,Lambda,_ = P_withbreak_rand_new(Lambda_2, Mu_2, alpha_e,statusmat_fire,busy_fire,transdown_fire)
			P_k_f,Lambda = P_equal_prob(Lambda_2, Mu_2, alpha_e)
			r_f = np.dot(P_k_f,list(range(N_fire+1)))/N_fire
			Q_f = Q_Calculation(N = N_fire, P_k = P_k_f, Rho = r_f)
		rho_i_f = Larson_Approx(N_fire, Mu_2, Q_f, Pre_L_2, G_f, Lambda_2_j, r_f, rho_i_f, alpha_e, alpha_other=np.concatenate((np.zeros(N_2), alpha_f[-(N-N_1-N_2):]), axis=None), epsilon=0.000033)
		# Get alpha_e
		alpha_f_old = alpha_f  # The old is for checking whether to exit or not. 
		alpha_f = np.zeros(N_med)
		for k in range(N_1,N_med):
			alpha_f[k] = rho_i_f[k+N_2-N_1]/(rho_i_f[k+N_2-N_1]+(1-rho_i_f[k+N_2-N_1])*(1-alpha_e[k+N_2-N_1]))
		# print('rho_med:',rho_i_e)
		# print('rho_fire:',rho_i_f)
		# print('alpha_e:',alpha_e)
		# print('alpha_f:',alpha_f)
		# print('Total Rho_e:',sum(rho_i_e))
		# print('Total Rho_f:',sum(rho_i_f))
		# print('-------------------------')
	print("--- %s seconds ---" % (time.time() - start_time))
	print('rho_med:',['%.4f' % x for x in rho_i_e])
	print('rho_fire:',['%.4f' % x for x in rho_i_f])
	print('alpha_e:',alpha_e)
	print('alpha_f:',alpha_f)
	return rho_i_e, rho_i_f, alpha_e, alpha_f

def alpha_multiple_approximation(Data, units_e = [], units_f = []):
	N, N_1, N_2, K = Data['N'], Data['N_1'], Data['N_2'], Data['K']
	Lambda_1, Lambda_2, Mu_1, Mu_2, Lambda_1_2, Lambda_2_2 = Data['Lambda_1'], Data['Lambda_2'], Data['Mu_1'], Data['Mu_2'], Data['Lambda_1_2'], Data['Lambda_2_2']
	e,f,Pre_L_1,Pre_L_2 = Data['e'], Data['f'], Data['Pre_L_1'], Data['Pre_L_2']

	# Lambda for each node
	Lambda_1_j = Lambda_1 * e
	Lambda_2_j = Lambda_2 * f
	Lambda_1_2_j = Lambda_1_2 * e
	Lambda_2_2_j = Lambda_2_2 * f

	# Get number of units for each subsystem
	N_med = N - N_2
	N_fire = N - N_1

	# Number of states for each subsystem
	State_med = 2 ** N_med
	State_fire = 2 ** N_fire

	if len(units_e) == 0:
		units_e = [x for x in range(N) if x not in list(range(N_1,N_1+N_2))]
		units_f = [x for x in range(N) if x not in list(range(N_1))]

	units_e_reorder = np.array(units_e).argsort().argsort()
	units_f_reorder = np.array(units_f).argsort().argsort()
	# Calculate Rho
	Rho_med = (Lambda_1+2*Lambda_1_2)/(Mu_1*N_med) 
	Rho_fire = (Lambda_2+2*Lambda_2_2)/(Mu_2*N_fire) 

	P_k_e = Solve_MultiDisp_PN(Lambda_1,Lambda_1_2, Mu_1, N_med)
	P_k_f = Solve_MultiDisp_PN(Lambda_2,Lambda_2_2, Mu_2, N_fire)

	# Initialization r
	r_e = sum(P_k_e*list(range(N_med+1)))/N_med # average utilization over all servers
	rho_i_e = np.full(N_med,r_e) # utilization of each unit i
	r_f = sum(P_k_f*list(range(N_fire+1)))/N_fire # average utilization over all servers
	rho_i_f = np.full(N_fire,r_f) # utilization of each unit i

	# Calculate Q
	Q_e_1 = Q_Calculation(N = N_med, P_k = P_k_e, Rho = r_e)
	Q_e_2 = Q_Calculation_2(N = N_med, P_k = P_k_e, Rho = r_e)
	# Calculate Q
	Q_f_1 = Q_Calculation(N = N_fire, P_k = P_k_f, Rho = r_f)
	Q_f_2 = Q_Calculation_2(N = N_fire, P_k = P_k_f, Rho = r_f)
		
		
	# Preference List Calculate G
	G_e = [[np.where(Pre_L_1[:,i] == j)[0] for i in range(N_med)] for j in range(N_med)] 
	G_f = [[np.where(Pre_L_2[:,i] == j)[0] for i in range(N_fire)] for j in range(N_fire)] 

	start_time = time.time()

	# Initialize alpha_e
	alpha_f_old = np.ones(N_med)
	alpha_f = np.zeros(N_med)
	alpha_e = np.zeros(N_fire)

	while (max(abs(alpha_f_old-alpha_f)) > 0.00005):
		# Medical Subsystem
		# Start Approximation
		# concatenate the alpha e
		if N-N_1-N_2 == 0:
			alpha_other = np.zeros(N_1)
		else:
			alpha_other = np.concatenate((np.zeros(N_1), alpha_e[-(N-N_1-N_2):]), axis=None)
		rho_i_e = Chelst_Approx(N_med, Mu_1, Q_e_1, Q_e_2, Pre_L_1,G_e,Lambda_1_j,Lambda_1_2_j, r_e, rho_i_e, alpha_f, alpha_other=alpha_other, epsilon=0.000033)
		# Get alpha_f
		alpha_e = np.zeros(N_fire)
		for k in range(N_2,N_fire):
			alpha_e[units_f_reorder[k]] = rho_i_e[units_e_reorder[k+N_1-N_2]]/(rho_i_e[units_e_reorder[k+N_1-N_2]]+(1-rho_i_e[units_e_reorder[k+N_1-N_2]])*(1-alpha_f[units_e_reorder[k+N_1-N_2]]))
		
		# Fire Subsystem
		# Start Approximation
		#Q_f = Q_Calculation(N = N_fire, P_k = P_k_f, Rho = r_f)
		# concatenate the alpha f
		if N-N_1-N_2 == 0:
			alpha_other = np.zeros(N_2)
		else:
			alpha_other = np.concatenate((np.zeros(N_2), alpha_f[-(N-N_1-N_2):]), axis=None)
		rho_i_f = Chelst_Approx(N_fire, Mu_2, Q_f_1,Q_f_2, Pre_L_2, G_f, Lambda_2_j,Lambda_2_2_j, r_f, rho_i_f, alpha_e, alpha_other=alpha_other, epsilon=0.000033)
		# Get alpha_e
		alpha_f_old = alpha_f  # The old is for checking whether to exit or not. 
		alpha_f = np.zeros(N_med)
		for k in range(N_1,N_med):
			alpha_f[units_e_reorder[k]] = rho_i_f[units_f_reorder[k+N_2-N_1]]/(rho_i_f[units_f_reorder[k+N_2-N_1]]+(1-rho_i_f[units_f_reorder[k+N_2-N_1]])*(1-alpha_e[units_f_reorder[k+N_2-N_1]]))
		
	print("--- %s seconds ---" % (time.time() - start_time))
	print(['%.4f' % x for x in rho_i_e])
	print(['%.4f' % x for x in rho_i_f])
	print(['%.4f' % x for x in alpha_e])
	print(['%.4f' % x for x in alpha_f])
	return rho_i_e, rho_i_f, alpha_e, alpha_f

def Larson_Approx(N,mu,Q,Pre_L,G,Lambda_j,r,rho_i_ef,alpha,alpha_other, epsilon=0.0001):
	# Step 0: Initialization
	n = 0
	# Step 1: Iteration
	rho_i_new = np.zeros(N) # temporary utilizations to store value at each step
	Indicator = True # Indicates when the program should stop
	#rho_i_fe = (1-alpha_other)*alpha/(1-alpha_other*alpha)  # This is derived from the alpha updates. Exactly the same as rho_i_other
	#print('alpha:',alpha)
	#print('rho_i_fe:',rho_i_fe)
	while (Indicator):
		n += 1
		#rho_i = rho_i_ef + rho_i_fe
		rho_i = rho_i_ef + (1-rho_i_ef)*alpha # Total rho equals to the rho_med + free*alpha. Then update rho_med
		#print('rho_i:',rho_i)
		
		for i in range(N):
			value = 1
			for k in range(N):
				temp = 0
				for j in G[i][k]:
					rho_prod = 1
					for l in range(k):
						rho_prod *= rho_i[Pre_L[j][l]]
					temp += Lambda_j[j]*Q[k]*rho_prod
				value += 1/mu*temp  # There should be a 1/mu here. because we don't assume it to be 1. 
			rho_i_new[i]= (1-((1-rho_i_ef)*alpha)[i])*(1-1/value)  # Need to multiple (1-rho_i_fe) to this according to the test derivation
			# In the test version here ,we are using  (1-rho_i_ef)*alpha to update rho_i_fe
		
		# Step 2: Normalize. 
		# Here the normalizing factor only takes on 1 service because we assume adding the other does not change much. 
		Gamma = 1/(rho_i_new.sum()/r/N)  # R is only used here for normalization
		
		rho_i_new *= Gamma
		rho_i_ef_new = rho_i_new
		#print (rho_i_ef_new)
		
		# Step 3: Convergence Test
		if abs(rho_i_ef_new - rho_i_ef).max() < epsilon:
			Indicator = False
			print ('Program stop in',n,'iterations')
			print (rho_i_ef_new)
			print('Total Rho:',sum(rho_i_ef_new))
			return rho_i_ef_new
		rho_i_ef = np.array(rho_i_ef_new)

# Returns the whole steady state distribution of 
# HEre the Rho is Lambda/N*Mu, and i is the state you want to look at


def Chelst_Approx(N,mu,Q_1,Q_2,Pre_L,G,Lambda_j,Lambda_2_j,r,rho_i_ef,alpha,alpha_other,epsilon=0.000033):
	# Step 0: Initialization
	n = 0
	# Step 1: Iteration
	rho_i_new = np.zeros(N) # temporary utilizations to store value at each step
	rho_i_fe = (1-alpha_other)*alpha/(1-alpha_other*alpha)  # This is derived from the alpha updates. Exactly the same as rho_i_other
	Indicator = True # Indicates when the program should stop
	while (Indicator):
		n += 1
		rho_i = rho_i_ef + (1-rho_i_ef)*alpha # Total rho equals to the rho_med + free*alpha. Then update rho_med
		for i in range(N):
			value = 1
			for k in range(N): # This is to sum up to N for Nth prefrerred
				temp_1 = 0 # For single
				temp_2 = 0 # For double
				for j in G[i][k]:
					# For single case
					rho_prod_1 = 1
					for l in range(k):
						rho_prod_1 *= rho_i[Pre_L[j][l]]
					temp_1 += (Lambda_j[j]+Lambda_2_j[j])*Q_1[k]*rho_prod_1
					rho_prod_2 = 1

					# For double case
					for m in range(k):
						rho_prod_2 = 1
						for l in range(k):
							if l != m:
								rho_prod_2 *= rho_i[Pre_L[j][l]]
							else:
								rho_prod_2 *= 1- rho_i[Pre_L[j][l]]
						temp_2 += Lambda_2_j[j]*Q_2[k]*rho_prod_2
				value += 1/mu*(temp_1+temp_2)
			rho_i_new[i]= (1-((1-rho_i_ef)*alpha)[i])*(1-1/value)
		print(rho_i_new)

		# Step 2: Normalize
		Gamma = 1/(rho_i_new.sum()/r/N)
		rho_i_new *= Gamma
		rho_i_ef_new = rho_i_new

		# Step 3: Convergence Test
		if abs(rho_i_ef_new - rho_i_ef).max() < epsilon:
			Indicator = False
			print ('Program stop in',n,'iterations')
			print (rho_i_ef_new)
			print('Total Rho:',sum(rho_i_ef_new))
			return rho_i_ef_new
		rho_i_ef = np.array(rho_i_ef_new)

# Calculate the Q Factor. For the lost system only, 
def Q_Calculation_Old(N,P_0,P_N,Rho):
	Q = np.zeros(N)
	for j in range(N):
		numerator = sum([math.factorial(N-j-1) * (N-k) / math.factorial(k-j)* N**k / math.factorial(N) * Rho**(k-j) for k in range(j,N)])
		denominator = (1-Rho) / P_0
		Q[j] = numerator/denominator / ((1-P_N)**j) / (1+Rho * P_N / (1-Rho)) # This part is to compensate for the loss system
	return Q

# Calculate the Q Factor. For the general case. Here for MM0, replace Rho by r. 
def Q_Calculation(N,P_k,Rho):
	Q = np.zeros(N)
	for j in range(N):
		Q[j] = sum([math.factorial(k)/math.factorial(k-j) * math.factorial(N-j)/math.factorial(N)* (N-k)/(N-j) * P_k[k] for k in range(j,N)])/ (Rho**(j) * (1-Rho))
	return Q

def Q_Calculation_2(N,P_k,Rho):
	Q_2 = np.zeros(N)
	for j in range(1,N):
		numerator = sum([comb(k,j-1)/comb(N,j)* (N-k)/j * (N-k-1) /(N-j)*P_k[k] for k in range(j-1,N)])
		denominator = (1-Rho)**2 * Rho**(j-1)
		Q_2[j] = numerator/denominator
	return Q_2

def MMN0_i(N,Rho,i=0):
	P_0 = 1 / (sum([N**i/math.factorial(i)*Rho**i for i in range(N+1)]))
	P_i = i**i * Rho**i * P_0 / math.factorial(i)
	return P_i

def ErlangLoss(Lambda, Mu, N=0):
	# The Erlangloss Model is exactly the same as MMN0 and this returns the whole probability distribution
	# Solves the Erlang loss system
	if N!=0:
		Lambda = np.ones(N)*Lambda
		Mu = Mu*(np.array(range(N))+1)
	else:
		N = len(Lambda)
	LoM = [1] + [l/m for l,m in zip(Lambda, Mu)]
	Prod = [np.prod(LoM[0:i]) for i in range(1,N+2)]
	P_n = Prod/sum(Prod)
	return P_n

def Solve_MultiDisp_PN(Lambda_1,Lambda_2,Mu,N): 
	# Here Lambda_1 is the arrival for calls requiring 1 unit
	# and Lambda_2 are for requiring 2 units. 
	# This is to solve P_N for multiple dispatch as in Chelst Paper. 
	Lambda = Lambda_1 + Lambda_2
	P_i = np.ones(N+1)
	P_i[1] = Lambda/Mu*P_i[0]
	P_i[2] = 1/(2*Mu)*((Lambda+Mu)*P_i[1]-Lambda_1*P_i[0])
	for i in range(3,N):
		P_i[i] = 1/(i*Mu)*((Lambda+(i-1)*Mu)*P_i[i-1]-Lambda_1*P_i[i-2]-Lambda_2*P_i[i-3])
	P_i[N] = 1/(N*Mu)*(Lambda_2*P_i[N-2]+Lambda*P_i[N-1])
	P_i /= sum(P_i)
	return P_i

def response_frac_approx(Data, rho_e, rho_f, units_e, units_f):
	# Here rho is the total rho for both services
	N, N_1, N_2, K = Data['N'], Data['N_1'], Data['N_2'], Data['K']
	Lambda_1, Lambda_2, Mu_1, Mu_2, Lambda_1_2, Lambda_2_2 = Data['Lambda_1'], Data['Lambda_2'], Data['Mu_1'], Data['Mu_2'], Data['Lambda_1_2'], Data['Lambda_2_2']
	frac_e, frac_f = Data['e'], Data['f']
	Pre_1, Pre_2 = Data['Pre_L_1'], Data['Pre_L_2']

	N_med = N - N_2
	N_fire = N - N_1
	print(units_e,units_f)
	units_e_reorder = np.array(units_e).argsort().argsort()
	units_f_reorder = np.array(units_f).argsort().argsort()

	insert_f = units_e_reorder[:N_1]
	insert_e = units_f_reorder[:N_2]

	# rho_e_total = rho_e + np.pad(rho_f[N_2:], (N_1, 0),'constant', constant_values=0)
	# rho_f_total = rho_f + np.pad(rho_e[N_1:], (N_2, 0),'constant', constant_values=0)
	if (N-N_1-N_2 == 0):
		rho_e_total = rho_e
		rho_f_total = rho_f
	else:
		rho_e_total = rho_e + np.insert(rho_f[units_f_reorder[N_2:]],insert_f,0)
		rho_f_total = rho_f + np.insert(rho_e[units_e_reorder[N_1:]],insert_e,0)
	#print(rho_e_total)
	#print(rho_f_total)

	P_k_e = Solve_MultiDisp_PN(Lambda_1,Lambda_1_2, Mu_1, N_med)
	P_k_f = Solve_MultiDisp_PN(Lambda_2,Lambda_2_2, Mu_2, N_fire)
	
	r_e = sum(P_k_e*list(range(N_med+1)))/N_med # average utilization over all servers
	r_f = sum(P_k_f*list(range(N_fire+1)))/N_fire # average utilization over all servers

	Q_e = Q_Calculation(N = N_med, P_k = P_k_e, Rho = r_e)
	Q_f = Q_Calculation(N = N_fire, P_k = P_k_f, Rho = r_f)
	Q_f_2 = Q_Calculation_2(N = N_fire, P_k = P_k_f, Rho = r_f)
	
	q_e = np.zeros([K,N_med])
	q_f = np.zeros([K,N_fire])
	
	for m in range(K):
		pre_k = Pre_1[m]
		for j in range(N_med):
			q_e[m,pre_k[j]] = Q_e[j]*np.prod(rho_e_total[pre_k[:j]])*(1-rho_e_total[pre_k[j]])

	single_frac = Lambda_2/(Lambda_2+Lambda_2_2)

	for m in range(K):
		pre_k = Pre_2[m]
		for j in range(N_fire):
			q_f[m,pre_k[j]] = Q_f[j]*np.prod(rho_f_total[pre_k[:j]])*(1-rho_f_total[pre_k[j]])

	q_f_1 = np.zeros([K,N_fire]) # Different from q_f. This is for sending 1 unit for type 2 call
	q_f_2 = np.zeros([K,N_fire,N_fire]) #This is for sending 2 units for type 2 call
	q_f_agg = np.zeros([K,N_fire])

	for m in range(K):
		pre_k = Pre_2[m]
		for j in range(N_fire):
			for k in range(j):
				q_f_2[m,pre_k[k],pre_k[j]] = Q_f_2[j]*np.prod(rho_f_total[np.delete(pre_k[:j],k)])*(1-rho_f_total[pre_k[j]])*(1-rho_f_total[pre_k[k]])
				q_f_agg[m,pre_k[k]] += q_f_2[m,pre_k[k],pre_k[j]]

	for m in range(K):
		pre_k = Pre_2[m]
		for j in range(N_fire):
			q_f_1[m,pre_k[j]] = Q_f[j]*np.prod(rho_f_total[np.delete(pre_k,j)])*(1-rho_f_total[pre_k[j]])
	return q_e, q_f, q_f_1, q_f_2, q_f_agg

def response_time_approx(Data, rho_e, rho_f, distance_file = "distance.csv", units_e = [], units_f = []):
	# Here rho is the total rho for both services
	N, N_1, N_2, K = Data['N'], Data['N_1'], Data['N_2'], Data['K']
	Lambda_1, Lambda_2, Mu_1, Mu_2, Lambda_1_2, Lambda_2_2 = Data['Lambda_1'], Data['Lambda_2'], Data['Mu_1'], Data['Mu_2'], Data['Lambda_1_2'], Data['Lambda_2_2']
	frac_e, frac_f = Data['e'], Data['f']

	if len(units_e) == 0:
		units_e = [x for x in range(N) if x not in list(range(N_1,N_1+N_2))]
		units_f = [x for x in range(N) if x not in list(range(N_1))]
	# N_fire = N - N_1
	q_e, q_f, q_f_1, q_f_2, q_f_agg = response_frac_approx(Data, rho_e, rho_f, units_e, units_f)

	q_e = (q_e.T/q_e.sum(axis=1)).T

	single_frac = Lambda_2/(Lambda_2+Lambda_2_2)
	q_f = (q_f.T/q_f.sum(axis=1)).T
	q_f2 = q_f_agg+q_f_1
	q_f2 = (q_f2.T/q_f2.sum(axis=1)).T
	q_f_total = (1-single_frac)*(q_f2)+single_frac*q_f
	#q_f_total = (q_f_total.T/q_f_total.sum(axis=1)).T

	distance = pd.read_csv(distance_file)
	distance = distance.drop(distance.columns[0], axis=1)
	distance = distance/1600
	distance[distance > 0.7712735431536174] = (distance[distance > 0.7712735431536174]*111.51113889304331+86.005591195132666)/60
	distance[distance < 0.7712735431536174] = 195.86302790816589*np.sqrt(distance[distance < 0.7712735431536174])/60

	time_e = (distance.iloc[:,np.sort(units_e)]*q_e).sum(axis=1)
	time_f = (distance.iloc[:,np.sort(units_f)]*q_f).sum(axis=1)

	return time_e, time_f

def P_rand_arr(Lambda_v, Mu_v, alpha):
	# This one takes into account the actual probability of getting into each state
	alpha = np.array(alpha)
	N = len(alpha)
	all_set = list(range(N))
	Lambda = np.ones(N)
	for i in range(N):
		comb = list(combinations(list(range(N)), N-i)) # All the combinations of N choose N-i units
		comb_size=len(comb) # Number of total combinations
		Prob_comb_list=np.array([Prob_comb(N,list(set(all_set)-set(c)),alpha) for c in comb]) # The probability of obtaining each combination given alpha
		Prob_comb_list = Prob_comb_list/sum(Prob_comb_list) # Normalize
		Lambda[i] = Lambda_v*sum([(1-np.prod(alpha[list(comb[c])]))*Prob_comb_list[c] for c in range(comb_size)])
	
	Mu = Mu_v*(np.array(range(N))+1)
	LoM = [1] + [l/m for l,m in zip(Lambda, Mu)]
	Prod = [np.prod(LoM[0:i]) for i in range(1,N+2)]
	P_n = Prod/sum(Prod)
	return P_n,Lambda

def P_equal_prob(Lambda_v, Mu_v, alpha):
	'''
		:Lambda_v, Mu_v, alpha: Inputs
		"Output: P_n, Lambda
		This one assumes every combination is with equal probability. Getting the steady state probability P_n
	'''
	alpha = alpha
	N = len(alpha)
	Lambda = np.ones(N)*Lambda_v # Lambdas of the birth and death chain
	for i in range(N):
		num_comb = comb(N,i+1)
		Lambda[N-i-1] = Lambda_v/num_comb*sum([1-np.prod(j) for j in list(itertools.combinations(alpha, i+1))]) # Using the equal probability
		#print(Lambda[N-i-1])
		# When there is no much difference of lambdas, we just assume it is Lambda_v rather than do massive computation
		if Lambda_v - Lambda[N-i-1] < 0.001:
			break
	#print('Lambda',Lambda)
	Mu = Mu_v*(np.array(range(N))+1)
	LoM = [1] + [l/m for l,m in zip(Lambda, Mu)]
	Prod = [np.prod(LoM[0:i]) for i in range(1,N+2)]
	P_n = Prod/sum(Prod)
	return P_n, Lambda

def Prob_comb(N,comb,alpha):
	# Given combination, which shows the busy servers, and alpha, calculate the probability of getting this combination
	# which is the sum of all cases that cause comb to be the busy servers.
	alpha = np.array(alpha)
	one_minus_alpha = 1-alpha
	comb = np.array(comb)
	comb_size = len(comb) # How many units in this combination
	if comb_size == 0: return 1 # If there is no unit, the probability is 1
	all_set = list(range(N)) # The set of all units, there are in total N units
	not_comb = np.array(list(set(all_set) - set(comb))) # The set of units that are not in the combination
	power_set = list(powerset(not_comb)) # The power set of set not_comb
	power_set_size = len(power_set) # size of the power set
	prob_list = np.zeros(power_set_size)
	for i in range(power_set_size):
		set_i = np.array(power_set[i]) # a set of the power set
		pow_size = len(set_i) # The number of units in this set 
		if pow_size == 0:
			# If the size of set i is 0, 
			prob_list[i] = comb_size * math.factorial(comb_size+pow_size-1) * math.factorial(N-comb_size-pow_size)
		else:
			prob_list[i] = comb_size * math.factorial(comb_size+pow_size-1) * math.factorial(N-comb_size-pow_size) * np.prod(alpha[set_i])
	prob_list = prob_list/math.factorial(N)*np.prod(one_minus_alpha[comb])
	prob = sum(prob_list)
	return prob
 
def P_withbreak_1step(Lambda_v, Mu_v, alpha, old_Lambda=None, P_n=None):
	alpha = np.array(alpha)
	N = len(alpha)
	all_set = list(range(N))
	Lambda = np.ones(N)
	for i in range(N):
		comb = list(combinations(list(range(N)), N-i))
		comb_size=len(comb)
		Prob_comb_list=np.array([Prob_comb(N,list(set(all_set)-set(c)),alpha) for c in comb])
		Prob_comb_list = Prob_comb_list/sum(Prob_comb_list)
		if i > 0:
			p = old_Lambda[i-1]*P_n[i-1]/(old_Lambda[i-1]*P_n[i-1]+(i+1)*Mu_v*P_n[i+1])
			Prob_comb_list = Prob_comb_list*p + 1/scipy.special.comb(N,i)*(1-p)
		Lambda[i] = Lambda_v*sum([(1-np.prod(alpha[list(comb[c])]))*Prob_comb_list[c] for c in range(comb_size)])

	Mu = Mu_v*(np.array(range(N))+1)
	LoM = [1] + [l/m for l,m in zip(Lambda, Mu)]
	Prod = [np.prod(LoM[0:i]) for i in range(1,N+2)]
	P_n = Prod/sum(Prod)
	#print(Lambda)
	return P_n,Lambda


def P_withbreak_new(Lambda_v, Mu_v, alpha):
	P_n, Lambda = P_rand_arr(Lambda_v, Mu_v, alpha)
	for _ in range(5):
		P_n, Lambda = P_withbreak_1step(Lambda_v, Mu_v, alpha, old_Lambda=Lambda, P_n=P_n)
	return P_n,Lambda

def P_withbreak_pref(Lambda_v, Mu_v, alpha, Pre, prob):
	alpha = np.array(alpha)
	N = len(alpha)
	all_set = list(range(N))
	Lambda = np.ones(N)
	dict_comb = {2**N-1:1}
	for i in range(N):
		comb = list(combinations(list(range(N)), i))
		comb_size=len(comb)
		Prob_comb_list=np.array([Prob_comb_pref(N,c,alpha,Pre, prob) for c in comb])
		Prob_comb_list = Prob_comb_list/sum(Prob_comb_list)
		Lambda[i] = Lambda_v*sum([(1-np.prod(alpha[list(set(all_set)-set(comb[c]))]))*Prob_comb_list[c] for c in range(comb_size)])
		comb_id = map(lambda x:sum(2**np.array(x)), comb)
		dict_temp = dict(zip(comb_id, Prob_comb_list))
		dict_comb = {**dict_temp, **dict_comb}
	Mu = Mu_v*(np.array(range(N))+1)
	LoM = [1] + [l/m for l,m in zip(Lambda, Mu)]
	Prod = [np.prod(LoM[0:i]) for i in range(1,N+2)]
	P_n = Prod/sum(Prod)
	return P_n,Lambda, dict_comb

def P_withbreak_pref_1step(Lambda_v, Mu_v, alpha, Pre, prob, old_Lambda=None, dict_old = None, P_n=None):
	# This is just a rough estimation to start with
	alpha = np.array(alpha)
	N = len(alpha)
	all_set = list(range(N))
	Lambda = np.ones(N)
	for i in range(N):
		comb = list(combinations(list(range(N)), i))
		comb_size=len(comb)
		Prob_comb_list=np.array([Prob_comb_pref(N,c,alpha,Pre, prob) for c in comb])
		Prob_comb_list = Prob_comb_list/sum(Prob_comb_list)
		if i > 0:
			p = old_Lambda[i-1]*P_n[i-1]/(old_Lambda[i-1]*P_n[i-1]+(i+1)*Mu_v*P_n[i+1])
			Prob_comb_list = Prob_comb_list*p + 1/scipy.special.comb(N,i)*(1-p)
		Lambda[i] = Lambda_v*sum([(1-np.prod(alpha[list(set(all_set)-set(comb[c]))]))*Prob_comb_list[c] for c in range(comb_size)])
		comb_id = map(lambda x:sum(2**np.array(x)), comb)
		dict_old.update(dict(zip(comb_id, Prob_comb_list)))

	Mu = Mu_v*(np.array(range(N))+1)
	LoM = [1] + [l/m for l,m in zip(Lambda, Mu)]
	Prod = [np.prod(LoM[0:i]) for i in range(1,N+2)]
	P_n = Prod/sum(Prod)
	#print(Lambda)
	return P_n,Lambda,dict_old

def P_withbreak_pref_1step_try(Lambda_v, Mu_v, alpha, inflow, outflow, dict_old=None, P_n=None):
	# This uses conservation of flow and seems work well!
	alpha = np.array(alpha)
	N = len(alpha)
	all_set = list(range(N))
	Lambda = np.ones(N)
	for i in range(N):
		comb = list(combinations(list(range(N)), i))
		comb_size=len(comb)
		if i == 0:
			Prob_comb_list = np.array([1])
		else:
			Prob_comb_list=np.array([Prob_comb_pref_update(c,inflow,outflow,dict_old,P_n) for c in comb])
			Prob_comb_list = Prob_comb_list/sum(Prob_comb_list)
		comb_id = map(lambda x:sum(2**np.array(x)), comb)
		dict_old.update(dict(zip(comb_id, Prob_comb_list)))
		Lambda[i] = Lambda_v*sum([(1-np.prod(alpha[list(set(all_set)-set(comb[c]))]))*Prob_comb_list[c] for c in range(comb_size)])
		
	Mu = Mu_v*(np.array(range(N))+1)
	LoM = [1] + [l/m for l,m in zip(Lambda, Mu)]
	Prod = [np.prod(LoM[0:i]) for i in range(1,N+2)]
	P_n = Prod/sum(Prod)
	#print(Lambda)
	return P_n,Lambda,dict_old


def countSetBits(num): 
	 # convert given number into binary 
	 # output: which bits has 1
	 binary = bin(num) 
  
	 # now separate out all 1's from binary string 
	 # we need to skip starting two characters 
	 # of binary string i.e; 0b 
	 setBits = [ones for ones in binary[2:] if ones=='1'] 
	 return len(setBits) 

def Prob_comb_pref_update(comb,inflow, outflow, dict_old,P_n):
	state_id = sum(2**np.array(comb))
	into_id = np.where(inflow[state_id]!=0)[0]
	into_state_id = list(map(countSetBits,into_id))
	pr = np.array([dict_old[x] for x in into_id])
	numerator = sum(P_n[into_state_id] * pr * inflow[state_id][into_id])
	denominator = outflow[state_id]
	return numerator/denominator


def Prob_comb_pref(N,comb,alpha,Pre, prob):
	### Calculate the probability of getting to state comb given alpha, Preference list and prob distrition of calls
	alpha = np.array(alpha)
	comb = np.array(comb)
	comb_size = len(comb)
	all_set = list(range(N))
	not_comb = np.array(list(set(all_set) - set(comb)))
	one_minus_alpha = 1-alpha
	K = len(prob)
	if comb_size == 0: return 1
	#if N-comb_size == 0: return 1
	prob_list = np.ones(K)
	for i in range(K):
		pre = Pre[i]
		n_c_input = []
		for j in range(N-1,-1,-1):
			if pre[j] in comb:
				n_c_input = np.array(list(set(pre[:j]) - set(comb)))
				break
		if len(n_c_input) == 0:
			prob_list[i] = prob[i]*np.prod(one_minus_alpha[comb])
		else:
			prob_list[i] = prob[i]*np.prod(alpha[n_c_input])*np.prod(one_minus_alpha[comb])
	comb_prob = sum(prob_list)
	return comb_prob



#### For Random 
def P_withbreak_rand(Lambda_v, Mu_v, alpha):
	# This one takes into account the actual probability of getting into each state
	alpha = np.array(alpha)
	N = len(alpha)
	all_set = list(range(N))
	Lambda = np.ones(N)
	dict_comb = {2**N-1:1}
	for i in range(N):
		comb = list(combinations(list(range(N)), N-i))
		comb_size=len(comb)
		Prob_comb_list=np.array([Prob_comb(N,list(set(all_set)-set(c)),alpha) for c in comb])
		Prob_comb_list = Prob_comb_list/sum(Prob_comb_list)
		Lambda[i] = Lambda_v*sum([(1-np.prod(alpha[list(comb[c])]))*Prob_comb_list[c] for c in range(comb_size)])
		comb_id = map(lambda x:sum(2**np.array(x)), comb)
		dict_temp = dict(zip(comb_id, Prob_comb_list))
		dict_comb = {**dict_temp, **dict_comb}

	Mu = Mu_v*(np.array(range(N))+1)
	LoM = [1] + [l/m for l,m in zip(Lambda, Mu)]
	Prod = [np.prod(LoM[0:i]) for i in range(1,N+2)]
	P_n = Prod/sum(Prod)
	return P_n,Lambda,dict_comb

def P_withbreak_rand_new(Lambda_v, Mu_v, alpha,statusmat,busy_list,transdown):
	N = len(alpha)
	transup = Calculate_transup_random(N,alpha,statusmat,busy_list,Lambda_v)
	inflow = (transup+transdown).T
	outflow = inflow.sum(axis=0)
	start_time = time.time()
	P_n, Lambda,dict_old = P_withbreak_rand(Lambda_v, Mu_v, alpha)
	for _ in range(0):
		P_n, Lambda,dict_old = P_withbreak_pref_1step_try(Lambda_v, Mu_v, alpha, inflow, outflow, dict_old=dict_old, P_n=P_n)
	print("------ %s seconds ------" % (time.time() - start_time))
	return P_n,Lambda,dict_old

#@pysnooper.snoop()
def P_withbreak_pref_new(Lambda_v, Mu_v, alpha, Pre, prob,statusmat_sub,busy_sub, transdown):
	start_time = time.time()
	N_sub = len(alpha)
	K = len(prob)
	transup = Calculate_transup(N_sub,alpha,statusmat_sub,busy_sub,Lambda_v,Pre, prob)
	# Inflow
	inflow = (transup+transdown).T
	# Outflow
	outflow = inflow.sum(axis=0)
	P_n, Lambda,dict_old  = P_withbreak_pref(Lambda_v, Mu_v, alpha, Pre, prob)
	#print([dict_old[i] for i in range(2**N_sub)])
	if np.all(alpha == 1):
		for _ in range(10):
			P_n, Lambda, dict_old = P_withbreak_pref_1step(Lambda_v, Mu_v, alpha, Pre, prob, old_Lambda=Lambda,dict_old=dict_old, P_n=P_n)
	else:
		for _ in range(10):
			P_n, Lambda, dict_old = P_withbreak_pref_1step_try(Lambda_v, Mu_v, alpha, inflow, outflow, dict_old=dict_old, P_n=P_n)
			# print([dict_old[i] for i in range(2**N_sub)])
	print("------ %s seconds ------" % (time.time() - start_time))
	return P_n,Lambda,dict_old

#############################################################################################
####################### Birth and Death Queue ###############################################
#@pysnooper.snoop()
def BnD_hypercube(Lambda_v, Mu_v, alpha, Pre, prob,statusmat_sub,busy_sub, transdown):
	#start_time = time.time()
	N_sub = len(alpha)
	K = len(prob)

	transup = Calculate_transup(N_sub,alpha,statusmat_sub,busy_sub,Lambda_v,Pre, prob) # this and transdown are fixed
	# Inflow
	inflow = (transup+transdown).T
	# Outflow
	outflow = inflow.sum(axis=0)
	# Solves the initial values of 
	# Here the Mu's and Lambda's will be used at each iteration
	#P_n, Lambda,dict_comb  = P_withbreak_pref(Lambda_v, Mu_v, alpha, Pre, prob)
	# print(dict_comb)
	
	Mu = Mu_v*(np.array(range(N_sub))+1)
	
	#p_n_B = np.ones(2**N_sub)
	#p_n_B_new = np.array([dict_comb[i] for i in range(2**N_sub)])

	# Get ID array 
	len_list = np.array(list(map(lambda x:len(x),busy_sub)))
	numcomb_for_state = np.array(list(map(lambda x:comb(N_sub,x),len_list)))
	#print(len_list)
	id_array = [list(np.where(len_list==i)[0]) for i in range(N_sub+1)]
	
	p_n_B = np.zeros(2**N_sub)
	p_n_B_new = 1/numcomb_for_state
	#print(p_n_B_new)
	Lambda = np.ones(N_sub)

	for i in range(N_sub):
		Lambda[i] = np.dot(p_n_B_new[id_array[i]],transup[id_array[i]].sum(axis=1))

	#print(transup)
	ite = 0
	time_list = []
	start_time = time.time()
	
	while (max(np.abs(p_n_B - p_n_B_new))>0.002):
		p_n_B = np.copy(p_n_B_new)
		start_time_ite = time.time()
		#print("l---- %s seconds ------" % (time.time() - start_time))
		# Now starts to update. We don't have to update the first and last state which is always 1
		for n in range(1,N_sub):
			#for j in id_array[n]:
				#p_n_B_new[j] = 1/(n*Mu_v+transup[j].sum())*(np.dot(p_n_B_new[id_array[n-1]],transup[id_array[n-1],j])*Mu[n-1]/Lambda[n-1]+np.dot(p_n_B_new[id_array[n+1]],transdown[id_array[n+1],j])*Lambda[n]/Mu[n])
			# Use test
			id_n = id_array[n]
			p_n_B_new[id_n] = 1/(n*Mu_v+transup[id_n].sum(axis=1))*(transup[id_array[n-1]][:,id_n].T.dot(p_n_B_new[id_array[n-1]])*Mu[n-1]/Lambda[n-1]+transdown[id_array[n+1]][:,id_n].T.dot(p_n_B_new[id_array[n+1]])*Lambda[n]/Mu[n])
			#print(n,max(np.abs(p_n_B[id_array[n]] - p_n_B_new[id_array[n]])))
			# print("----- %s seconds ------" % (time.time() - start_time))
			time_list += [time.time() - start_time_ite]
			start_time_ite = time.time()
			# Use old
			#p_n_B_new[id_array[n]] = 1/(n*Mu_v+transup[id_array[n]].sum(axis=1))*(transup[id_array[n-1]][:,id_array[n]].T.dot(p_n_B[id_array[n-1]])*Mu[n-1]/Lambda[n-1]+transdown[id_array[n+1]][:,id_array[n]].T.dot(p_n_B[id_array[n+1]])*Lambda[n]/Mu[n])
			#print(sum(p_n_B_new[id_array[n]]))
			#Lambda[n] = np.dot(p_n_B_new[id_array[n]],transup[id_array[n]].sum(axis=1))
			#print("l---- %s seconds ------" % (time.time() - start_time))
			#print(Lambda)
		#print("------ %s ite ------" % (ite))
		#print(p_n_B_new)
		ite += 1
		# Update Lambda
		# Lambda = np.ones(N_sub)
		# for i in range(N_sub):
		# 	Lambda[i] = np.dot(p_n_B_new[id_array[i]],transup[id_array[i]].sum(axis=1))
	# Solve for BnD Solution
	print("------ %s seconds ------" % (time.time() - start_time))
	print(ite)
	LoM = [1] + [l/m for l,m in zip(Lambda, Mu)]
	Prod = [np.prod(LoM[0:i]) for i in range(1,N_sub+2)]
	P_n = Prod/sum(Prod)
	
	p_dist = np.zeros(len(p_n_B_new))
	for c in range(len(p_n_B_new)):
		p = "{0:b}".format(c^0).count('1')
		p_dist[c] = P_n[p]*p_n_B_new[c]
	#print("------ %s seconds ------" % (time.time() - start_time))
	#print(ite)
	return P_n,p_n_B_new,p_dist, time_list

def Calculate_transup_random(N,alpha,statusmat,busy_list,Lambda_v):
	alpha = np.array(alpha)
	N_state = 2**N
	transup = np.zeros([N_state,N_state]) # Initilize 
	all_set = list(range(N))
	for n in range(N_state):
		free = list(set(all_set)-set(busy_list[n]))
		for i in free:
			trans_to = list(set(free)-set([i]))
			trans_to_id = (N_state-1) ^ (sum(2**np.array(trans_to)))
			trans_to_conv = [free.index(i)]
			transup[n,trans_to_id] = Prob_comb(len(free),trans_to_conv,alpha[free])
	return transup*Lambda_v

def Calculate_transdown(N,Mu,busy_list):
	'''
		:N,Mu: Input data
		:busy_list: busy units list for each state
		Output: transdown (2^N by 2^N matrix) the downward transition rates
		Automaticall generate latex code for table in the paper
	'''
	N_state = 2**N
	transdown = np.zeros([N_state,N_state]) # Initilize 
	for n in range(N_state):
		for i in busy_list[n]:
			trans_to = list(set(busy_list[n])-set([i]))
			transdown[n,sum(2**np.array(trans_to))] = Mu
	return transdown

def Calculate_transup(N,alpha,statusmat,busy_list,Lambda_v,Pre, prob):
	### Calculates the rate of entering each state
	alpha = np.array(alpha) # change array to 
	N_state = 2**N
	transup = np.zeros([N_state,N_state]) # Initilize 
	all_set = list(range(N)) # Get all the available sets
	for n in range(N_state):
		free = list(set(all_set)-set(busy_list[n]))  # Gets the free units for each state
		if len(free) != 0:
			# calculate the corresponding preference list for available units for each state
			Pre_l = Pre.flatten()[list(map(lambda x:x in free, Pre.flatten()))].reshape(-1, len(free)).argsort().argsort() 
		for i in free:
			trans_to = list(set(free)-set([i]))	# One state that state n transition to 
			trans_to_id = (N_state-1) ^ (sum(2**np.array(trans_to)))  # The State_Id in number that state n transition to
			trans_to_conv = [free.index(i)] # Index of i in free list
			transup[n,trans_to_id] = Prob_comb_pref(len(free),trans_to_conv,alpha[free],Pre_l, prob) # Calculate the probability of transitioning to state trans_to from state n
	return transup*Lambda_v


#############################################################################################
####################### Miscellaneous #######################################################
def Generate_Latex_Table(N,N_1,N_2,rho_med,rho_fire,rho_med_approx,rho_fire_approx):
	'''
		:N,N_1,N_2,rho_med,rho_fire,rho_med_approx,rho_fire_approx: Input data
		Automaticall generate latex code for table in the paper
	'''
	N_med = N - N_2
	N_fire = N - N_1
	Sep_med = list(range(1,N_1+1))
	Sep_fire = list(range(N_1+1,N_1+N_2+1))
	Med = [x for x in range(1,N+1) if x not in Sep_fire]
	Fire = [x for x in range(1,N+1) if x not in Sep_med]
	latex_code = '\\begin{table}[htbp]\n'+'\\small\n'+' \\centering\n'+' \\renewcommand\\arraystretch{0.5}\n'+'\\caption{Comparison between Exact Model and $\\alpha$-Hypercube for Single Dispatch}\n'
	latex_code += ' \\begin{tabular}{c|c|c|c}\n\\hline\n'
	latex_code += ' \\multicolumn{4}{c}{Saint Paul Case}\\\\\n'
	latex_code += ' \\hline\n Unit & Exact & $\\alpha$-Hypercube & Iterated $\\alpha-Q$ & Error\\\\\n'
	latex_code += '\\hline\n'
	for i in range(N_med):
		latex_code += '$\\rho^e_{{{}}}$ & {:.4f}  & {:.4f} & {:.2f}\\% \\\\\n'.format(Med[i], rho_med[i], rho_med_approx[i],100*abs(rho_med[i]-rho_med_approx[i])/rho_med[i])
	latex_code += '\\hline\n'
	for i in range(N_fire):
		latex_code += '$\\rho^f_{{{}}}$ & {:.4f}  & {:.4f} & {:.2f}\\% \\\\\n'.format(Fire[i], rho_fire[i], rho_fire_approx[i],100*abs(rho_fire[i]-rho_fire_approx[i])/rho_fire[i])
	latex_code += '\\hline\n'
	latex_code += '\\end{tabular}%\n'
	latex_code += '\\end{table}%'
	print(latex_code)

def Generate_Latex_Table_BnD(P_n):
	'''
		:P_n: Input data probability distributions
		Automaticall generate latex code for table in the paper
	'''
	N = int(len(P_n)/12)
	Rem = len(P_n) - 12*N
	latex_code = '\\begin{table}[htbp]\n'+'\\small\n'+' \\centering\n'+' \\renewcommand\\arraystretch{0.5}\n'+'\\caption{ Solution}\n'
	latex_code += ' \\begin{tabular}{c|c|c|c|c|c|c|c|c|c|c|c}\n\\hline\n'
	latex_code += '\\multicolumn{12}{c}{Birth and Death Chain Solution ( secs)}\\\\\n'
	latex_code += '\\hline\n'
	for i in range(N):
		latex_code += '{:.3f}  & {:.3f} & {:.3f} & {:.3f}  & {:.3f} & {:.3f} & {:.3f}  & {:.3f} & {:.3f} & {:.3f}  & {:.3f} & {:.3f} \\\\\n'.format(*P_n[i*12:(i+1)*12])
	rem_code = '{:.3f}  &'*Rem + ' &'*(12-Rem-1) +'\\\\\n'
	latex_code += rem_code.format(*P_n[N*12:])
	latex_code += '\\hline\n'
	latex_code += '\\end{tabular}%\n'
	latex_code += '\\end{table}%'
	print(latex_code)

def separate_all_cases(Data, Pre_1, Pre_2, takeoutnum = 3, takeout_comb=[]):
	N, N_1, N_2,K = Data['N'], Data['N_1'], Data['N_2'], Data['K']
	N_med = N-N_2-takeoutnum
	N_fire = N-N_1-takeoutnum
	N_tot = N_med+N_fire
	if len(takeout_comb) == 0:
		takeout_comb = list(combinations(list(range(N_1+N_2+1, N+1)),takeoutnum))
	ite = 0
	response_list = []

	for c in takeout_comb:
		ite += 1
		units_e = np.array(list(set(Pre_1[0])-set(c)))-1
		units_f = np.array(list(set(Pre_2[0])-set(c)))-1
		Pre_e = []
		Pre_f = []
		for k in range(K):
			Pre_e.append([x for x in Pre_1[k] if x not in c])
			Pre_f.append([x for x in Pre_2[k] if x not in c])
		Pre_e = np.array(Pre_e).argsort().argsort()
		Pre_f = np.array(Pre_f).argsort().argsort()
		Data['N'] = N_tot
		Data['N_1'] = N_med
		Data['N_2'] = N_fire
		Data['Pre_L_1'], Data['Pre_L_2'] = Pre_e, Pre_f
		rho_med, rho_fire, alpha_e, alpha_f,prob_dist_e, prob_dist_f,_,_ = alpha_hypercube(Data)
		response_e,response_e_i = response_time(Pre=Pre_e, steady_dist=prob_dist_e, alpha=alpha_f, frac_call=Data['e'], units = units_e)
		response_f,response_f_i = response_time(Pre=Pre_f, steady_dist=prob_dist_f, alpha=alpha_e, frac_call=Data['f'], units = units_f)
		print(response_e, response_f)
		response_list.append([response_e, response_f])
	return response_list,response_e_i,response_f_i
