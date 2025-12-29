import numpy as np
import pandas as pd

def ClassParam_to_Data(state_class):
    keys = ['K', 'N', 'N_1', 'N_2']
    keys_sub = ['Lambda', 'Mu', 'frac_j', 'pre_list']
    
    Data = {}
    for key in keys:
        Data[key] = state_class.data_dict_1[key]
    for key in keys_sub:
        Data[key+'_1'] = state_class.sub1.data_dict[key]
        Data[key+'_2'] = state_class.sub2.data_dict[key]
    
    Data['Pre_L_1'] = Data.pop('pre_list_1')
    Data['Pre_L_2'] = Data.pop('pre_list_2')
    Data['e'] = Data.pop('frac_j_1')
    Data['f'] = Data.pop('frac_j_2')
    
    Data['Lambda_1_2'], Data['Lambda_2_2'] = 0,0
    
    t_mat_1 = state_class.sub1.data_dict['t_mat']
    t_mat_2 = state_class.sub2.data_dict['t_mat']
    return Data, t_mat_1, t_mat_2

def Data_to_Param(Data):
    N_1, N_2 = Data['N_1'], Data['N_2']
    K = Data['K']
    Lambda_1, Lambda_2 = Data['Lambda_1'], Data['Lambda_2']
    Mu_1, Mu_2 = Data['Mu_1'], Data['Mu_2']
    N = Data['N']
    pre_list_1, pre_list_2 = Data['Pre_L_1'], Data['Pre_L_2']
    frac_j_1, frac_j_2 = Data['e'], Data['f']
    return N_1, N_2, K, Lambda_1, Lambda_2, Mu_1, Mu_2, N, pre_list_1, pre_list_2, frac_j_1, frac_j_2

def Get_Jacobian(A, rho, N_sub, alpha, busy_list, Lambda_v, pre_list, prob_j): # not finished
	'''
		:A, rho, N_sub,alpha,busy_list,Lambda_v,Pre, prob_j: Inputs. prob_j is e or f. 
		Output: J  the jacobian matrix
		A function that helps check the convergence condition
	'''
	# column vector b
	b = np.zeros(2**N_sub).T
	b[-1] = 1
	A_inv = np.linalg.inv(A)
	p_j = A_inv[:,-1]
	############ Some print functions to check ##############
	#print ('A', A)
	#print('A^-1',A_inv)
	# print('alpha:',alpha)
	# print('rho:',rho)
	# print('column sum of A^-1\n',np.linalg.inv(A).sum(axis=0))
	# print('p_j', p_j)
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
		# print(M_i)
		# print('eig(A)\n',np.linalg.eig(A)[0])
		# print('eig(A^-1)\n',np.linalg.eig(np.linalg.inv(A))[0])
		# print('Ai\n',Ai)
		# print('A Ai^-1\n',np.matmul(A,np.linalg.inv(Ai)))
		# print('Ai^-1\n',Ai_inv)
		# print('eig(Ai^-1)\n',np.linalg.eig(np.linalg.inv(Ai))[0])
		# print('column sum of Ai^-1\n',np.linalg.inv(Ai).sum(axis=0))
		# v = np.linalg.inv(Ai).sum(axis=0)[:,np.newaxis] # column sum of Ai^-1
		# print('Ai^-1-A^-1\n',Ai_inv-A_inv)
		# print('A^-1 M_i', A_inv.dot(M_i))
		#print('M_i A^-1', M_i.dot(A_inv))
		# print('A^-1 M_i A^-1\n', A_inv.dot(M_i).dot(A_inv))
		c_i = (M_i.dot(A_inv))[-1,:]
		#print('c_i', c_i)
		# print('row sum of Ai^-1-A^-1\n',(np.linalg.inv(Ai)-np.linalg.inv(A)).sum(axis=1))
		# print('(Ai-rhoiA)\n',Ai-rho[i]*A)
		# print('(Ai^-1-A^-1)A ^T\n',np.matmul(np.linalg.inv(Ai)-np.linalg.inv(A),A).T)
		# print('(Ai^-1 -A^-1) A\n',(Ai_inv-A_inv).dot(A))
		# print('e_i*A^-1', e_i.dot(A_inv))
		#########################################################
		J_norm = 0 # This is the value for |J| which checks for the converegence of the algorithm
		sum_dA = np.zeros([2**N_sub,2**N_sub])
		for j in range(N_sub): # Here j is the m in the paper
			#print('differentiate over unit:', j)
			dA_j = A_prime(N_sub,alpha,busy_list,Lambda_v,pre_list, prob_j, j)
			# sum_dA += dA_j # The sum of matrices dA_j over all units j
			tr_j=np.matrix.trace(np.matmul(Ai_inv-A_inv,dA_j)) # Calculate the trace for the term
			drho_ij = rho[i]*tr_j  # This is drho_ij as calculated in the paper
			############ Some print functions to check ##############
			# print('dA_j\n',dA_j)
			#print('Pk dA_j\n',np.linalg.inv(A)[:,-1]*dA_j)
			#print('row sum of Pk dA\n',(np.linalg.inv(A)[:,-1]*dA_j).sum(axis=1))
			#w = (np.linalg.inv(A)[:,-1]*dA).sum(axis=1)[:,np.newaxis] #row sum of Pk dA_j
			#print('v wT',np.matmul(v.T,w))
			#print('A^-1 dA_j\n',np.matmul(A_inv,dA_j))
			# print('Ai^-1 dA_j',np.matmul(np.linalg.inv(Ai),dA_j))
			# print('(Ai^-1 -A^-1) dA_j\n',(Ai_inv-A_inv).dot(dA_j))
			#print('1/rho_i A^-1 b\n', np.mat(A_inv.dot(b)).T.dot(np.mat(e_i))/rho[i])
			#print('(Ai^-1 -A^-1) dA_j\n',np.matmul(np.linalg.inv(Ai)-np.linalg.inv(A),dA_j))
			#print('eig((Ai^-1 -A^-1) dA_j)\n',np.linalg.eig(np.matmul(np.linalg.inv(Ai)-np.linalg.inv(A),dA))[0])
			#print('rho_i*tr_j:',rho[i]*tr_j) # 2020.1.6 Tested that this is equal to drho_ij
			#print('M_i A^-1 dA_j\n', M_i.dot(A_inv).dot(dA_j))
			# print('sum of c_i dA_j',[sum([c_i[i]*dA_j[i,j] for i in range(2**N_sub)]) for j in range(2**N_sub)])
			#print('drho_ij another method:', -sum([p_j[j]*sum([c_i[i]*dA_j[i,j] for i in range(2**N_sub)]) for j in range(2**N_sub)]))
			# print('drho_ij:',-e_i.dot(A_inv).dot(dA_j).dot(A_inv).dot(b))
			#print('tr(A^-1 M_i A^-1 dA_j)', np.matrix.trace(A_inv.dot(M_i).dot(A_inv).dot(dA_j)))
			#print('A^-1 dA_j A^-1\n',A_inv.dot(dA_j).dot(A_inv))
			#print('column sum of A^-1 dA_j A^-1\n',A_inv.dot(dA_j).dot(A_inv).sum(axis=0))
			#########################################################
			if j == i:
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

def A_prime(N,alpha,busy_list,Lambda_v,Pre, prob,j):  # not finished. need to change to more efficient
	'''
		:N,alpha,busy_list,Lambda_v,Pre, prob,j: Inputs
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