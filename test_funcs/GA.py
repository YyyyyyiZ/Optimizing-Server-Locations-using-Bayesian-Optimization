import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from utils import *
from scipy.special import comb
import time
import copy
import contextlib


from .dispatch_main import *
from .alpha_hypercube import *

class GA:
    def __init__(self, new_N, p, K, Lambda_2, Mu_2, frac_j_2, t_mat):
        self.store_history = {}
        candi = np.arange(1, new_N) # len(candi)=new_N
        self.ga_unit, self.ga_opt, self.ga_path = self.genetic_algorithm(len(candi), p, candi, K, Lambda_2, Mu_2, frac_j_2, t_mat)

    def population_size(self, n, p): # P(n,p) in Alp paper
        d = np.ceil(n/p)
        S = comb(n,p)
        P = np.max([2, np.ceil(n/100 * np.log(S)/d)])*d
        k = P/d
        return int(P), int(k)

    def generate_population(self, n, p, P, candi): # flow Alp paper
        # idea is to first make it a 1-D long array with proper elements and then reshape it
        total_pop_len = P * p # total length of this array
        population = candi # initialize the population
        k = int(np.floor(total_pop_len/n))
        for j in range(1,k): 
            for i in range(j+1): 
                new_pop = candi[np.arange(i,n,j+1)] # this is to with increment 2,3,... j+1
                population = np.append(population,new_pop)
        resi_len = total_pop_len - len(population) # how many elements need to be filled in 
        rand_pop = np.random.choice(candi, resi_len) # randomize
        population = np.append(population, rand_pop)
        population = np.reshape(population,(P,p)).tolist() # reshape to corresponding shape
        return population

    def new_species(self, parents, p, K, Lambda, Mu, frac_j, distance, common_gene):
        all_gene = set(parents[0]+ parents[1]) # all genes represents 
        fixed_gene = set(parents[0]).intersection(set(parents[1])) - set(common_gene) # overlap of the two parents that will be in the test species
        free_gene = (all_gene - fixed_gene)|set(common_gene)# genes candidate for the remaining genes
        new_species = list(all_gene) # initilize the test species
        if len(new_species) == p:
            MRT = self.fitness_evaluation(p,K,Lambda,Mu,frac_j,new_species,distance)
        while len(new_species) > p:
            unit_out, MRT = self.greedy_selection(new_species, free_gene, K, Lambda, Mu, frac_j, distance)
            new_species.remove(unit_out)
            free_gene.remove(unit_out)
        return new_species, MRT
            
    def greedy_selection(self, species, free_gene, K, Lambda, Mu, frac_j, distance):
        MRT_vec = np.zeros(len(free_gene))
        N = len(species) - 1
        for i, u in enumerate(free_gene):
            species_cand = list(set(species)-set([u]))
            MRT_vec[i] = self.fitness_evaluation(N,K,Lambda,Mu,frac_j,species_cand,distance)
        #print('free genes', free_gene)
        #print('MRT from greedy', MRT_vec)
        u_id = np.argmin(MRT_vec)
        min_MRT = np.min(MRT_vec)
        unit = list(free_gene)[u_id]
        return unit, min_MRT

    # fitness function
    def fitness_evaluation(self, N,K,Lambda,Mu,frac_j,units,distance):
        # N: # ambulance
        units_code = sum(2**np.array(units))
        if units_code in self.store_history.keys():
            MRT = self.store_history[units_code]
        else:
            new_t_mat = np.array(distance[:,units])
            #t_mat = np.array(distance.iloc[:,units])
            new_pre_list = new_t_mat.argsort(axis=1)
            two_hc = Two_State_Hypercube({'Lambda':Lambda, 'Mu': Mu})
            two_hc.Update_Parameters(N = N, K = K, pre_list = new_pre_list, frac_j = frac_j, t_mat=new_t_mat)
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                two_hc.Larson_Approx()
                MRT,_=two_hc.Get_MRT_Approx()
            self.store_history[units_code] = MRT
        return MRT

    def genetic_algorithm(self, n, p, candi, K, Lambda, Mu, frac_j, distance):
        P,k = self.population_size(n, p)
        population = self.generate_population(n, p, P, candi)
        common_gene = list(set.intersection(*map(set, population)))  # genes appear in all population
        MRT_pop = [self.fitness_evaluation(p,K,Lambda,Mu,frac_j,units,distance) for units in population]
        print('population', population)
        print('MRT of population', MRT_pop)
        max_MRT, max_id = np.max(MRT_pop), np.argmax(MRT_pop)
        MaxIter = 0
        path = copy.deepcopy(MRT_pop)
        while MaxIter <= np.ceil(n*np.sqrt(p)):
            parents = random.sample(population,k=2)
            # print('parents', parents)
            new_member, MRT = self.new_species(parents, p, K, Lambda,Mu,frac_j, distance, common_gene)
            path.append(MRT)
            # print('test member and MRT', new_member, MRT)
            if MRT < max_MRT and new_member not in population:
                del population[max_id]
                del MRT_pop[max_id]
                population += [new_member]
                MRT_pop += [MRT]
                print('test population', population)
                # print('test MRT pop', MRT_pop)
                common_gene = list(set.intersection(*map(set, population))) 
                max_MRT, max_id = np.max(MRT_pop), np.argmax(MRT_pop)
                MaxIter = 0
            else:
                MaxIter += 1
            #path.append(np.min(MRT_pop))
        min_MRT, min_id= np.min(MRT_pop), np.argmin(MRT_pop)
        min_species = population[min_id]
        return min_species, min_MRT, path