# Author: Ricardo Baptista and Matthias Poloczek
# Date:   June 2018
#
# See LICENSE.md for copyright information
#

import random
import numpy as np
from itertools import combinations


def binary_converter(ten_list, n_vars):
	binary_list = np.zeros(n_vars)
	for i in ten_list:
		binary_list[i] = 1
	return binary_list

def sample_models(n_models, n_vars):
	# SAMPLE_MODELS: Function samples the binary models to
	# generate observations to train the statistical model

	# Generate matrix of zeros with ones along diagonals
	binary_models = np.zeros((n_models, n_vars))

	# Sample model indices
	model_num = np.random.randint(2**n_vars, size=n_models)

	strformat = '{0:0' + str(n_vars) + 'b}'
	# Construct each binary model vector
	for i in range(n_models):
		model = strformat.format(model_num[i])
		binary_models[i,:] = np.array([int(b) for b in model])

	print(binary_models)

	return binary_models

'''
def sample_models_constraints(n_models, n_vars, sigma):
	binary_models = np.zeros((n_models, n_vars))
	for j in range(n_models):
		index_list = [i for i in range(n_vars)]
		comb = list(combinations(index_list, sigma))
		length = len(comb)
		index = np.random.randint(length)
		for k in range(len(binary_converter(comb[index], n_vars))):
			binary_models[j,k] = binary_converter(comb[index], n_vars)[k]

	print(binary_models)

	return binary_models
'''

def sample_models_constraints(n_models, n_vars, sigma):
	binary_models = np.zeros((n_models, n_vars))
	for j in range(n_models):
		index_list = [i for i in range(n_vars)]
		sample_list = random.sample(index_list, sigma)
		for k in range(len(binary_converter(sample_list, n_vars))):
			binary_models[j,k] = binary_converter(sample_list, n_vars)[k]
	return binary_models


def get_all_combinations(n_vars, sigma):
	# SAMPLE_MODELS: Function samples the binary models to
	# generate observations to train the statistical model

	# Generate matrix of zeros with ones along diagonals
	index_list = [i for i in range(n_vars)]
	comb = list(combinations(index_list, sigma))
	length = len(comb)
	binary_models = np.zeros((length, n_vars))

	for j in range(length):
		index = j
		for k in range(len(binary_converter(comb[index], n_vars))):
			binary_models[j,k] = binary_converter(comb[index], n_vars)[k]

	print(binary_models)

	return binary_models


def sample_with_constraints(old_x):
	# other methods: continuation
	index_1 = []
	index_2 = []
	new_x = old_x.copy()
	for i in range(len(new_x[0])):
		if new_x[0, i] == 0:
			index_1.append(i)
		else:
			index_2.append(i)
	flip_bit_1 = np.random.randint(len(index_1))
	flip_bit_2 = np.random.randint(len(index_2))

	new_x[0, index_1[flip_bit_1]] = 1. - new_x[0, index_1[flip_bit_1]]
	new_x[0, index_2[flip_bit_2]] = 1. - new_x[0, index_2[flip_bit_2]]

	return new_x


def sample_initial_points(n_init, n_vars, fix_num):
	initial_points = np.zeros((n_init, n_vars), dtype=int)
	unique_points = set()
	i = 0
	while i < n_init:
		point = np.zeros(n_vars, dtype=int)
		point[:fix_num] = 1
		np.random.shuffle(point)
		point_tuple = tuple(point)

		if point_tuple not in unique_points:
			unique_points.add(point_tuple)
			initial_points[i] = point
			i += 1

	return initial_points

# -- END OF FILE --