#!/usr/bin/env python
#SBATCH --mem=60000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brando90@mit.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00

import numpy as np
import json
#from sklearn.cross_validation import train_test_split

from tensorflow.examples.tutorials.mnist import input_data

import my_tf_pkg as mtf
import time

import pdb

logD = 5
D = 2**logD
f = np.prod
#
#nbins = 10
#M = 100
#M = 6*nbins**(2**logD)
#M = 6*10*(10**6)
M = 8*(10**4)
#M = 2**D
#M = int(9.5*10**6)
#
type_input_dist = 'full_random_M'
#type_input_dist = 'full_2^D_space'
file_name = '/home_simulation_research/hbf_tensorflow_code/tf_experiments_scripts/data/f_%sD_binary_parity_N%s.npz'%(str(D),str(M))
file_name = './data/f_%sD_binary_parity_N%s.npz'%(str(D),str(M))
print('D ', D,flush=True)
print('M ', M,flush=True)
print('->f ', file_name,flush=True)
#N_train, N_cv, N_test = N, N, N
print('file_name ', file_name)
start_time = time.time()
data = mtf.generate_and_save_data_set_general_D_binary(file_name,f,D,M, type_input_dist)
X_train, Y_train, X_cv, Y_cv, X_test, Y_test = data

print('max: ', np.max(Y_train))
print('min: ', np.min(Y_train))

print('mean: ',np.mean(Y_train))
print('std: ',np.std(Y_train))

seconds = (time.time() - start_time)
minutes = seconds/ 60
hours = minutes/ 60
print("--- %s seconds ---" % seconds )
print("--- %s minutes ---" % minutes )
print("--- %s hours ---" % hours )
