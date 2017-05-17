#!/usr/bin/env python
#SBATCH --mem=60000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brando90@mit.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00

import numpy as np
import json
import time
#from sklearn.cross_validation import train_test_split

from tensorflow.examples.tutorials.mnist import input_data

import my_tf_pkg as mtf

import pdb

logD = 5
D = 2**logD
f = np.prod
params = []

N = 80000*16*2
N_train, N_cv, N_test = N, N, N

file_name = './data/f_%sD_product_continuous_%s.npz'%(D,N)
#file_name = '/home_simulation_research/hbf_tensorflow_code/tf_experiments_scripts/data/f_32D_product_continuous_80000'
print('file_name ', file_name)
start_time = time.time()
data = mtf.generate_and_save_data_set_general_D(file_name,f,D,params=params,N_train=N_train,N_cv=N_cv,N_test=N_test,low_x=-1.5,high_x=1.5)
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
