import numpy as np
import json
#from sklearn.cross_validation import train_test_split

from tensorflow.examples.tutorials.mnist import input_data

import my_tf_pkg as mtf
import time

import pdb

logD = 6
D = 2**logD
f = np.prod
#
nbins = 10
#M = 6*10
#M = 6*nbins**(2**logD)
M = 6*10*(10**6)
#
type_input_dist = 'full_random_M'
file_name = './data/f_product_%sD_binary_parity_N%s.npz'%(str(D),str(M))
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
