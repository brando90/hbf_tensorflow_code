import numpy as np
import json
#from sklearn.cross_validation import train_test_split

from tensorflow.examples.tutorials.mnist import input_data

import my_tf_pkg as mtf

import pdb

D = 64
f = np.prod
#
type_input_dist = 'full_random_M'
file_name = './data/f_product_%sD_binary.npz'%(str(D))
#
M = 60000
#N_train, N_cv, N_test = N, N, N
print('file_name ', file_name)
data = mtf.generate_and_save_data_set_general_D_binary(file_name,f,D,M, type_input_dist)
X_train, Y_train, X_cv, Y_cv, X_test, Y_test = data

print('max: ', np.max(Y_train))
print('min: ', np.min(Y_train))

print('mean: ',np.mean(Y_train))
print('std: ',np.std(Y_train))
