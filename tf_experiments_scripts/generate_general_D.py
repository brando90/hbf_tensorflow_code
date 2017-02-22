import numpy as np
import json
from sklearn.cross_validation import train_test_split

from tensorflow.examples.tutorials.mnist import input_data

import my_tf_pkg as mtf

import pdb

logD = 5
D = 2**logD
f,h_list,params = mtf.get_ppt_function(L=logD)
file_name = './data/f_32D_ppt.npz'
N = 60000
N_train, N_cv, N_test = N, N, N
print('file_name ', file_name)
data = mtf.generate_and_save_data_set_general_D(file_name,f,D,params=params,N_train=N_train,N_cv=N_cv,N_test=N_test,low_x=-1,high_x=1)
X_train, Y_train, X_cv, Y_cv, X_test, Y_test = data

print('max: ', np.max(Y_train))
print('min: ', np.min(Y_train))

print('mean: ',np.mean(Y_train))
print('std: ',np.std(Y_train))
