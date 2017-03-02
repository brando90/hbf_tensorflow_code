import numpy as np
import json
#from sklearn.cross_validation import train_test_split

from tensorflow.examples.tutorials.mnist import input_data

import my_tf_pkg as mtf

import pdb

D = 8
params = []
f = mtf.get_product_function()
#pdb.set_trace()
file_name = './data/f_product_8D_continuous.npz'
N = 60000
N_train, N_cv, N_test = N, N, N
print('file_name ', file_name)
data = mtf.generate_and_save_data_set_general_D(file_name,f,D,params=params,N_train=N_train,N_cv=N_cv,N_test=N_test,low_x=-1.5,high_x=1.5)
X_train, Y_train, X_cv, Y_cv, X_test, Y_test = data

print('max: ', np.max(Y_train))
print('min: ', np.min(Y_train))

print('mean: ',np.mean(Y_train))
print('std: ',np.std(Y_train))
