import numpy as np
import json
from sklearn.cross_validation import train_test_split

import namespaces as ns

from tensorflow.examples.tutorials.mnist import input_data

import my_tf_pkg as mtf

import pdb

arg = ns.Namespace()

arg.data_dirpath = './data/'
#arg.data_file_name = 'MNIST_flat_auto_encoder'
#arg.data_file_name = 'f_4D_conv_2nd'
#arg.data_file_name = 'f_4D_conv_2nd'
arg.data_file_name = 'f_8D_conv_cos_poly1_poly1_shuffled'
arg.data_file_name = 'f_8D_conv_quad_cubic_sqrt_shuffled'
X_train, Y_train, X_cv, Y_cv, X_test, Y_test = mtf.get_data(arg)

print('\n-----> ', arg)
print('\n')

print(X_train)
print(X_cv)
print(X_test)

print('max: ', np.max(Y_train))
print('min: ', np.min(Y_train))

print('mean: ',np.mean(Y_train))
print('std: ',np.std(Y_train))
