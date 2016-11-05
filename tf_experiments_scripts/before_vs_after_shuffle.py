import numpy as np
import json
from sklearn.cross_validation import train_test_split

import namespaces as ns

from tensorflow.examples.tutorials.mnist import input_data

import my_tf_pkg as mtf

import pdb

arg = ns.Namespace()

arg.data_dirpath = './data/'
print('\ng-----> ', arg)
print('\n')
arg.data_file_name = 'f_8D_conv_quad_cubic_sqrt'
X_train, Y_train, X_cv, Y_cv, X_test, Y_test = mtf.get_data(arg)
arg.data_file_name = 'f_8D_conv_quad_cubic_sqrt_shuffled'
X_train_shuff, Y_train, X_cv_shuff, Y_cv, X_test_shuff, Y_test = mtf.get_data(arg)

print('before shuffle')
print(X_train)
# print(X_cv)
# print(X_test)

print('after shuffled')

print(X_train_shuff)
# print(X_cv_shuff)
# print(X_test_shuff)
