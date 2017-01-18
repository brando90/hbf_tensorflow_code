import numpy as np
import json
from sklearn.cross_validation import train_test_split

from tensorflow.examples.tutorials.mnist import input_data

import my_tf_pkg as mtf
import namespaces as ns

import pdb

#task_name = 'task_MNIST_flat_auto_encoder'
#task_name = 'task_f_4D_conv_2nd'
arg = ns.Namespace()
arg.data_dirpath = './data/'
arg.data_file_name = 'f_8D_conv_cos_poly1_poly1'
X_train, Y_train, X_cv, Y_cv, X_test, Y_test = mtf.get_data(arg)

print('max: ', np.max(Y_train))
print('min: ', np.min(Y_train))

print('mean: ',np.mean(Y_train))
print('std: ',np.std(Y_train))
