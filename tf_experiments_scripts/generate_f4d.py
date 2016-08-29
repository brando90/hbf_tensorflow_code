import numpy as np
import json
from sklearn.cross_validation import train_test_split

from tensorflow.examples.tutorials.mnist import input_data

import my_tf_pkg as mtf

import pdb

f = mtf.f_4D_conv_test
file_name = 'f_4d_task_conv_test.npz'
X_train, Y_train, X_cv, Y_cv, X_test, Y_test = mtf.make_data_set_4D(f, file_name)

print 'max: ', np.max(Y_train)
print 'min: ', np.min(Y_train)

print 'mean: ',np.mean(Y_train)
print 'std: ',np.std(Y_train)
