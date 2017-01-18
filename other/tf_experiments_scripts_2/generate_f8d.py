import numpy as np
import json
from sklearn.cross_validation import train_test_split

from tensorflow.examples.tutorials.mnist import input_data

import my_tf_pkg as mtf

import pdb

f = mtf.f_8D_conv_quad_cubic_sqrt()
folder_loc = './data/f_8D_conv_quad_cubic_sqrt.npz'
#pdb.set_trace()
X_train, Y_train, X_cv, Y_cv, X_test, Y_test = mtf.make_data_set_8D(f, folder_loc)

print('max: ', np.max(Y_train))
print('min: ', np.min(Y_train))

print('mean: ',np.mean(Y_train))
print('std: ',np.std(Y_train))
