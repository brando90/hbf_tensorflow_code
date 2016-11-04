import numpy as np
import json
from sklearn.cross_validation import train_test_split

from tensorflow.examples.tutorials.mnist import input_data

import my_tf_pkg as mtf

import pdb

f = mtf.f_4D_simple_ReLu_BT_2_units_1st
folder_loc = './data/f_4D_simple_ReLu_BT_2_units_1st.npz'
X_train, Y_train, X_cv, Y_cv, X_test, Y_test = mtf.make_data_set_4D(f, folder_loc)

print('max: ', np.max(Y_train))
print('min: ', np.min(Y_train))

print('mean: ',np.mean(Y_train))
print('std: ',np.std(Y_train))
