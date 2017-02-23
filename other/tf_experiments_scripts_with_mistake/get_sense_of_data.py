import numpy as np
import json
#from sklearn.cross_validation import train_test_split

from tensorflow.examples.tutorials.mnist import input_data

import my_tf_pkg as mtf

import pdb

#task_name = 'task_MNIST_flat_auto_encoder'
#task_name = 'task_f_4D_conv_2nd'
task_name = 'task_f_8D_conv_cos_poly1_poly1'
X_train, Y_train, X_cv, Y_cv, X_test, Y_test = mtf.get_data(task_name)

print('max: ', np.max(Y_train))
print('min: ', np.min(Y_train))

print('mean: ',np.mean(Y_train))
print('std: ',np.std(Y_train))
