import os
import sys

import pickle
import namespaces as ns
import argparse
import pdb

import numpy as np
import tensorflow as tf

import my_tf_pkg as mtf
import scipy

task_name = 'task_f_4D_conv_2nd'
X_train, Y_train, X_cv, Y_cv, X_test, Y_test = mtf.get_data(task_name)

print('max: ', np.max(Y_train))
print('min: ', np.min(Y_train))

print('mean: ',np.mean(Y_train))
print('std: ',np.std(Y_train))

noise_level = 3
print('-- noise_level: ' , noise_level)

N, D = X_train.shape
noise = np.random.normal(0,noise_level,(N, D))
Y_train = Y_train+noise

N, D = X_cv.shape
noise = np.random.normal(0,noise_level,(N, D))
Y_cv = Y_cv+noise

N, D = X_test.shape
noise = np.random.normal(0,noise_level,(N, D))
Y_test = Y_test+noise

# print( 'np.min(noise)', np.min(noise) )
# print( 'np.max(noise)', np.max(noise) )

folder_loc = './data/f_4D_conv_2nd_noise_3_0_25std.npz'
np.savez(folder_loc, X_train=X_train,Y_train=Y_train, X_cv=X_cv,Y_cv=Y_cv, X_test=X_test,Y_test=Y_test)


print('max: ', np.max(Y_train))
print('min: ', np.min(Y_train))

print('mean: ',np.mean(Y_train))
print('std: ',np.std(Y_train))
