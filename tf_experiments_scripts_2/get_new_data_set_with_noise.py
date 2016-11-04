import os
import sys

import pickle
import namespaces as ns
import argparse
import pdb

import namespaces as ns

import numpy as np
import tensorflow as tf

import my_tf_pkg as mtf
import scipy

arg = ns.Namespace()
arg.data_dirpath = './data/'
#arg.data_file_name = 'f_4D_conv_2nd'
arg.data_file_name = 'f_8D_conv_cos_poly1_poly1'
X_train, Y_train, X_cv, Y_cv, X_test, Y_test = mtf.get_data(arg)

print('max: ', np.max(Y_train))
print('min: ', np.min(Y_train))

print('mean: ',np.mean(Y_train))
print('std: ',np.std(Y_train))

noise_level = 1.6723421882625
print('-- noise_level: ' , noise_level)

N, D_out = Y_train.shape
noise = np.random.normal(0,noise_level,(N, D_out))
Y_train = Y_train+noise

N, D_out = Y_cv.shape
noise = np.random.normal(0,noise_level,(N, D_out))
Y_cv = Y_cv+noise

N, D_out = Y_test.shape
noise = np.random.normal(0,noise_level,(N, D_out))
Y_test = Y_test+noise

# print( 'np.min(noise)', np.min(noise) )
# print( 'np.max(noise)', np.max(noise) )

folder_loc = './data/f_8D_conv_cos_poly1_poly1_noise_1_67_0125std.npz'
np.savez(folder_loc, X_train=X_train,Y_train=Y_train, X_cv=X_cv,Y_cv=Y_cv, X_test=X_test,Y_test=Y_test)

print('max: ', np.max(Y_train))
print('min: ', np.min(Y_train))

print('mean: ',np.mean(Y_train))
print('std: ',np.std(Y_train))

print('--')

print('Y_train.shape: ', Y_train.shape)
print('Y_cv.shape: ', Y_cv.shape)
print('Y_test.shape: ', Y_test.shape)
