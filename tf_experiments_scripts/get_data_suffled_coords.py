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

def shuffle_data_set(X):
    N,D = X.shape
    for i in range(N):
        x_data = X[i,:]
        x_data_suffled = np.random.shuffle(x_data)
        X[i,:] = x_data_suffled
    return X

arg = ns.Namespace()
arg.data_dirpath = './data/'
#arg.data_file_name = 'f_4D_conv_2nd'
#arg.data_file_name = 'f_8D_conv_cos_poly1_poly1'
arg.data_file_name = 'f_8D_conv_quad_cubic_sqrt'
X_train, Y_train, X_cv, Y_cv, X_test, Y_test = mtf.get_data(arg)

X_train = shuffle_data_set(X_train)
X_cv = shuffle_data_set(X_cv)
X_test = shuffle_data_set(X_test)

folder_loc = './data/'+arg.data_file_name+'_shuffled.npz'
np.savez(folder_loc, X_train=X_train,Y_train=Y_train, X_cv=X_cv,Y_cv=Y_cv, X_test=X_test,Y_test=Y_test)

print('max: ', np.max(Y_train))
print('min: ', np.min(Y_train))

print('mean: ',np.mean(Y_train))
print('std: ',np.std(Y_train))

print('--')

print('Y_train.shape: ', Y_train.shape)
print('Y_cv.shape: ', Y_cv.shape)
print('Y_test.shape: ', Y_test.shape)
