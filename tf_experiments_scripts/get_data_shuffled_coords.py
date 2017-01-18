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

def shuffle_data_set_according_to_shuffle_array(X,arg):
    print(arg)
    print('shuffle_data_set_according_to_shuffle_array')
    N,D = X.shape
    for i in range(N):
        x_data = X[i,:]
        #print('before shuffle ', x_data)
        x_shuffled = np.zeros((1,4))
        x_shuffled[0,arg.shuffle_array] = x_data
        #print('x_shuffled ', x_shuffled)
        X[i,arg.shuffle_array] = x_shuffled
        #print('after shuffle ', X[i,:])
        #pdb.set_trace()
    return X

def shuffle_random(X,arg):
    print(arg)
    print('shuffle_random')
    N,D = X.shape
    for i in range(N):
        if arg.consistent_shuffle == '_consistent_shuffle_True':
            #print('seed')
            np.random.seed(arg.seed)
        x_data = X[i,:]
        #print('before shuffle ', x_data)
        X[i,:] = np.random.permutation(x_data)
        #print('after shuffle ', X[i,:])
        #pdb.set_trace()
    return X

def shuffle_data_set(X,arg):
    if arg.shuffle_array != None:
        X = shuffle_data_set_according_to_shuffle_array(X,arg)
    else:
        X = shuffle_random(X,arg)
    return X

def shuffle_data_set_unit_test():
    print('running unit test')
    x = np.arange(4)
    print('Before permutation: ', x)
    #
    print('Test the permutation is different from x and x is unchanged')
    np.random.seed(1)
    x_permutation = np.random.permutation(x)
    print('x_permutation: ', x_permutation)
    x_permutation2 = np.random.permutation(x)
    #
    print('x_permutation2 ', x_permutation2)
    print('Another permutation on the original x and its still different from x')
    #
    print('Testing that the permutation is the same as the first one')
    np.random.seed(1)
    print('new x_permutation: ', np.random.permutation(x))
    print('original x_permutation: ', x_permutation)
    #

def main():
    arg = ns.Namespace()
    arg.shuffle_array = None
    arg.data_dirpath = './data/'
    arg.data_file_name = 'f_4D_conv_2nd'
    arg.shuffle_array = [3, 2, 1, 0]
    #arg.data_file_name = 'f_8D_conv_cos_poly1_poly1'
    #arg.data_file_name = 'f_8D_conv_quad_cubic_sqrt'

    arg.consistent_shuffle = '_consistent_shuffle_True'
    arg.seed = 3
    X_train, Y_train, X_cv, Y_cv, X_test, Y_test = mtf.get_data(arg)
    print('\n -------> ', arg)
    print()

    print(type(X_train))
    print('X_train.shape: ', X_train.shape)
    print('X_train.shape: ', X_train.shape)
    print('X_train.shape: ', X_train.shape)

    X_train = shuffle_data_set(X_train,arg)
    X_cv = shuffle_data_set(X_cv,arg)
    X_test = shuffle_data_set(X_test,arg)

    folder_loc = './data/'+arg.data_file_name+'_shuffled'+arg.consistent_shuffle+'_array_shuffle_worst_case.npz'
    #folder_loc = './data/'+arg.data_file_name+'_shuffled'+arg.consistent_shuffle+'_seed_'+str(arg.seed)+'.npz'
    np.savez(folder_loc, X_train=X_train,Y_train=Y_train, X_cv=X_cv,Y_cv=Y_cv, X_test=X_test,Y_test=Y_test)

    print('max: ', np.max(Y_train))
    print('min: ', np.min(Y_train))

    print('mean: ',np.mean(Y_train))
    print('std: ',np.std(Y_train))

    print('--')

    print('Y_train.shape: ', Y_train.shape)
    print('Y_cv.shape: ', Y_cv.shape)
    print('Y_test.shape: ', Y_test.shape)

if __name__ == '__main__':
    #shuffle_data_set_unit_test()
    main()
