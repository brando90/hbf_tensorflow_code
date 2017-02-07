import numpy as np
import json
#from sklearn.cross_validation import train_test_split

from tensorflow.examples.tutorials.mnist import input_data

import pdb

def ReLu(x):
    return max(0,x)

def f_4D_simple_ReLu_BT_2_units_1st(A):
    h11 = lambda A: ReLu( (1.0*A[0] + 3.0*A[1] - 0.25) ) # [0,4]
    h12 = lambda A: ReLu( (1.0*A[0] - 1.0*A[1]) ) # [0,2]
    h13 = h12
    h14 = h12
    #
    h = [ h11(A[0:2]), h12(A[0:2]), h13(A[2:4]), h14(A[2:4]) ]
    h21 = lambda A: ReLu( 2.5*A[0] - 2*A[1] + 1.5*A[2] - 1.0*A[3] )
    h22 = lambda A: ReLu( -2.1*A[0] + 1.6*A[1] - 1.1*A[2] + 0.6*A[3] )
    f = 1.1*h21(h) + 1.5*h22(h)
    return f

def f_4D_simple_ReLu_BT(A):
    h11 = lambda A: ReLu( (1.0*A[0] + 3.0*A[1]) ) # [0,4]
    h12 = h11
    h21 = lambda A: ReLu( 1.3*A[0] - 1.5*A[1] )
    left, right = h11(A[0:2]), h12(A[2:4])
    f = h21( [left,right] )
    return f
#

def f_4D_cos_x2_BT(A):
    h11 = lambda A: np.cos(2*np.pi* (A[0]+A[1]) )
    h12 = h11
    h21 = lambda A: 2*(A[0] + A[1])**2 - 1
    left, right = h11(A[0:2]), h12(A[2:4])
    f = h21( [left,right] )
    return f

#

def f_4D_non_conv(A):
    h11 = lambda A: (1.0/20)*(1.0*A[0] + 2.0*A[1])**2.0
    h12 = lambda A: (1.0/10)*(3.0*A[0] + 4.0*A[1])**4.0
    h21 = lambda A: (1.0/2)*(5.0*A[0] + 6.0*A[1])**0.5
    left, right = h11(A[0:2]), h12(A[2:4])
    f = h21( [left,right] )
    return f

def f_4D_conv(A):
    h11 = lambda A: (5.0*A[0] + 9.0*A[1])**2.0
    h12 = h11
    h21 = lambda A: (0.9*A[0] + 0.7*A[1])**0.5
    left, right = h11(A[0:2]), h12(A[2:4])
    f = h21( [left,right] )
    return f

def f_4D_conv_2nd(A):
    h11 = lambda A: (2.0*A[0] + 3.0*A[1])**4.0
    h12 = h11
    h21 = lambda A: (4*A[0] + 5*A[1])**0.5
    left, right = h11(A[0:2]), h12(A[2:4])
    f = h21( [left,right] )
    return f

def f_4D_conv_test(A):
    h11 = lambda A: (2.0*A[0] + 3.0*A[1])**4.0
    h12 = h11
    h21 = lambda A: (4*A[0] + 5*A[1])**0.5
    left, right = h11(A[0:2]), h12(A[2:4])
    f = h21( [left,right] )
    return f

def get_labels_4D(X,f):
    N = X.shape[0] # N x D = N x 4
    Y = np.zeros( (N,1) )
    for i in range(N):
        Y[i] = f(X[i])
    return Y

def generate_data_4D(f, N_train=60000, N_cv=60000, N_test=60000, low_x=-1, high_x=1):
    D = 4
    # train
    X_train = low_x + (high_x - low_x) * np.random.rand(N_train,D)
    Y_train = get_labels_4D(X_train, f)
    # CV
    X_cv = low_x + (high_x - low_x) * np.random.rand(N_cv,D)
    Y_cv = get_labels_4D(X_cv, f)
    # test
    X_test = low_x + (high_x - low_x) * np.random.rand(N_test,D)
    Y_test = get_labels_4D(X_test, f)
    return (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)

def make_data_set_4D(f, folder_loc):
    X_train, Y_train, X_cv, Y_cv, X_test, Y_test = generate_data_4D(f)
    np.savez(folder_loc, X_train=X_train,Y_train=Y_train, X_cv=X_cv,Y_cv=Y_cv, X_test=X_test,Y_test=Y_test)
    return X_train, Y_train, X_cv, Y_cv, X_test, Y_test

def f_4D(A):
    h11 = lambda A: (1.0/20)*(1.0*A[0] + 2.0*A[1])**2.0
    h12 = lambda A: (1.0/10)*(3.0*A[0] + 4.0*A[1])**4.0
    h21 = lambda A: (1.0/2)*(5.0*A[0] + 6.0*A[1])**0.5
    left, right = h11(A[0:2]), h12(A[2:4])
    f = h21( [left,right] )
    return f

def get_labels_4D(X,f):
    N = X.shape[0] # N x D = N x 4
    Y = np.zeros( (N,1) )
    for i in range(N):
        Y[i] = f(X[i])
    return Y

def generate_data_4D(N_train=60000, N_cv=60000, N_test=60000, low_x=-1, high_x=1):
    D = 4
    f = f_4D
    # train
    X_train = low_x + (high_x - low_x) * np.random.rand(N_train,D)
    Y_train = get_labels_4D(X_train, f)
    # CV
    X_cv = low_x + (high_x - low_x) * np.random.rand(N_cv,D)
    Y_cv = get_labels_4D(X_cv, f)
    # test
    X_test = low_x + (high_x - low_x) * np.random.rand(N_test,D)
    Y_test = get_labels_4D(X_test, f)
    return (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)

def make_data_set():
    '''
    '''
    X_train, Y_train, X_cv, Y_cv, X_test, Y_test = generate_data_4D()
    file_name = 'f_4d_task.npz'
    np.savez(file_name, X_train=X_train,Y_train=Y_train, X_cv=X_cv,Y_cv=Y_cv, X_test=X_test,Y_test=Y_test)
    return X_train, Y_train, X_cv, Y_cv, X_test, Y_test
