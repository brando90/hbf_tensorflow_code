import numpy as np
import json
from sklearn.cross_validation import train_test_split

from tensorflow.examples.tutorials.mnist import input_data

import pdb

def f_8D(A):
    h11 = lambda A: (1.0/20)*(1*A[0] + 2*A[1])**4
    h12 = lambda A: (1.0/2)*(3*A[0] + 4*A[1])**3
    h21 = lambda A: (1.0/50)*(5*A[0] + 6*A[1])**2

    h13 = lambda A: (1.0/20)*(1*A[0] + 2*A[1])**4
    h14 = lambda A: (1.0/2)*(3*A[0] + 4*A[1])**3
    h22 = lambda A: (1.0/50)*(5*A[0] + 6*A[1])**2

    h31 = lambda A: (1.1/1)*(A[0] + (1/100)*A[1] + 1)**0.5;

    H11, H12 = h11(A[0:2]), h12(A[2:4])
    H21 = h21([H11, H12])
    H13, H14 = h13(A[4:6]), h14(A[6:8])
    H22 = h22([H13, H14])
    H31 = h31([H21, H22])
    return H31

def f_8D_conv_test(A):
    h11 = lambda A: (1.0/20)*(1*A[0] + 2*A[1])**4
    h12 = h11
    h13 = h11
    h14 = h11

    h21 = lambda A: (1.0/50)*(5*A[0] + 6*A[1])**2
    h22 = h21

    h31 = lambda A: (1.1/1)*(A[0] + (1/100)*A[1] + 1)**0.5;

    H11, H12 = h11(A[0:2]), h12(A[2:4])
    H21 = h21([H11, H12])
    H13, H14 = h13(A[4:6]), h14(A[6:8])
    H22 = h22([H13, H14])
    H31 = h31([H21, H22])
    return H31

##

def get_labels_8D(X,f):
    N = X.shape[0] # N x D = N x 4
    Y = np.zeros( (N,1) )
    for i in range(N):
        Y[i] = f(X[i])
    return Y

def generate_data_8D(f, N_train=60000, N_cv=60000, N_test=60000, low_x=-1, high_x=1):
    D = 8
    # train
    X_train = low_x + (high_x - low_x) * np.random.rand(N_train,D)
    Y_train = get_labels_8D(X_train, f)
    # CV
    X_cv = low_x + (high_x - low_x) * np.random.rand(N_cv,D)
    Y_cv = get_labels_8D(X_cv, f)
    # test
    X_test = low_x + (high_x - low_x) * np.random.rand(N_test,D)
    Y_test = get_labels_8D(X_test, f)
    return (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)

def make_data_set_8D(f, file_name):
    X_train, Y_train, X_cv, Y_cv, X_test, Y_test = generate_data_8D(f)
    np.savez(file_name, X_train=X_train,Y_train=Y_train, X_cv=X_cv,Y_cv=Y_cv, X_test=X_test,Y_test=Y_test)
    return X_train, Y_train, X_cv, Y_cv, X_test, Y_test
