import numpy as np
import json
from sklearn.cross_validation import train_test_split

from tensorflow.examples.tutorials.mnist import input_data

import pdb

def f_4D(A):
    h11 = lambda A: (1.0/20)*(1.0*A[1] + 2.0*A[2])**2.0
    h12 = lambda A: (1.0/10)*(3.0*A[1] + 4.0*A[2])**4.0
    h21 = lambda A: (1.0/100)*(5.0*A[1] + 6.0*A[2])**0.5
    left, right = h11(A[0:2]), h12(A[2:4])
    f = h21( [left,right] )
    return f

def get_labels(X,Y,f):
    N = X.shape[0] # N x D = N x 4
    Y = np.zeros( (N,1) )
    for i in range(N):
        Y[i] = f(X[i])
    return Y

def generate_data(D=1, N_train=60000, N_cv=60000, N_test=60000, low_x_var=-1, high_x_var=1):
    #
    f = f_4D(A)
    low_x, high_x = low_x_var, high_x_var
    # train
    X_train = low_x + (high_x - low_x) * np.random.rand(N_train,D)
    Y_train = get_labels(X_train, f)
    # CV
    X_cv = low_x + (high_x - low_x) * np.random.rand(N_cv,D)
    Y_cv = get_labels(X_cv, f)
    # test
    X_test = low_x + (high_x - low_x) * np.random.rand(N_test,D)
    Y_test = get_labels(X_test, np.zeros( (N_test,D) ), f)
    return (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)
