import numpy as np
import json
from sklearn.cross_validation import train_test_split

from tensorflow.examples.tutorials.mnist import input_data

import pdb

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

def f_4D_conv_3rd(A):
    h11 = lambda A: (2.0* np.cos( 4.2 * np.pi * (A[0]+A[1]) )  )**2.0 + (2*A[0]+3*A[1])**4.0
    h12 = h11
    h21 = lambda A: (4*A[0] + 5*A[1])**0.5
    left, right = h11(A[0:2]), h12(A[2:4])
    f = h21( [left,right] )
    return f

def f_4D_conv_4th(A):
    h11 = lambda A: (2.0* np.cos( 4.2 * np.pi * (A[0]+A[1]) )  )**2.0 + (2*A[0]+3*A[1])**4.0
    h12 = h11
    h21 = lambda A: (4*A[0] + 5*A[1])**1.0
    left, right = h11(A[0:2]), h12(A[2:4])
    f = h21( [left,right] )
    return f

def f_4D_conv_5th(A):
    h11 = lambda A: (1.1*A[0]+1.5*A[1])**4.0
    h12 = h11
    h21 = lambda A: 2*A[0]**2.0 + 3*A[1]**2.0 + 0.5*A[0]*A[1]
    left, right = h11(A[0:2]), h12(A[2:4])
    f = h21( [left,right] )
    return f

def f_4D_conv_6th(A):
    h11 = lambda A: (1.3* np.cos( 4.2 * np.pi * (A[0]+A[1]) )  )**2.0 + (1.2*A[0]+2.1*A[1])**4.0
    h12 = h11
    h21 = lambda A: (1.1*A[0] + 2.1*A[1])**1.0
    left, right = h11(A[0:2]), h12(A[2:4])
    f = h21( [left,right] )
    return f

##

def f_4D_conv_changing(A):
    h11 = lambda A: (2.0* np.cos( 4.2 * np.pi * (A[0]+A[1]) )  )**4.0
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

def make_data_set_4D(f, file_name):
    X_train, Y_train, X_cv, Y_cv, X_test, Y_test = generate_data_4D(f)
    np.savez(file_name, X_train=X_train,Y_train=Y_train, X_cv=X_cv,Y_cv=Y_cv, X_test=X_test,Y_test=Y_test)
    return X_train, Y_train, X_cv, Y_cv, X_test, Y_test
