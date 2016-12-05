import numpy as np
import json
from sklearn.cross_validation import train_test_split
import unittest

from tensorflow.examples.tutorials.mnist import input_data

import pdb

import my_tf_pkg as mtf

def get_shallow_synthetic_func(D,nb_shallow_units,low_c=-2.0,high_c=2.0,low_w=-2.0,high_w=2.0,low_b=-1.0,high_b=1.0):
    '''
    Gets a shallow net function to generate data set.

    f(x) = \sum^(nb_shallow_units)_(i=1) c_i |Wx + b|_+
    '''
    N=3 #doesn't matter, let it be any value
    X = np.random.uniform(low=low_c, high=high_c, size=(N,D))
    params_for_units = get_shallow_params(X,nb_shallow_units)
    f = get_f_shallow_net(params_for_units)
    return f

def get_shallow_params(A,nb_shallow_units,low_c=-2.0,high_c=2.0,low_w=-2.0,high_w=2.0,low_b=-1.0,high_b=1.0):
    '''
    Creates the parameters for a shallow synthetic function f(x) = \sum^(nb_shallow_units)_(i=1) c_i |Wx + b|_+

    N - number of data examples
    D - input dimensionality of the data
    nb_shallow_units - number of (activation) units for the shallow network
    '''
    N,D = A.shape
    params_for_units = []
    for i in range(nb_shallow_units):
        rand_index = np.random.randint(low=0,high=N)
        x = A[rand_index,:] # 8D data array [1 x D]
        current_unit_val_x = 1
        #i = 0
        while current_unit_val_x != 0 :
            #print(i)
            #x = A[rand_index,:] # 8D data array [1 x D]
            c = np.random.uniform(low=low_c, high=high_c, size=1)
            w = np.random.uniform(low=low_w, high=high_w, size=(D,1))
            b = np.random.uniform(low=low_b, high=high_b, size=1)
            current_unit_val_x = c*ReLu(np.dot(x,w)+b)
            #i=i+1
        params_for_units.append( (c,w,b) )
    return params_for_units

def get_f_shallow_net(params_for_units):
    '''
    Gets a function handler/pointer that evaluates the shallow network f(x) = \sum^(nb_shallow_units)_(i=1) c_i |Wx + b|_+
    '''
    f8D = lambda x: _f_eval_shallow_net(x,params_for_units=params_for_units)
    return f8D

def _f_eval_shallow_net(A,params_for_units):
    '''
    Evaluates the actual shallow function f(x) = \sum^(nb_shallow_units)_(i=1) c_i |Wx + b|_+ given its parameters.

    A - data matrix [N , D] to evaluate
    params_for_units - parameters for each unit of the shallow network (i.e. [...,(c,w,b),...])
    '''
    # computes sum c|wx+b|_+
    c,w,b = params_for_units[0]
    f = c*ReLu(np.dot(A,w)+b) # [N,D]
    for i in range(1,len(params_for_units)):
        c,w,b = params_for_units[i]
        current_unit_val_x = c*ReLu(np.dot(A,w)+b) # [N,D] x [D,1] = [N,1]
        f += current_unit_val_x
    return f
#

def ReLu(x):
    return max(0,x)

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

def f_8D_conv_quad_cubic_sqrt():
    h1 = lambda A: 0.7*(1.0*A[0] + 2.0*A[1])**2
    h2 = lambda A: 0.6*(1.1*A[0] + 1.9*A[1])**3
    h3 = lambda A: 1.3*(1.2*A[0] + 1.3*A[1])**0.5
    h_list = [None, h1, h2, h3]
    ##
    D = 8
    f8D = lambda x: mtf.f_bt(x,h_list=h_list,l=3,left=0,right=D)
    return f8D

def f_8D_conv_cos_poly1_poly1():
    h1 = lambda A: (0.59)*np.cos( 1.5*np.pi*(A[0]+A[1])  )
    h2 = lambda A: 1.1*(A[0]+A[1]+1)**2 - 1
    h3 = lambda A: 1.1*(A[0]+A[1]+1)**2 - 1
    h_list = [None, h1, h2, h3]
    ##
    D = 8
    f8D = lambda x: mtf.f_bt(x,h_list=h_list,l=3,left=0,right=D)
    return f8D

def f_8D_single_relu():
    h1 = lambda A: ReLu( (3.0*A[0] + 2.0*A[1] + 1.0) ) # [0,4]
    h2 = lambda A: ReLu( (2.0*A[0] - 1.0*A[1] - 2.0) ) # [0,4]
    h3 = lambda A: ReLu( (-0.5*A[0] + 0.75*A[1] - 2.0) ) # [0,4]
    h_list = [None, h1, h2, h3]
    ##
    D = 8
    f8D = lambda x: mtf.f_bt(x,h_list=h_list,l=3,left=0,right=D)
    return f8D

##

def get_labels_8D(X,f):
    N = X.shape[0] # N x D = N x 4
    Y = np.zeros( (N,1) )
    for i in range(N):
        #pdb.set_trace()
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

#########

class Test_Shallow_Synthetic(unittest.TestCase):
    #make sure methods start with word test
    def test_shallow_network_8D(self, N=3, D=8, low_x=-1, high_x=1, nb_shallow_units=5):
        '''
        check that f(x) >= 0 since its positive linear combination of
        '''
        #X = np.arange(N*D).reshape((N,D))
        print('running test')
        #X = np.random.uniform(low=low_x, high=high_x, size=(N,D))
        X = low_x + (high_x - low_x) * np.random.rand(100,D)
        params_for_units = get_shallow_params(X,nb_shallow_units,low_c=0.0,high_c=5.0) # low_c is positive to make the unit test pass/make sense
        f = get_f_shallow_net(params_for_units)
        X_train, Y_train, X_cv, Y_cv, X_test, Y_test = generate_data_8D(f, N_train=N, N_cv=N, N_test=N, low_x=low_x, high_x=high_x)
        correct = []
        correct.append(np.all( Y_train >= 0))
        correct.append(np.all( Y_cv >= 0))
        correct.append(np.all( Y_test >= 0))
        self.assertTrue(np.all(correct))

if __name__ == '__main__':
    unittest.main()
