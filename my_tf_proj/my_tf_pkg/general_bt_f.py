import numpy as np
import unittest

import pdb

import my_tf_pkg.f_4D_data as f_4D_data
import my_tf_pkg.f_8D_data as f_8D_data

def get_labels_bt(X,f):
    N = X.shape[0] # N x D = N x 4
    Y = np.zeros( (N,1) )
    for i in range(N):
        Y[i] = f(X[i])
    return Y

def f_bt(x, h_list, l, left, right):
    '''
        computes a binary tree function
        x = is a vector in 2,4,8,...2^K dimension
        h_list = [None, h1, h2, ...] first is empty cuz its the input data
        left = pointers to the section the current subtree cares about
        right = pointers to the section the current subtree cares about
    '''
    # print('--')
    # print( x )
    # print( 'l ', l )
    # print( left, right )
    if l == 1:
        h_l = h_list[l]
        #print(x[left:right])
        return h_l(x[left:right])
    else:
        h_l = h_list[l]
        dif = int((right - left)/2)
        h_left, h_right = f_bt(x, h_list, l-1, left, left+dif), f_bt(x, h_list, l-1, left+dif, right)
        return h_l( [h_left,h_right] )

##

def get_list_of_functions_per_layer_ppt(L):
    '''
    get a list of synthetic functions according to ppt = poly, poly, trig.

    for example, for L = 4:
        h_list = [None, poly1, poly2, cos1, poly3]
    '''
    h_list = [None]
    params = [None]
    for l in range(1,L+1):
        if l % 3 == 0:
            #trig
            a = np.random.uniform(low=0.01, high=2.0, size=None)
            freq = np.random.uniform(low=0.9, high=1.7, size=None)
            trig = lambda x: a*np.cos( freq*np.pi*( x[0] + x[1] ) )
            h_list.append(trig)
            params.append([a,freq,'a*np.cos( freq*np.pi*( x[0] + x[1] ) )'])
        else:
            #poly
            degree = np.random.randint(2, high=4, size=None)
            coeff = np.random.uniform(low=0.01, high=1.3, size=2)
            poly = lambda x: (coeff[0]*x[0]+coeff[1]*x[1])**degree
            h_list.append(poly)
            params.append([degree,coeff,'(coeff[0]*x[0]+coeff[1]*x[1])**degree'])
    return h_list, params

def get_ppt_function(L):
    '''
    gets the function according to ppt structure and the function list that
    corresponds to each layer and the parameters for those function per layer.
    '''
    logD = L
    D = 2**logD
    h_list, params = get_list_of_functions_per_layer_ppt(L=logD)
    f_general = lambda x: f_bt(x,h_list=h_list,l=logD,left=0,right=D)
    return f_general, h_list, params

##

def get_poly():
    '''
    check if this is a good idea to put at each node
    '''
    degree = np.random.randint(2, high=4, size=None, dtype='l')
    coeff = np.random.uniform(low=0.01, high=3.0, size=degree)
    poly = lambda x: numpy.polyval(coeff, x[0]+x[1])
    return poly

##

def get_function_each_layer_relu():
    '''
    gives you an array of functions/handlers for evaluating nodes in a BT network.
    '''

    return

def get_single_node_f():
    '''

    '''
    f8D = lambda x: _f_eval_shallow_net(x,params_for_units=params_for_units)
    return f8D

def _f_btclear_eval_node():
    '''
    evaluate a single node in a binary tree. i.e. compute h_l(x,y) for some
    h_l(.,.) parametrixed
    '''
    return

##

def generate_and_save_data_set_general_D(file_name,f,D, params=None,N_train=60000, N_cv=60000, N_test=60000, low_x=-1, high_x=1):
    '''
    Generates a data set according to file
    note: it also returns the data set to inspect it if desired.
    '''
    X_train, Y_train, X_cv, Y_cv, X_test, Y_test = _generate_data_general_D(f,D, params,N_train, N_cv, N_test, low_x, high_x)
    D = np.array(D)
    params = np.array([]) if params == None else params
    np.savez(file_name, file_name,file_name, params=params,D=D,X_train=X_train,Y_train=Y_train, X_cv=X_cv,Y_cv=Y_cv, X_test=X_test,Y_test=Y_test)
    return X_train, Y_train, X_cv, Y_cv, X_test, Y_test

def _generate_data_general_D(f,D,params=None,N_train=60000, N_cv=60000, N_test=60000, low_x=-1, high_x=1):
    # train
    X_train = low_x + (high_x - low_x) * np.random.rand(N_train,D)
    Y_train = _get_labels_general_D(X_train, f)
    # CV
    X_cv = low_x + (high_x - low_x) * np.random.rand(N_cv,D)
    Y_cv = _get_labels_general_D(X_cv, f)
    # test
    X_test = low_x + (high_x - low_x) * np.random.rand(N_test,D)
    Y_test = _get_labels_general_D(X_test, f)
    return (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)

def _get_labels_general_D(X,f):
    N = X.shape[0] # N x D = N x 4
    Y = np.zeros( (N,1) )
    for i in range(N):
        Y[i] = f(X[i])
    return Y
#

class TestF_BT(unittest.TestCase):
    #make sure methods start with word test

    def test_BT4D(self, D=4, N_train=100, low_x=-1, high_x=1):
        h1 = lambda A: (2.0*A[0] + 3.0*A[1])**4.0
        h2 = lambda A: (4*A[0] + 5*A[1])**0.5
        h_list = [None, h1, h2]
        ##
        f_general = lambda x: f_bt(x,h_list=h_list,l=2,left=0,right=D)
        f_hard_coded = f_4D_data.f_4D_conv_2nd
        #f_general([1,2,3,4])
        ## compare the functions
        X_train = low_x + (high_x - low_x) * np.random.rand(N_train,D)
        Y_train_general = get_labels_bt(X_train, f_general)
        Y_train_hard_coded  = get_labels_bt(X_train, f_hard_coded)
        correct = np.array_equal(Y_train_general, Y_train_hard_coded)
        self.assertTrue(correct)

    def test_BT8D(self, D=8, N_train=100, low_x=-1, high_x=1):
        h1 = lambda A: (1.0/20)*(1*A[0] + 2*A[1])**4
        h2 = lambda A: (1.0/50)*(5*A[0] + 6*A[1])**2
        h3 = lambda A: (1.1/1)*(A[0] + (1/100)*A[1] + 1)**0.5;
        h_list = [None, h1, h2, h3]
        ##
        f_general = lambda x: f_bt(x,h_list=h_list,l=3,left=0,right=D)
        f_hard_coded = f_8D_data.f_8D_conv_test
        #f_general([1,2,3,4,5,6,7,8])
        ## compare the functions
        X_train = low_x + (high_x - low_x) * np.random.rand(N_train,D)
        Y_train_general = get_labels_bt(X_train, f_general)
        Y_train_hard_coded  = get_labels_bt(X_train, f_hard_coded)
        correct = np.array_equal(Y_train_general, Y_train_hard_coded)
        self.assertTrue(correct)

    def test_BT256D(self, logD=8, N_train=100, low_x=-1, high_x=1):
        D = 2**logD
        h_list, params = get_list_of_functions_per_layer_ppt(L=logD)
        ##
        f_general = lambda x: f_bt(x,h_list=h_list,l=logD,left=0,right=D)
        ## compare the functions
        X_train = low_x + (high_x - low_x) * np.random.rand(N_train,D)
        Y_train_general = get_labels_bt(X_train, f_general)
        correct = Y_train_general
        self.assertIsNotNone(correct)

if __name__ == '__main__':
    unittest.main()
