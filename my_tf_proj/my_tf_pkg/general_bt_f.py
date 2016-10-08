import numpy as np
import unittest

import my_tf_pkg.f_4D_BT_data
import my_tf_pkg.f_8D_data

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

class TestF_BT(unittest.TestCase):
    #make sure methods start with word test

    def test_BT4D(self, D=4, N_train=100, low_x=-1, high_x=1):
        h1 = lambda A: (2.0*A[0] + 3.0*A[1])**4.0
        h2 = lambda A: (4*A[0] + 5*A[1])**0.5
        h_list = [None, h1, h2]
        ##
        f_general = lambda x: f_bt(x,h_list=h_list,l=2,left=0,right=D)
        f_hard_coded = f_4D_BT_data.f_4D_conv_2nd
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
        f_hard_coded = f_8D_data.f_8D_test
        #f_general([1,2,3,4,5,6,7,8])
        ## compare the functions
        X_train = low_x + (high_x - low_x) * np.random.rand(N_train,D)
        Y_train_general = get_labels_bt(X_train, f_general)
        Y_train_hard_coded  = get_labels_bt(X_train, f_hard_coded)
        correct = np.array_equal(Y_train_general, Y_train_hard_coded)
        self.assertTrue(correct)

if __name__ == '__main__':
    unittest.main()
