import numpy as np
import unittest

import f_4D_BT_data

def get_labels_ut(X,f):
    N = X.shape[0] # N x D = N x 4
    Y = np.zeros( (N,1) )
    for i in range(N):
        Y[i] = f(X[i])
    return Y

def f(x, h_list, l, left, right):
    #print( left, right )
    if l == 1:
        h_l = h_list[l]
        return h_l(x[left:right])
    else:
        h_l = h_list[l]
        h_left, h_right = f(x, h_list, l-1, left, int(right/2)), f(x, h_list, l-1, int(right/2), right)
        return h_l( [h_left,h_right] )

class TestStringMethods(unittest.TestCase):
    #make sure methods start with word test

    def test_unit_test_BT4D(self, D=4, N_train=100, low_x=-1, high_x=1):
        h1 = lambda A: (2.0*A[0] + 3.0*A[1])**4.0
        h2 = lambda A: (4*A[0] + 5*A[1])**0.5
        h_list = [None, h1, h2]
        ##
        f_general = lambda x: f(x,h_list=h_list,l=2,left=0,right=4)
        f_hard_coded = f_4D_BT_data.f_4D_conv_2nd
        ## compare the functions
        X_train = low_x + (high_x - low_x) * np.random.rand(N_train,D)
        Y_train_general = get_labels_ut(X_train, f_general)
        Y_train_hard_coded  = get_labels_ut(X_train, f_hard_coded)
        correct = np.array_equal(Y_train_general, Y_train_hard_coded)
        self.assertEqual(correct, True)

if __name__ == '__main__':
    unittest.main()
