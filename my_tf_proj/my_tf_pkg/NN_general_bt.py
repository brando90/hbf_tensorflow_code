import numpy as np

import sklearn as sk
from sklearn.metrics.pairwise import euclidean_distances

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

def bt_mdl(arg,x,W,l,left,right):
    '''
    Returns a BT NN.
    '''
    if l == 1:
        z = x[:,left:right] * W[l] # (M x F^(1)) = (M x D) x (D x F^(1))
        a = arg.act(z)  # (M x F^(1))
        return a # (M x F^(l)) = (M x F^(1))
    else:
        dif = int((right - left)/2)
        bt_left = bt_mdl(x, W, l-1, left, left+dif) # (M x F^(l-1))
        bt_right = bt_mdl(x, W, l-1, left+dif, right) # (M x F^(l-1))
        bt = bt_left + bt_right # (M x 2F^(l-1))
        bt = bt * W[l] # (M x F^(l)) = (M x 2F^(l-1)) x (2F^(l-1) x F^(l))
        return arg.act( bt )

def bt_mdl_conv(arg,x,F,L):
    '''
    Returns a BT NN.
    '''
    for range(l,0,-1):
        conv = tf.contrib.layers.convolution2d(inputs=x,num_outputs=F[l],kernel_size=[kernel_height, kernel_width],)
        x = arg.act(conv)

    return



##

class TestNN_BT(unittest.TestCase):
    #make sure methods start with word test

    def test_NN_BT4D(self):
        #self.assertTrue(correct)

    def test_NN_BT8D(self):
        #self.assertTrue(correct)

##
