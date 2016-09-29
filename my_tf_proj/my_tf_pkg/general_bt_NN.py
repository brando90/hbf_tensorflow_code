import numpy as np
import pdb

import unittest
import namespaces as ns

import sklearn as sk
from sklearn.metrics.pairwise import euclidean_distances

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

def bt_mdl(arg,x,W,l,left,right):
    '''
    Returns a BT NN.
    '''
    # TODO: tramsfpr,pseudocode to real code:
    # if l == 1:
    #     z = x[:,left:right] * W[l] # (M x F^(1)) = (M x D) x (D x F^(1))
    #     a = arg.act(z)  # (M x F^(1))
    #     return a # (M x F^(l)) = (M x F^(1))
    # else:
    #     dif = int((right - left)/2)
    #     bt_left = bt_mdl(x, W, l-1, left, left+dif) # (M x F^(l-1))
    #     bt_right = bt_mdl(x, W, l-1, left+dif, right) # (M x F^(l-1))
    #     bt = bt_left + bt_right # (M x 2F^(l-1))
    #     bt = bt * W[l] # (M x F^(l)) = (M x 2F^(l-1)) x (2F^(l-1) x F^(l))
    #     return arg.act( bt )
    pass

def bt_mdl_conv(arg,x):
    '''
    Returns a BT NN.
    '''
    print('len(arg.F) ', len(arg.F))
    print('range(1,len(arg.F) ', list(range(1,len(arg.F))) )
    print('arg.F ', arg.F)
    # zeroth layer (the data)
    conv = x
    filter_width = 2 # 2 because of a binary tree
    stride_width = filter_width
    l=0
    print('--')
    print('l ', l)
    print('arg.F', arg.F)
    print('nb_filters ', arg.F[l])
    print(conv)
    # make each layer
    for l in range(1,len(arg.F)):
        nb_filters = arg.F[l] # nb of filters for current layer
        print('--')
        print('l ', l)
        print('arg.F', arg.F)
        print('nb_filters ', arg.F[l])
        # print('nb_filters ', nb_filters)
        # print('filter_width ', filter_width)
        # print('stride_width ', stride_width)
        #pdb.set_trace()
        conv = get_activated_conv_layer(arg=arg,x=conv,filter_width=filter_width,nb_filters=nb_filters,stride_width=stride_width,scope=arg.scope_name+str(l))
        print(conv)
        # setup for next iteration
        filter_width = 2*nb_filters
        stride_width = filter_width
    print('----')
    return conv

def get_activated_conv_layer(arg,x,filter_width,nb_filters,stride_width,scope):
    '''
    arg = NS arg for params: act, normalization (BN) layer
    x = x-input (ahouls be previous activated conv layer)
    filter_shape/kernel_size = kernel/filter size/shape, also means the width and height of the filters/kernel for convolution
    stride_width = stride for conv (not stride height is 1)
    scope = scope (name)

    NOTE: height is 1 because data are vectors
    '''
    filter_height, filter_width = [1, filter_width] # filter_height=1 cuz convolutions to vectors
    stride_height, stride_width = [1, stride_width] # stride_height=1 cuz convolutions to vectors
    conv = tf.contrib.layers.convolution2d(inputs=x,
        num_outputs=nb_filters,
        kernel_size=[filter_height, filter_width], # [1, filter_width]
        stride=[stride_height, stride_width],
        padding=arg.padding,
        rate=1,
        activation_fn=arg.act,
        normalizer_fn=arg.normalizer_fn,
        normalizer_params=None,
        weights_initializer=arg.weights_initializer,
        biases_initializer=arg.biases_initializer,
        scope=scope,
        trainable=True,
        reuse=False
    )
    conv = tf.reshape(conv, [-1,1,filter_width*nb_filters,1]) #flat
    return conv

##

class TestNN_BT(unittest.TestCase):
    #make sure methods start with word test

    def get_args(self,L,F):
        '''
        L = layers
        F = list of nb of filters [F^(1),F^(2),...]
        '''
        arg = ns.Namespace(L=L,padding='VALID')
        arg.act = tf.nn.relu
        #weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
        arg.weights_initializer = tf.constant_initializer(value=1.0, dtype=tf.float32)
        #biases_initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
        arg.biases_initializer = tf.constant_initializer(value=0.1, dtype=tf.float32)
        arg.normalizer_fn = None
        #arg.normalizer_fn = tf.contrib.layers.batch_norm
        arg.F = F
        return arg

    def test_NN_BT4D(self):
        print('\n -------test')
        D = 4
        x = tf.placeholder(tf.float32, shape=[None,1,D,1], name='x-input') #[M, 1, D, 1]
        # prepare args
        arg = self.get_args(L=2,F=[None,3,5])
        arg.scope_name = 'BT4D'
        # get NN BT
        bt_mdl = bt_mdl_conv(arg,x)
        # do check
        M = 2
        X_data = np.array( [np.arange(0,4),np.arange(4,8)] )
        print('X_data ', X_data)
        X_data = X_data.reshape(M,1,D,1)
        with tf.Session() as sess:
            sess.run( tf.initialize_all_variables() )
            print('output: ', sess.run(fetches=bt_mdl, feed_dict={x:X_data}) )
        #self.assertTrue(correct)

    # def test_NN_BT8D(self):
    #     print('\n -------test')
    #     D = 8
    #     x = tf.placeholder(tf.float32, shape=[None,1,D,1], name='x-input') #[M, 1, D, 1]
    #     # prepare args
    #     arg = self.get_args(L=3,F=[3,5,7])
    #     arg.scope_name = 'BT8D'
    #     # get NN BT
    #     bt_mdl = bt_mdl_conv(arg,x)
    #     # do check
    #     M = 2
    #     X_data = np.array( [np.arange(0,9),np.arange(9,17)] )
    #     print('X_data ', X_data)
    #     X_data = X_data.reshape(M,1,D,1)
    #     with tf.Session() as sess:
    #         sess.run( tf.initialize_all_variables() )
    #         print('output: ', sess.run(fetches=bt_mdl, feed_dict={x:X_data}) )
    #     #self.assertTrue(correct)

if __name__ == '__main__':
    unittest.main()
