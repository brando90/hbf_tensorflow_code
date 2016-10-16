import numpy as np
import pdb

import unittest
import namespaces as ns

import sklearn as sk
from sklearn.metrics.pairwise import euclidean_distances

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

import my_tf_pkg as mtf

def bt_mdl(arg,x,W,l,left,right):
    '''
    Returns a BT NN.

    generates the tree
    '''
    if l == 1:
        z = x[:,left:right] * W[l] # (M x F^(1)) = (M x D) x (D x F^(1))
        a = arg.act(z)  # (M x F^(1))
        return a # (M x F^(l)) = (M x F^(1))
    elif l == len(arg.F):
        return bt
    else:
        dif = int((right - left)/2)
        bt_left = bt_mdl(x, W, l-1, left, left+dif) # (M x F^(l-1))
        bt_right = bt_mdl(x, W, l-1, left+dif, right) # (M x F^(l-1))
        bt = bt_left + bt_right # (M x 2F^(l-1))
        #bt = bt * W[l] # (M x F^(l)) = (M x 2F^(l-1)) x (2F^(l-1) x F^(l))
        bt = tf.matmul(bt,W[l])
        return arg.act( bt )
    pass

def get_recursive_bt(arg,x,D):
    bt = bt_mdl(x,W=W,l=len(arg.F),left=0,right=D)
    # final layer (function approximation)
    C = mtf.get_W(init_W=arg.weights_initializer,l='Out_Layer',dims=[arg.F[len(arg.F)-1],1],dtype=tf.float32)
    bt = tf.matmul(A2,C)
    return bt

def bt_mdl_conv(arg,x):
    '''
    Returns a BT NN.
    '''
    # print('len(arg.F) ', len(arg.F))
    # print('range(1,len(arg.F) ', list(range(1,len(arg.F))) )
    # print('arg.F ', arg.F)
    # zeroth layer (the data)
    conv = x
    filter_width = 2 # 2 because of a binary tree
    stride_width = filter_width
    l=0
    if arg.verbose:
        print('------------------------------------------------------------------')
        print('--')
        print('l ', l)
        print('arg.F', arg.F)
        print('nb_filters ', arg.F[l])
        print('filter_width ', filter_width)
        print('stride_width ', stride_width)
        print(conv)
    # make each layer
    for l in range(1,len(arg.F)):
        nb_filters = arg.F[l] # nb of filters for current layer
        if arg.verbose:
            print('--')
            print('l ', l)
            print('arg.F', arg.F)
            print('nb_filters ', arg.F[l])
            print('filter_width ', filter_width)
            print('stride_width ', stride_width)
        #pdb.set_trace()
        conv = get_activated_conv_layer(arg=arg,x=conv,l=l,filter_width=filter_width,nb_filters=nb_filters,stride_width=stride_width,scope=arg.scope_name+str(l))
        if arg.verbose:
            print(conv)
        # setup for next iteration
        filter_width = 2*nb_filters
        stride_width = filter_width
    conv = tf.squeeze(conv)
    C = get_final_weight(arg=arg,shape=[arg.F[len(arg.F)-1],1],name='C_'+str(len(arg.F)))
    mdl = tf.matmul(conv,C)
    return mdl

def get_activated_conv_layer(arg,x,l,filter_width,nb_filters,stride_width,scope):
    '''
    arg = NS arg for params: act, normalization (BN) layer
    x = x-input (ahouls be previous activated conv layer)
    filter_width/kernel_size = kernel/filter width/shape, also means the width and height of the filters/kernel for convolution
    stride_width = stride for conv (not stride height is 1), note for BT should be equal to filter_shape
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
        trainable=arg.trainable,
        reuse=False
    )
    #flatten layer
    conv = flatten_conv_layer(x=conv,l=l,nb_filters=nb_filters,L=len(arg.F)-1)
    return conv

def flatten_conv_layer(x,l,nb_filters,L):
    '''
    l = which layer in the BT we want to flatten. This tells use how to flatten because it will tell us how many times we used the convolution in a layer.
        Equivalently, we apply one convolution locally at the image/vector depending on how many times the function is locally composed.
        So a binary tree in 4D the layer 1, has 4 = 2^3 - l = 2^2 locations where the function is locally shared.
    nb_filters = how many units each local convolutions has for this layer.
    L = number of total (conv) layers the BT network has (not including the approximation layer). Similarly, its just the index of the layer right before the approximation layer.
        For example, if we have a BT in 8D then we have 3 convolution layers and 1 approximation layer. Thus, we get that L = 3. For a 4D L=2, for 16D L = 4. etc.
        Note that the final conv layer is just a fully connected layer. i.e. at L = len(arg.F) - 1
    '''
    #conv = tf.reshape(conv, [-1,1,filter_width*nb_filters,1]) #flat
    nb_convolution_sections = 2**(L-l)
    conv = tf.reshape(x, [-1,1,nb_convolution_sections*nb_filters,1]) #flat
    return conv

def get_final_weight(arg,shape,name='C',dtype=tf.float32,regularizer=None,trainable=True):
    C = tf.get_variable(shape=shape,name=name,dtype=dtype,initializer=arg.weights_initializer,regularizer=regularizer,trainable=True)
    return C

##

class TestNN_BT(unittest.TestCase):
    #make sure methods start with word test

    def get_args(self,L,F,verbose=False,scope_name='BT'):
        '''
        L = layers
        F = list of nb of filters [F^(1),F^(2),...]
        '''
        arg = ns.Namespace(L=L,trainable=True,padding='VALID',scope_name=scope_name)
        #weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
        arg.weights_initializer = tf.constant_initializer(value=1.0, dtype=tf.float32)
        #biases_initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
        arg.biases_initializer = tf.constant_initializer(value=0.1, dtype=tf.float32)
        arg.normalizer_fn = None
        #arg.normalizer_fn = tf.contrib.layers.batch_norm
        arg.F = F
        arg.act = tf.nn.relu
        arg.verbose = verbose
        return arg

    def get_test_data(self,M,D):
        # do check
        M = 2
        X_data = np.arange(D*M).reshape((M, D))
        #print('X_data ', X_data)
        X_data = X_data.reshape(M,1,D,1)
        return X_data

    def get_hard_coded_bt4D(self,x):
        arg = ns.Namespace()
        #arg.init_type = 'xavier'
        arg.init_type = 'manual'
        arg.weights_initializer = tf.constant_initializer(value=1.0, dtype=tf.float32)
        arg.nb_filters = 3 #F1
        arg.nb_final_hidden_units = 5 # F2
        #arg.mu = [0.0,0.0,0.0]
        #arg.std = [0.9,0.9,0.9]
        #arg.mu = [None, None, 0.0]
        #arg.std = [None, None, 0.1]
        #arg.get_W_mu_init = lambda arg: arg.mu
        #arg.get_W_std_init = lambda arg: arg.std
        #arg.std_low, arg.std_high = 0.001, 0.1
        #arg.get_W_std_init = lambda arg: [float(i) for i in np.random.uniform(low=arg.std_low, high=arg.std_high, size=3)]
        arg.act = tf.nn.relu
        #arg.act = tf.nn.elu
        #arg.act = tf.nn.softplus
        #
        arg.stride_convd1, arg.filter_size = 2, 2 #fixed for Binary Tree BT
        #arg.mean, arg.stddev = arg.get_W_mu_init(arg), arg.get_W_std_init(arg)
        mdl = mtf.build_binary_tree_4D_hidden_layer(x,arg,phase_train=None)
        return mdl

    def test_NN_BT4D(self,M=2,D=4,F=[None,3,5],L=2):
        print('\n -------test'+str(D))
        #
        x = tf.placeholder(tf.float32, shape=[None,1,D,1], name='x-input') #[M, 1, D, 1]
        # prepare args
        arg = self.get_args(L=L,F=F,verbose=False,scope_name='BT_'+str(D)+'D')
        # get NN BT
        bt_mdl = bt_mdl_conv(arg,x)
        X_data = self.get_test_data(M,D)
        with tf.Session() as sess:
            sess.run( tf.initialize_all_variables() )
            bt_output = sess.run(fetches=bt_mdl, feed_dict={x:X_data})

        bt_hardcoded = self.get_hard_coded_bt4D(x)
        with tf.Session() as sess:
            sess.run( tf.initialize_all_variables() )
            bt_hardcoded_output = sess.run(fetches=bt_hardcoded, feed_dict={x:X_data})

        #print(bt_output)
        #print(bt_output.shape)

        #print(bt_hardcoded_output)
        #print(bt_hardcoded_output.shape)
        correct = np.array_equal(bt_output, bt_hardcoded_output)
        self.assertTrue(correct)

    # def test_NN_BT4D_single_batch(self,M=1,D=4,F=[None,3,5],L=2):
    #     print('\n -------test'+str(D))
    #     #
    #     x = tf.placeholder(tf.float32, shape=[None,1,D,1], name='x-input') #[M, 1, D, 1]
    #     # prepare args
    #     arg = self.get_args(L=L,F=F,verbose=False,scope_name='BT_'+str(D)+'D')
    #     # get NN BT
    #     bt_mdl = bt_mdl_conv(arg,x)
    #     X_data = self.get_test_data(M,D)
    #     with tf.Session() as sess:
    #         sess.run( tf.initialize_all_variables() )
    #         bt_output = sess.run(fetches=bt_mdl, feed_dict={x:X_data})
    #
    #     bt_hardcoded = self.get_hard_coded_bt4D(x)
    #     with tf.Session() as sess:
    #         sess.run( tf.initialize_all_variables() )
    #         bt_hardcoded_output = sess.run(fetches=bt_hardcoded, feed_dict={x:X_data})
    #
    #     #print(bt_output)
    #     #print(bt_output.shape)
    #
    #     #print(bt_hardcoded_output)
    #     #print(bt_hardcoded_output.shape)
    #     correct = np.array_equal(bt_output, bt_hardcoded_output)
    #     self.assertTrue(correct)
    #
    # def test_NN_BT8D(self,M=2,D=8,F=[None,3,5,7],L=3):
    #     print('\n -------test'+str(D))
    #     #
    #     x = tf.placeholder(tf.float32, shape=[None,1,D,1], name='x-input') #[M, 1, D, 1]
    #     # prepare args
    #     arg = self.get_args(L=L,F=F,verbose=False,scope_name='BT'+str(D)+'D')
    #     # get NN BT
    #     bt_mdl = bt_mdl_conv(arg,x)
    #     X_data = self.get_test_data(M,D)
    #     with tf.Session() as sess:
    #         sess.run( tf.initialize_all_variables() )
    #         bt_output = sess.run(fetches=bt_mdl, feed_dict={x:X_data})
    #
    #     #bt_hardcoded = self.get_hard_coded_bt4D(x)
    #     #with tf.Session() as sess:
    #     #    sess.run( tf.initialize_all_variables() )
    #     #    bt_hardcoded_output = sess.run(fetches=bt_hardcoded, feed_dict={x:X_data})
    #
    #     #print(bt_output)
    #     #print(bt_output.shape)
    #
    #     #print(bt_hardcoded_output)
    #     #print(bt_hardcoded_output.shape)
    #     #correct = np.array_equal(bt_output, bt_hardcoded_output)
    #     #self.assertTrue(correct)
    #
    # def test_NN_BT16D(self,M=2,D=16,F=[None,3,5,7,9],L=4):
    #     print('\n -------test'+str(D))
    #     #
    #     x = tf.placeholder(tf.float32, shape=[None,1,D,1], name='x-input') #[M, 1, D, 1]
    #     # prepare args
    #     arg = self.get_args(L=L,F=F,verbose=False,scope_name='BT'+str(D)+'D')
    #     # get NN BT
    #     bt_mdl = bt_mdl_conv(arg,x)
    #     X_data = self.get_test_data(M,D)
    #     with tf.Session() as sess:
    #         sess.run( tf.initialize_all_variables() )
    #         bt_output = sess.run(fetches=bt_mdl, feed_dict={x:X_data})
    #
    #     #bt_hardcoded = self.get_hard_coded_bt4D(x)
    #     #with tf.Session() as sess:
    #     #    sess.run( tf.initialize_all_variables() )
    #     #    bt_hardcoded_output = sess.run(fetches=bt_hardcoded, feed_dict={x:X_data})
    #
    #     #print(bt_output)
    #     #print(bt_output.shape)
    #
    #     #print(bt_hardcoded_output)
    #     #print(bt_hardcoded_output.shape)
    #     #correct = np.array_equal(bt_output, bt_hardcoded_output)
    #     #self.assertTrue(correct)

if __name__ == '__main__':
    unittest.main()
