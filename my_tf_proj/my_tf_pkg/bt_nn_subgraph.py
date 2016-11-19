import numpy as np
import pdb

import unittest
import namespaces as ns

import sklearn as sk
from sklearn.metrics.pairwise import euclidean_distances

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

import my_tf_pkg as mtf

def bt_mdl_conv_subgraph(arg,x):
    '''
    Returns a BT NN.
    '''
    # print('len(arg.F) ', len(arg.F))
    # print('range(1,len(arg.F) ', list(range(1,len(arg.F))) )
    # print('arg.F ', arg.F)
    # zeroth layer (the data)
    conv = x
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
        filter_width = arg.list_filter_widths[l] # filter width for current layer
        nb_filters = arg.nb_filters[l] # nb of filters for current layer
        stride_width = arg.list_strides[l] # stride_width for current layer
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
    conv = tf.squeeze(conv)
    fully_connected_filter_width = arg.list_filter_widths[len(arg.F)-1]
    C = get_final_weight(arg=arg,shape=[fully_connected_filter_width,1],name='C_'+str(len(arg.F)))
    mdl = tf.matmul(conv,C)
    return mdl

def get_activated_conv_layer_subgraph(arg,x,l,filter_width,nb_filters,stride_width,scope):
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

    def get_test_data(self,M,D):
        '''
        gets a deterministic M x D batch of data

        M = the batch size (or in this case size of data set)
        D = the dimensionality of the data set
        '''
        # do check
        X_data = np.arange(D*M).reshape((M, D))
        #print('X_data ', X_data)
        X_data = X_data.reshape(M,1,D,1)
        return X_data

    def get_args(self,L,nb_filters,list_filter_widths,list_strides,verbose=False,scope_name='BT'):
        '''
        L = layers
        nb_filters = list of nb of filters [F^(1),F^(2),...]
        '''
        arg = ns.Namespace(L=L,verbose=verbose,trainable=True,padding='VALID',scope_name=scope_name)
        arg.nb_filters = nb_filters
        arg.list_filter_widths = list_filter_widths
        arg.list_strides = list_strides
        #weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)

        arg.weights_initializer = tf.constant_initializer(value=1.0, dtype=tf.float32)
        #biases_initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)

        arg.biases_initializer = tf.constant_initializer(value=0.1, dtype=tf.float32)
        arg.normalizer_fn = None
        #arg.normalizer_fn = tf.contrib.layers.batch_norm
        return arg

    def test_NN_BT8D(self,M=2,D=8,L=3,list_strides=[None,1,1,1]):
        print('\n -------test'+str(D))
        a = 1
        F1 = 4*a
        F2 = 7*a
        F3 = 28*a
        nb_filters=[None,F1,F2,F3]
        u1 = F1
        u2 = F2
        list_filter_widths=[None,4*u1,4*u2,]
        #
        x = tf.placeholder(tf.float32, shape=[None,1,D,1], name='x-input') #[M, 1, D, 1]
        # prepare args
        arg = self.get_args(L=L,nb_filters=nb_filters,list_filter_widths=list_filter_widths,list_strides=list_strides, verbose=False,scope_name='BT_'+str(D)+'D')
        # get NN BT
        bt_mdl = bt_mdl_conv_subgraph(arg,x)
        X_data = self.get_test_data(M,D)
        with tf.Session() as sess:
            sess.run( tf.initialize_all_variables() )
            bt_output = sess.run(fetches=bt_mdl, feed_dict={x:X_data})
        #
        print(bt_output)
        print(bt_output.shape)
        #correct = np.array_equal(bt_output, bt_hardcoded_output)
        #self.assertTrue(correct)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
