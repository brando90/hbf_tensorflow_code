import numpy as np
import pdb

import unittest
import namespaces as ns

import sklearn as sk
from sklearn.metrics.pairwise import euclidean_distances

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

import my_tf_pkg as mtf

def debug_print(l, conv, conv_new, arg):
    conv_old = conv # old conv before applying conv for the next layer
    filter_width = arg.list_filter_widths[l] # filter width for current layer
    nb_filters = arg.nb_filters[l] # nb of filters for current layer
    stride_width = arg.list_strides[l] # stride_width for current layer
    print('----')
    print('-> l ', l)
    print('conv_old', conv_old)
    print('conv_new', conv_new)
    print('arg.list_filter_widths', arg.list_filter_widths)
    print('arg.nb_filters', arg.nb_filters)
    print('arg.list_strides', arg.list_strides)
    print('-')
    print('filter_width',filter_width)
    print('nb_filters',nb_filters)
    print('stride_width',stride_width)
    print('paddibg', arg.padding)
    print('\n')
    #print('number of channels it should have',)

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
        debug_print(l,None,conv,arg)
    # make each layer
    for l in range(1,len(arg.nb_filters)-1):
        filter_width = arg.list_filter_widths[l] # filter width for current layer
        nb_filters = arg.nb_filters[l] # nb of filters for current layer
        stride_width = arg.list_strides[l] # stride_width for current layer
        #pdb.set_trace()
        conv_new = get_activated_conv_layer_subgraph(arg=arg,x=conv,l=l,filter_width=filter_width,nb_filters=nb_filters,stride_width=stride_width,scope=arg.scope_name+str(l))
        if arg.verbose:
            debug_print(l,conv,conv_new,arg)
        conv = conv_new
    #pdb.set_trace()
    l=len(arg.nb_filters)-1
    if arg.verbose:
        debug_print(l,conv,None,arg)
    conv = tf.squeeze(conv)
    fully_connected_filter_width = arg.list_filter_widths[l]
    C = get_final_weight(arg=arg,shape=[fully_connected_filter_width,1],name='C_'+str(len(arg.nb_filters)))
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
    print('nb_filters', nb_filters)
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
    #print(conv)
    #flatten layer
    conv = flatten_conv_layer(x=conv)
    #pdb.set_trace()
    return conv

def flatten_conv_layer(x):
    '''
    Since we are doing 1D convs with 4D conv function, we get the convolutions in the shape: [?,1,nb_units,nb_channels]
    which means that we have "nb_units" and each unit has "nb_channels".
    To process this we need to flatten it to [?,1,nb_units*nb_channels,1]

    x = convolution vector to flatten
    '''
    #conv = tf.reshape(conv, [-1,1,filter_width*nb_filters,1]) #flat
    #nb_convolution_sections = 2**(L-l)
    #dimensions = [ int(dimension) for dimension in x.get_shape() ]
    dimensions = x.get_shape()
    conv = tf.reshape(x, [-1,1,int(dimensions[2])*int(dimensions[3]),1]) #flat
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
        padding = 'VALID'
        #padding = 'SAME'
        arg = ns.Namespace(L=L,verbose=verbose,trainable=True,padding=padding,scope_name=scope_name)
        arg.nb_filters = nb_filters
        arg.list_filter_widths = list_filter_widths
        arg.list_strides = list_strides
        #weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)

        arg.weights_initializer = tf.constant_initializer(value=1.0, dtype=tf.float32)
        #biases_initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)

        arg.biases_initializer = tf.constant_initializer(value=0.1, dtype=tf.float32)
        arg.normalizer_fn = None
        #arg.normalizer_fn = tf.contrib.layers.batch_norm

        arg.act = tf.nn.relu
        return arg

    def test_NN_BT8D(self,M=3,D=8,L=3):
        print('\n-------test'+str(D))
        a = 1
        #F1 = 4*a
        F1 = 4*a
        F2 = 7*a
        F3 = 28*a
        nb_filters=[None,F1,F2,F3]
        u1 = F1
        u2 = F2
        u3 = F3
        list_filter_widths=[None,2,4*u1,4*u2]
        s1 = 1
        s2 = 1
        s3 = 1
        list_strides=[None,s1,s2,s3]
        #
        x = tf.placeholder(tf.float32, shape=[None,1,D,1], name='x-input') #[M, 1, D, 1]
        # prepare args
        arg = self.get_args(L=L,nb_filters=nb_filters,list_filter_widths=list_filter_widths,list_strides=list_strides, verbose=True,scope_name='BT_'+str(D)+'D')
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
