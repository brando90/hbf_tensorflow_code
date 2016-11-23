import numpy as np
import pdb

import unittest
import namespaces as ns

import sklearn as sk
from sklearn.metrics.pairwise import euclidean_distances

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

import my_tf_pkg as mtf

def count_number_trainable_params(graph):
    '''
    Counts the number of trainable variables in the given graph.

    graph = tensorflow graph with the parameters to count.
    '''
    tot_nb_params = 0
    with graph.as_default():
        for trainable_variable in tf.trainable_variables():
            #print('trainable_variable ', trainable_variable.__dict__)
            shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
            current_nb_params = get_nb_params_shape(shape)
            tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params

def get_nb_params_shape(shape):
    '''
    Computes the total number of params for a given shap.
    Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
    '''
    nb_params = 1
    for dim in shape:
        nb_params = nb_params*int(dim)
    return nb_params

##

def debug_print(l, conv, conv_new, arg):
    conv_old = conv # old conv before applying conv for the next layer
    filter_width = arg.list_filter_widths[l] # filter width for current layer
    nb_filters = arg.nb_filters[l] # nb of filters for current layer
    stride_width = arg.list_strides[l] # stride_width for current layer
    print('----')
    print('-> l ', l)
    print('conv_old l = %d: '%(l-1), conv_old)
    print('conv_new l = %d: '%(l), conv_new)
    # print('-')
    # print('arg.list_filter_widths', arg.list_filter_widths)
    # print('arg.nb_filters', arg.nb_filters)
    # print('arg.list_strides', arg.list_strides)
    # print('-')
    # print('filter_width',filter_width)
    # print('nb_filters',nb_filters)
    # print('stride_width',stride_width)
    # print('paddibg', arg.padding)
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
        print(arg)
        debug_print(l,None,conv,arg)
    # make each layer
    for l in range(1,len(arg.nb_filters)):
        print('>>loop index ', l)
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
    conv = tf.squeeze(conv)
    fully_connected_filter_width = arg.list_filter_widths[l]
    C = get_final_weight(arg=arg,shape=[fully_connected_filter_width,1],name='C_'+str(len(arg.nb_filters)))
    mdl = tf.matmul(conv,C)
    if arg.verbose:
        debug_print(l,conv,mdl,arg)
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
    # if arg.verbose:
    #     print('==> conv_raw', conv)
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

    def check_variables(self,sg_bt_mdl, bt_mdl):
        with tf.variable_scope('SG', reuse=True):
            print(sg_bt_mdl)
            pdb.set_trace()
            xx = tf.get_variable('MatMul:0')
        #
        pdb.set_trace()
        correct =False
        return correct

    def check_count_number_trainable_params(self,graph_sg_bt,graph_bt):
        nb_params_sg_bt = count_number_trainable_params(graph_sg_bt)
        print('count_number_trainable_params (SG_BT) ', count_number_trainable_params(graph_sg_bt))
        nb_params_bt = count_number_trainable_params(graph_bt)
        print('count_number_trainable_params (BT) ', count_number_trainable_params(graph_bt))
        correct = (nb_params_bt == nb_params_sg_bt)
        return correct

    def get_args_standard_bt(self,L,F,verbose=False,scope_name='BT'):
        '''
        L = layers
        F = list of nb of filters [F^(1),F^(2),...]
        '''
        padding = 'VALID'
        #padding = 'SAME'
        arg = ns.Namespace(L=L,verbose=verbose,trainable=True,padding=padding,scope_name=scope_name)
        arg.F = F
        ##
        #weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
        arg.weights_initializer = tf.constant_initializer(value=1.0, dtype=tf.float32)

        #biases_initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
        arg.biases_initializer = tf.constant_initializer(value=0.1, dtype=tf.float32)

        arg.normalizer_fn = None
        #arg.normalizer_fn = tf.contrib.layers.batch_norm

        arg.act = tf.nn.relu
        return arg

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

        ##
        #weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
        arg.weights_initializer = tf.constant_initializer(value=1.0, dtype=tf.float32)

        #biases_initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
        arg.biases_initializer = tf.constant_initializer(value=0.1, dtype=tf.float32)

        arg.normalizer_fn = None
        #arg.normalizer_fn = tf.contrib.layers.batch_norm

        arg.act = tf.nn.relu
        return arg

    # def test_NN_BT8D(self,M=3,D=8,L=3):
    #     print('\n-------test'+str(D))
    #     a = 1
    #     # nb of filters per unit
    #     #F1 = 4*a
    #     # F1 = 4*a
    #     # F2 = 7*a
    #     # F3 = 28*a
    #     #F1, F2, F3 = 4*a, 7*a, 28*a
    #     #F1, F2, F3 = a, 2*a, 6*a
    #     #F1, F2, F3 = 2*a, 3*a, 12*a
    #     F1, F2, F3 = a, 2*a, 4*a
    #     nb_filters=[None,F1,F2,F3]
    #     # width of filters
    #     # u1 = F1
    #     # u2 = F2
    #     # u3 = F3
    #     u1, u2, u3 = F1, F2, F3
    #     #w1, w2, w3 = 2,4*u1,4*u2
    #     #w1, w2, w3 = 3,2*u1,3*u2
    #     #w1, w2, w3 = 3,3*u1,4*u2
    #     w1, w2, w3 = 2,2*u1,2*u2
    #     list_filter_widths=[None,w1,w2,w3]
    #     # stride
    #     # s1 = 1
    #     # s2 = 1*F1
    #     # s3 = 1
    #     #s1, s2, s3 = 1, 1*F1, 1
    #     #s1, s2, s3 = 1, 2*F1, 1
    #     #s1, s2, s3 = 1, 1*F1, 1
    #     s1, s2, s3 = 2, 2*F1, 1
    #     list_strides=[None,s1,s2,s3]
    #     #
    #     x = tf.placeholder(tf.float32, shape=[None,1,D,1], name='x-input') #[M, 1, D, 1]
    #     # prepare args
    #     arg = self.get_args(L=L,nb_filters=nb_filters,list_filter_widths=list_filter_widths,list_strides=list_strides, verbose=True,scope_name='BT_'+str(D)+'D')
    #     # get NN BT
    #     bt_mdl = bt_mdl_conv_subgraph(arg,x)
    #     X_data = self.get_test_data(M,D)
    #     with tf.Session() as sess:
    #         sess.run( tf.initialize_all_variables() )
    #         bt_output = sess.run(fetches=bt_mdl, feed_dict={x:X_data})
    #     #
    #     print('Output of mdl on a data set that is M,D = %d, %d'%(M,D))
    #     print('note the output should b M,1 = %d, %d'%(M,1))
    #     print('bt_output')
    #     print(bt_output)
    #     print('bt_output.shape ', bt_output.shape)
    #     #correct = np.array_equal(bt_output, bt_hardcoded_output)
    #     #self.assertTrue(correct)
    #     print('count_number_trainable_params ', count_number_trainable_params(tf.get_default_graph()))
    #     self.assertTrue(True)

    def test_NN_BT8D_vs_BT8D_other_lib(self,M=3,D=8,L=3):
        '''
        Test aims to compare the old hard coded BT library with the subgraph one when
        both are suppose to output the exact same model architecture. In this sense
        to check if the code is correct the BT formed from the subgraph function
        should match
        '''
        print('\n-------test'+str(D))
        a = 3
        # nb of filters per unit
        F1, F2, F3 = a, 2*a, 4*a
        nb_filters=[None,F1,F2,F3]
        # width of filters
        u1, u2, u3 = F1, F2, F3
        w1, w2, w3 = 2,2*u1,2*u2
        list_filter_widths=[None,w1,w2,w3]
        # stride
        s1, s2, s3 = 2, 2*F1, 1
        list_strides=[None,s1,s2,s3]
        #
        # prepare args
        arg_sg_bt = self.get_args(L=L,nb_filters=nb_filters,list_filter_widths=list_filter_widths,list_strides=list_strides, verbose=True,scope_name='SG_BT_'+str(D)+'D')
        arg_bt = self.get_args_standard_bt(L=L,F=nb_filters,verbose=True,scope_name='Standard_BT_'+str(D)+'D')
        # get NN mdls
        #pdb.set_trace()
        print('------SG')
        graph_sg_bt = tf.Graph()
        with graph_sg_bt.as_default():
            x_sg_bt = tf.placeholder(tf.float32, shape=[None,1,D,1], name='x-input') #[M, 1, D, 1]
            with tf.variable_scope('SG_BT'):
                sg_bt_mdl = bt_mdl_conv_subgraph(arg_sg_bt,x_sg_bt)
                print('------BT')
        graph_bt = tf.Graph()
        with graph_bt.as_default():
            x_bt = tf.placeholder(tf.float32, shape=[None,1,D,1], name='x-input') #[M, 1, D, 1]
            with tf.variable_scope('BT'):
                bt_mdl = mtf.bt_mdl_conv(arg_bt,x_bt)
        #
        X_data = self.get_test_data(M,D)
        #
        with tf.Session(graph=graph_sg_bt) as sess:
            sess.run( tf.initialize_all_variables() )
            sg_bt_output = sess.run(fetches=sg_bt_mdl, feed_dict={x_sg_bt:X_data})
        with tf.Session(graph=graph_bt) as sess:
            sess.run( tf.initialize_all_variables() )
            bt_output = sess.run(fetches=bt_mdl, feed_dict={x_bt:X_data})
        #
        #correct = self.check_variables(sg_bt_mdl, bt_mdl)
        #self.assertTrue(correct)
        #should have same number of params
        correct = self.check_count_number_trainable_params(graph_sg_bt,graph_bt)
        #self.assertTrue(correct)
        # should output the same on the smae data set and same params
        print('sg_bt_output')
        print(sg_bt_output)
        print('bt_output')
        print(bt_output)
        correct = np.array_equal(bt_output, sg_bt_output)
        self.assertTrue(correct)

if __name__ == '__main__':
    unittest.main()
