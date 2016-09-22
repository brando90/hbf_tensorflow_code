import numpy as np

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

def bt_mdl_conv(arg,x):
    '''
    Returns a BT NN.
    '''
    conv = x
    kernel_width = 2
    stride_width = kernel_width
    for l in range(arg.L-1,0,-1):
        nb_layer = arg.L - l
        print('l ', nb_layer)
        print('kernel_width ', kernel_width)
        print('stride_width ', stride_width)
        print(conv)
        conv = get_activated_conv_layer(nb_layer,arg,x=conv,kernel_size=[1,kernel_width],stride=[1,kernel_width],scope=arg.scope_name+str(nb_layer))
        # setup for next iteration
        kernel_width = 2*arg.F[l]
        stride_width = kernel_width
    return conv

def get_activated_conv_layer(l,arg,x,kernel_size,stride,scope):
    kernel_height, kernel_width = kernel_size
    stride_height, stride_width = stride
    conv = tf.contrib.layers.convolution2d(inputs=x,
        num_outputs=arg.F[l],
        kernel_size=[kernel_height, kernel_width],
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
    return conv

##

class TestNN_BT(unittest.TestCase):
    #make sure methods start with word test

    def get_args(self,L,F):
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
        arg = self.get_args(L=2,F=[3,5])
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

    def test_NN_BT8D(self):
        print('\n -------test')
        D = 8
        x = tf.placeholder(tf.float32, shape=[None,1,D,1], name='x-input') #[M, 1, D, 1]
        # prepare args
        arg = self.get_args(L=3,F=[3,5,7])
        arg.scope_name = 'BT8D'
        # get NN BT
        bt_mdl = bt_mdl_conv(arg,x)
        # do check
        M = 2
        X_data = np.array( [np.arange(0,9),np.arange(9,17)] )
        print('X_data ', X_data)
        X_data = X_data.reshape(M,1,D,1)
        with tf.Session() as sess:
            sess.run( tf.initialize_all_variables() )
            print('output: ', sess.run(fetches=bt_mdl, feed_dict={x:X_data}) )
        #self.assertTrue(correct)

if __name__ == '__main__':
    unittest.main()