import numpy as np
import unittest
from numpy.testing import assert_array_equal

#import sklearn as sk # TODO do I need this?
#from sklearn.metrics.pairwise import euclidean_distances

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

import pdb

def hello_world_hbf():
    print( "Hello World from HBF!" )

# def build_HBF(x, dims, inits, phase_train=None, trainable=True):
#     (_,inits_W,inits_S) = inits
#     layer = x
#     nb_hidden_layers = len(dims)-1
#     for l in range(1,nb_hidden_layers): # from 1 to L-1
#         layer = get_HBF_layer(l=str(l),x=layer,init=(inits_W[l],inits_S[l]),dims=(dims[l-1],dims[l]),phase_train=phase_train)
#     return layer

def tri(a, m, s, x, t):
    b = 1.0 / s**2.0
    return a * tf.maximum(0.0, m - b * l1_norm(x, t))

def l1_norm(x, t):
    n, d = x.shape
    return tf.norm(x.reshape((n, 1, d)) - t, ord=1, axis=2)

def build_HBF2(x, dims, inits, phase_train=None, trainable_bn=True,trainable_S=True):
    (_,inits_W,inits_S) = inits
    layer = x
    nb_hidden_layers = len(dims)-1
    for l in range(1,nb_hidden_layers): # from 1 to L-1
        #print nb_hidden_layers
        #print len(inits_W)
        layer = get_HBF_layer2(l=str(l),x=layer,init=(inits_W[l],inits_S[l]),dims=(dims[l-1],dims[l]),phase_train=phase_train, trainable_bn=trainable_bn,trainable_S=trainable_S)
        #layer = get_HBF_layer3(l=str(l),x=layer,init=(inits_W[l],inits_S[l]),dims=(dims[l-1],dims[l]),phase_train=phase_train, trainable_bn=trainable_bn,trainable_S=trainable_S)
    return layer

def get_HBF_layer2(l, x, dims, init, phase_train=None, layer_name='HBFLayer', trainable_bn=True, trainable_S=True):
    (init_W,init_S) = init
    with tf.name_scope(layer_name+l):
        with tf.name_scope('templates'+l):
            #W = tf.get_variable(name='W'+l, dtype=tf.float64, initializer=init_W, regularizer=None, trainable=True)
            W = get_W(init_W, l, dims)
        with tf.name_scope('rbf_stddev'+l):
            print( '--> init_S: ', init_S)
            print('--> trainable_S: ', trainable_S)
            S = tf.get_variable(name='S'+l, dtype=tf.float64, initializer=init_S, regularizer=None, trainable=trainable_S)
            beta = tf.pow(tf.div( tf.constant(1.0,dtype=tf.float64),S), 2)
        with tf.name_scope('Z'+l):
            WW =  tf.reduce_sum(W*W, reduction_indices=0, keep_dims=True) # (1 x D^(l)) = sum( (D^(l-1) x D^(l)), 0 )
            XX =  tf.reduce_sum(x*x, reduction_indices=1, keep_dims=True) # (M x 1) = sum( (M x D^(l-1)), 1 )
            # -|| x - w ||^2 = -(-2<x,w> + ||x||^2 + ||w||^2) = 2<x,w> - (||x||^2 + ||w||^2)
            Delta_tilde = 2.0*tf.matmul(x,W) - tf.add(WW, XX) # (M x D^(l)) - (M x D^(l)) = (M x D^(l-1)) * (D^(l-1) x D^(l)) - (M x D^(l))
            Z = beta * ( Delta_tilde ) # (M x D^(l))
        # if phase_train is not None:
        #     Z = add_batch_norm_layer(l, Z , phase_train, trainable_bn=trainable_bn)
        #     Z = tf.abs(Z)
            #A = tf.exp(-Z) # (M x D^(l))
        with tf.name_scope('A'+l):
            A = tf.exp(Z) # (M x D^(l))
        # if phase_train is not None:
        #     A = add_batch_norm_layer(l, A , phase_train, trainable_bn=trainable_bn)
            #z = add_batch_norm_layer(l, z, phase_train, trainable_bn=trainable_bn)
    var_prefix = 'vars_'+layer_name+l
    put_summaries(var=W,prefix_name=var_prefix+W.name,suffix_text=W.name)
    put_summaries(var=S,prefix_name=var_prefix+S.name,suffix_text=S.name)
    act_stats = 'acts_'+layer_name+l
    put_summaries(Z,prefix_name=act_stats+'Z'+l,suffix_text='Z'+l)
    put_summaries(A,prefix_name=act_stats+'A'+l,suffix_text='A'+l)
    put_summaries(Delta_tilde,prefix_name=act_stats+'Delta_tilde'+l,suffix_text='Delta_tilde'+l)
    put_summaries(beta,prefix_name=act_stats+'beta'+l,suffix_text='beta'+l)
    return A

def get_HBF_layer3(l, x, dims, init, phase_train=None, layer_name='HBFLayer', trainable_bn=True, trainable_S=True):
    (init_W,init_S) = init
    with tf.name_scope(layer_name+l):
        with tf.name_scope('templates'+l):
            #W = tf.get_variable(name='W'+l, dtype=tf.float64, initializer=init_W, regularizer=None, trainable=True)
            W = get_W(init_W, l, dims)
        with tf.name_scope('rbf_stddev'+l):
            print( '--> init_S: ', init_S)
            print( '--> trainable_S: ', trainable_S)
            S = tf.get_variable(name='S'+l, dtype=tf.float64, initializer=init_S, regularizer=None, trainable=trainable_S)
            beta = tf.pow(tf.div( tf.constant(1.0,dtype=tf.float64),S), 2)
        with tf.name_scope('Z'+l):
            WW =  tf.reduce_sum(W*W, reduction_indices=0, keep_dims=True) # (1 x D^(l)) = sum( (D^(l-1) x D^(l)), 0 )
            XX =  tf.reduce_sum(x*x, reduction_indices=1, keep_dims=True) # (M x 1) = sum( (M x D^(l-1)), 1 )
            # -|| x - w ||^2 = -(-2<x,w> + ||x||^2 + ||w||^2) = 2<x,w> - (||x||^2 + ||w||^2)
            Delta_tilde = 2.0*tf.matmul(x,W) - tf.add(WW, XX) # (M x D^(l)) - (M x D^(l)) = (M x D^(l-1)) * (D^(l-1) x D^(l)) - (M x D^(l))

            #Delta_tilde = tf.Print(Delta_tilde,[Delta_tilde], message="my Delta_tilde-values:",first_n=10)

            Z = beta * ( Delta_tilde ) # (M x D^(l))
            #Z = tf.Print(Z,[Z], message="Z:",first_n=10)
        if phase_train is not None:
            Z = add_batch_norm_layer(l, Z , phase_train, trainable_bn=trainable_bn)
            #Z = tf.Print(Z,[Z], message="Z:",first_n=10)
        with tf.name_scope('A'+l):
            Y = tf.square(Z)
            #Y = tf.Print(Y,[Y], message="Z:",first_n=10)
            #
            init_a = tf.constant(2.0,dtype=tf.float64)
            a = tf.get_variable(name='a'+l, dtype=tf.float64, initializer=init_a, regularizer=None, trainable=True)
            precision = tf.pow(tf.div( tf.constant(1.0,dtype=tf.float64),a), 2)
            precision = 1.0
            #
            A = tf.exp(-precision*Y) # (M x D^(l))
    var_prefix = 'vars_'+layer_name+l
    put_summaries(var=W,prefix_name=var_prefix+W.name,suffix_text=W.name)
    put_summaries(var=S,prefix_name=var_prefix+S.name,suffix_text=S.name)
    act_stats = 'acts_'+layer_name+l
    put_summaries(Z,prefix_name=act_stats+'Z'+l,suffix_text='Z'+l)
    put_summaries(Y,prefix_name=act_stats+'Y'+l,suffix_text='Y'+l)
    put_summaries(A,prefix_name=act_stats+'A'+l,suffix_text='A'+l)
    put_summaries(Delta_tilde,prefix_name=act_stats+'Delta_tilde'+l,suffix_text='Delta_tilde'+l)
    put_summaries(beta,prefix_name=act_stats+'beta'+l,suffix_text='beta'+l)
    return A

def build_summed_HBF(x, dims, inits, phase_train=None):
    (inits_C,inits_W,inits_S) = inits
    layer = x
    nb_hidden_layers = len(dims)-1
    for l in range(1,nb_hidden_layers): # from 1 to L-1
        layer = get_HBF_layer(l=str(l),x=layer,init=(inits_W[l],inits_S[l]),dims=(dims[l-1],dims[l]),phase_train=phase_train)
        layer = get_summation_layer(str(l),layer,inits_C[l])
    return layer

class TriTest(unittest.TestCase):
    sess = tf.InteractiveSession()

    def test_norm_one_point_one_center(self):
        x = np.array([[1, 2, 3]]).astype(np.float32)
        t = np.array([[4, 3, 1]]).astype(np.float32)
        norm = l1_norm(x, t).eval()
        exp = 6
        assert_array_equal(norm, exp)

    def test_norm_one_point_three_centers(self):
        x = np.array([[1, 2, 3]]).astype(np.float32)
        t = np.array([[4, 3, 1], [2, 5, 9], [9, 3, 1]]).astype(np.float32)
        norm = l1_norm(x, t).eval()
        exp = np.array([[6, 10, 11]]).astype(np.float32)
        assert_array_equal(norm, exp)

    def test_norm_two_points_one_center(self):
        x = np.array([[5, 1, 2], [3, 1, 7]]).astype(np.float32)
        t = np.array([[1, 3, 1]]).astype(np.float32)
        norm = l1_norm(x, t).eval()
        exp = np.array([[7],[10]]).astype(np.float32)
        assert_array_equal(norm, exp)

    def test_norm_three_points_two_centers(self):
        x = np.array([[1, 2, 3],[2, 5, 9], [9, 3, 1]]).astype(np.float32)
        t = np.array([[4, 3, 1],[5, 6, 2]]).astype(np.float32)
        norm = l1_norm(x, t).eval()
        exp = np.array([[6, 9], [12, 11], [5, 8]]).astype(np.float32)
        assert_array_equal(norm, exp)

    def test_tri_1d_multiple_points_centers(self):
        a = 2; m = 1; s = 2
        x = np.array([[3.5],[0.2]]).astype(np.float32)
        t = np.array([[0],[3]]).astype(np.float32)
        print (tri(a, m, s, x, t).eval())

if __name__ == '__main__':
    unittest.main()
