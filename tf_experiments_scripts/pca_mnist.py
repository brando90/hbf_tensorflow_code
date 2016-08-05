import pdb

import numpy as np
from sklearn.decomposition import PCA
from numpy import linalg as LA

import my_tf_pkg as mtf

from keras.datasets import mnist

task_name='task_MNIST_flat_auto_encoder'
print '----====> TASK NAME: %s' % task_name
(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data(task_name)

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# print x_train.shape
# print x_test.shape
print np.sum(x_train)
print np.sum(X_train)+np.sum(X_cv)
print repr( np.sum(x_train) )
print repr( np.sum(X_train)+np.sum(X_cv) )
# print x_train[0][163]
print 'about to compute SVD'
U, s, V = np.linalg.svd(x_train)

pdb.set_trance()

def get_reconstruction(X_train,k):
    pca = PCA(n_components=k)
    pca = pca.fit(X_train)
    X_pca = pca.transform(X_train) # M_train x K
    #print 'X_pca' , X_pca.shape
    X_reconstruct = pca.inverse_transform(X_pca)
    return X_reconstruct, pca

number_units_list = np.linspace(start=10, stop=250, num=5)
#number_units_list = [10]
for k in number_units_list:
    ## Do PCA
    X_reconstruct_tf, _ = get_reconstruction(X_train,k)
    X_reconstruct_keras, _ = get_reconstruction(x_train,k)
    print '---'
    print 'tensorflow error: ',(1.0/X_train.shape[0])*LA.norm(X_reconstruct_tf - X_train)
    print 'keras error: ',(1.0/x_train.shape[0])*LA.norm(X_reconstruct_keras - x_train)
