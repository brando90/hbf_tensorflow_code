import pdb

import numpy as np
from sklearn.decomposition import PCA
from numpy import linalg as LA

import my_tf_pkg as mtf

from keras.datasets import mnist

task_name='task_MNIST_flat_auto_encoder'
print '----====> TASK NAME: %s' % task_name
(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data(task_name)
data = (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)
#X_train =  np.vstack((X_train,X_cv))

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# print x_train.shape
# print x_test.shape
print np.sum(x_train)
print np.sum(X_train)
print repr( np.sum(x_train) )
print repr( np.sum(X_train) )
# print x_train[0][163]

def get_reconstruction(X_train,k):
    pca = PCA(n_components=k)
    pca = pca.fit(X_train)
    X_pca = pca.transform(X_train) # M_train x K
    #print 'X_pca' , X_pca.shape
    X_reconstruct = pca.inverse_transform(X_pca)
    #print dir(pca)
    return X_reconstruct, pca

#number_units_list = np.linspace(start=10, stop=250, num=5)
number_units_list = [12]
stddevs = [525.32626263]
nb_inits = 5
for k in number_units_list:
    mdl_best_params, mdl_mean_params, errors_best, errors_stats, reconstructions_best, reconstructions_mean = mtf.evalaute_models(data, stddevs, number_units_list, replace=False, nb_inits=nb_inits)
    Y_pred_train_best, Y_pred_cv_best, Y_pred_test_best = reconstructions_best
    #pdb.set_trace()
    ## Do PCA
    X_reconstruct_tf, _ = get_reconstruction(X_train,k)
    X_reconstruct_keras, pca = get_reconstruction(x_train,k)
    U = pca.components_
    print 'U_fingerprint', np.sum(U)
    X_my_reconstruct = np.dot(  U.T , np.dot(U, X_train.T) )
    print '---'
    N_train_tf = X_train.shape[0]
    N_train_keras = x_train.shape[0]
    print 'tensorflow error: ',(1.0/N_train_tf)*LA.norm(X_reconstruct_tf - X_train)
    print 'keras error: ',(1.0/N_train_keras)*LA.norm(X_reconstruct_keras - x_train)
    print 'U error: ',(1.0/X_train.shape[0])*LA.norm(X_reconstruct_tf - X_train)
    print 'RBF error: ',(1.0/X_train.shape[0])*LA.norm(np.squeeze(Y_pred_train_best) - X_train)
    #pdb.set_trace()
