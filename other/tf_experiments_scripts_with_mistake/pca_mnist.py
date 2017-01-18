import pdb

import tensorflow as tf
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from numpy import linalg as LA

import my_tf_pkg as mtf

from keras.datasets import mnist

task_name='task_MNIST_flat_auto_encoder'
print '----====> TASK NAME: %s' % task_name
(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data(task_name)
data = (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)
#X_train =  np.vstack((X_train,X_cv))

# (x_train, _), (x_test, _) = mnist.load_data()
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# print x_train.shape
# print x_test.shape
#print 'MNIST keras fingerprint: ', np.sum(x_train)
#print 'MNIST keras fingerprint: ', np.sum(X_train)+np.sum(X_train)

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
nb_inits = 1
for k in number_units_list:
    mdl_best_params, mdl_mean_params, errors_best, errors_stats, reconstructions_best, reconstructions_mean = mtf.evalaute_models(data, stddevs, number_units_list, replace=False, nb_inits=nb_inits)
    Y_pred_train_best, Y_pred_cv_best, Y_pred_test_best = reconstructions_best
    Y_pred_train_best = np.squeeze(Y_pred_train_best)
    ## do sklearn PCA
    #X_reconstruct_keras, pca = get_reconstruction(x_train,k)
    #X_reconstruct_tf, _ = get_reconstruction(X_train,k)
    pca = PCA(n_components=k)
    pca = pca.fit(X_train)
    X_pca = pca.transform(X_train)
    X_reconstruct = pca.inverse_transform(X_pca)
    # do manual PCA
    U = pca.components_
    #print 'U_fingerprint', np.sum(np.absolute(U))
    X_my_reconstruct = np.dot(  U.T , np.dot(U, X_train.T) ).T
    print '--- PCA Errors'
    N_train_tf = X_train.shape[0]
    #N_train_keras = x_train.shape[0]
    #print 'keras error: ',(1.0/N_train_keras)*LA.norm(X_reconstruct_keras - x_train)**2
    print 'MSE error: ', sklearn.metrics.mean_squared_error(X_reconstruct, X_train)
    print 'X_recon - X_my_reconstruct', (1.0/X_train.shape[0])*LA.norm(X_my_reconstruct - X_reconstruct)**2
    print 'U.T*U*X  error: ',(1.0/X_train.shape[0])*LA.norm(X_my_reconstruct - X_train)**2
    print 'RBF error: ',(1.0/X_train.shape[0])*LA.norm( Y_pred_train_best - X_train)**2
    y_ = tf.placeholder(tf.float64, shape=[None, 784])
    y = tf.placeholder(tf.float64, shape=[None, 784])
    with tf.Session() as sess:
        l2_loss1 = tf.reduce_sum( tf.reduce_mean(tf.square(y_-y), 0) )
        l2_loss2 = (2.0/N_train_tf)*tf.nn.l2_loss(y_-y)
        error_tf1=sess.run(fetches=l2_loss1, feed_dict={y:X_reconstruct,y_:X_train})
        error_tf2=sess.run(fetches=l2_loss2, feed_dict={y:X_reconstruct,y_:X_train})
        print 'TensorFlow error1: ', error_tf1
        print 'TensorFlow error2: ', error_tf2
        print 'U.T*U*X  error1: ',sess.run(fetches=l2_loss1, feed_dict={y:X_my_reconstruct,y_:X_train})
        print 'U.T*U*X  error2: ',sess.run(fetches=l2_loss2, feed_dict={y:X_my_reconstruct,y_:X_train})
