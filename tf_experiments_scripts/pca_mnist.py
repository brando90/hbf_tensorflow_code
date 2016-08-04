import numpy as np
from sklearn.decomposition import PCA
from numpy import linalg as LA

import my_tf_pkg as mtf

task_name='task_MNIST_flat_auto_encoder'
print '----====> TASK NAME: %s' % task_name
(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data(task_name)

number_units_list = np.linspace(start=10, stop=250, num=5)
#number_units_list = [10]
for k in number_units_list:
    N_train, D = X_train.shape
    ## Do PCA
    pca = PCA(n_components=k)
    pca = pca.fit(X_train)
    X_pca = pca.transform(X_train) # M_train x K
    print 'X_pca' , X_pca.shape
    X_reconstruct = pca.inverse_transform(X_pca)

    print (1.0/N_train)*LA.norm(X_reconstruct - X_train)
