import numpy as np
from sklearn import linear_model
from sklearn import datasets

import my_tf_pkg as mtf

import maps

print()

arg = maps.NamedDict()
arg.classificaton = True
arg.classificaton = False
arg.data_dirpath = './data/' # path to datasets
arg.data_filename = 'f_16D_binary_parity_N65536'
arg.data_filename = 'f_2D_binary_parity_N2'
arg.task_folder_name = mtf.get_experiment_folder(arg.data_filename)
arg.N_frac = 2
X_train, Y_train, X_cv, Y_cv, X_test, Y_test = mtf.get_data(arg,arg.N_frac)
print( '(N_train,D) = (%d,%d) \n (N_test,D_out) = (%d,%d) ' % (arg.N_train,arg.D, arg.N_test,arg.D_out) )

C = 1e10 # inverse of regularization. Smaller values more regularization
logreg = linear_model.LogisticRegression(C=C)

logreg.fit(X_train, Y_test)

y_prediction = logreg.predict(X_train)

#print('Y_test: ', Y_train)
#print('y_prediction: ', y_prediction)

print( np.mean( (Y_train == y_prediction).astype(int) ) )
