import numpy as np
import scipy.io
import my_tf_pkg as mtf

task_name = 'task_f_4D_conv_2nd'
(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data(task_name)

scipy.io.savemat('f_4D_conv_2nd.mat', dict(X_train=X_train, Y_train=Y_train, X_cv=X_cv, Y_cv=Y_cv, X_test=X_test, Y_test=Y_test))
