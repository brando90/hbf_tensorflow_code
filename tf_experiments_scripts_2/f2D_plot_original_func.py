import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import my_tf_pkg as mtf

## get mesh data
# start_val, end_val = -1,1
# N = 100
# x_range = np.random.uniform(low=start_val, high=end_val, size=N)
# y_range = np.random.uniform(low=start_val, high=end_val, size=N)
# #x_range = np.linspace(start_val, end_val, N)
# #y_range = np.linspace(start_val, end_val, N)
# (X,Y) = np.meshgrid(x_range, y_range)
# Z = np.sin(2*np.pi*X) + 4*np.power(Y - 0.5, 2) # h_add
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
# plt.title('Original function')
#
# plt.show()
# get data form file created
nb_recursive_layers = 3
file_name = 'f_2d_task2_ml_xsinlog1_x_depth_%sdata_and_mesh.npz'%nb_recursive_layers
print 'get data form file created'
(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data_from_file(file_name=file_name)
X,Y,Z = mtf.make_meshgrid_data_from_training_data(X_data=X_train, Y_data=Y_train)
# plot
print 'plot data'
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
#surf = ax.plot_surface(X, Y, Z, cmap=cm.BrBG)
plt.title('Original function depth = %d'%nb_recursive_layers)
plt.show()
