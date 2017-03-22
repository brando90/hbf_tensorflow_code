import numpy as np
import matplotlib.pyplot as plt
import pickle

import my_tf_pkg as mtf
from my_tf_pkg import main_hp

import namespaces as ns

import pdb

#
arg = ns.Namespace()
D = 8
arg.nb_bins = 50
# load pickle file
loaded_stuff = pickle.load( open( "./tmp_om_pickle/W_hist_data_8D_test.p", "rb" ) )
W_hist_data = np.array( loaded_stuff['W_hist_data'] )
print('W_hist_data.shape: ', W_hist_data.shape)
for i in range(D):
    plt.figure(i+1)
    plt.hist(W_hist_data[:,i],bins=arg.nb_bins,normed=True)
    plt.title("Histogram W")
plt.show()
#H, binedges = np.histogram(W_hist_data[:,0],bins=arg.nb_bins,density=True)
#plt.hist(W_hist_data[:,0],bins=arg.nb_bins,normed=True)
