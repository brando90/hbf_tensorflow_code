import numpy as np
import matplotlib.pyplot as plt
import pickle

import my_tf_pkg as mtf
from my_tf_pkg import main_hp

import namespaces as ns

import pdb

#
arg = ns.Namespace()
arg.nb_bins = 35
# load pickle file
loaded_stuff = pickle.load( open( "./tmp_om_pickle/W_hist_data.p", "rb" ) )
W_hist_data = loaded_stuff['W_hist_data']
# display pickle file
W_hist_data = np.array(W_hist_data)
H, xedges, yedges = np.histogram2d(x=W_hist_data[:,0], y=W_hist_data[:,1],bins=arg.nb_bins)
plt.imshow(H)
plt.show()
