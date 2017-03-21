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
loaded_stuff = pickle.load( open( "./tmp_om_pickle/W_hist_data_0p7.p", "rb" ) )
W_hist_data = loaded_stuff['W_hist_data']
# display pickle file
plt.hist(W_hist_data,bins=arg.nb_bins,normed=True)
plt.title("Histogram W")
plt.show()
