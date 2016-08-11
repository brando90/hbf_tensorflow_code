import json
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import re

import krls

def display():
    list_units = [5*5, 10*5, 15*5]
    #list_units = 6*np.array([6,12,18])

    nn_list_train_errors = [0.107502, 0.0313551, 0.00974167]
    krls.plot_errors(list_units, nn_list_train_errors,label='NN train error', markersize=3, colour='b')
    #krls.plot_errors(list_units, nn1_list_test_errors,label='HBF1 test', markersize=3, colour='c')
    nn_list_train_errors = [0.047651, 0.012382, 0.008429]
    krls.plot_errors(list_units, nn_list_train_errors,label='Binary Tree NN train error', markersize=3, colour='r')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    #display_results_NN_xsinglog1_x()
    display()
