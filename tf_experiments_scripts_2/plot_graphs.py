import json
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import re

import krls

def display4D():
    list_units = [5*5, 10*5, 15*5]
    #list_units = 6*np.array([6,12,18])

    nn_list_train_errors = [0.107502, 0.0313551, 0.00974167]
    krls.plot_errors(list_units, nn_list_train_errors,label='NN train error', markersize=3, colour='b')
    #krls.plot_errors(list_units, nn1_list_test_errors,label='HBF1 test', markersize=3, colour='c')
    nn_list_train_errors = [0.047651, 0.012382, 0.008429]
    krls.plot_errors(list_units, nn_list_train_errors,label='Binary Tree NN train error', markersize=3, colour='r')

    plt.legend()
    plt.show()

def display4D_2():
    list_units = [5*5, 10*5, 15*5]
    #list_units = 6*np.array([6,12,18])

    nn1_list_test_errors =  (8.971309392885537, 8.96969783241346, 8.971491006192313)
    krls.plot_errors(list_units, nn1_list_test_errors,label='NN train error', markersize=3, colour='b')
    #krls.plot_errors(list_units, nn1_list_test_errors,label='HBF1 test', markersize=3, colour='c')
    bt_multiple_experiment_results =  (8.96799373626709, 8.967977523803711, 8.968118667602539)
    krls.plot_errors(list_units, bt_multiple_experiment_results,label='Binary Tree NN train error', markersize=3, colour='r')

    plt.legend()
    plt.show()

def display8D():
    list_units = [7, 22, 45]
    #list_units = 6*np.array([6,12,18])

    nn1_list_test_errors = [0.951, 0.156, 0.0746]
    krls.plot_errors(list_units, nn1_list_test_errors,label='NN test error', markersize=3, colour='b')
    #krls.plot_errors(list_units, nn1_list_test_errors,label='HBF1 test', markersize=3, colour='c')
    bt_multiple_experiment_results = [0.2495, 0.0668, 0.01066]
    krls.plot_errors(list_units, bt_multiple_experiment_results,label='Binary Tree NN test error', markersize=3, colour='r')

    plt.legend()
    plt.show()

def display4D():
    list_units = [5*5, 10*5, 15*5]
    #list_units = 6*np.array([6,12,18])

    nn_list_train_errors = [0.107502, 0.0313551, 0.00974167]
    krls.plot_errors(list_units, nn_list_train_errors,label='NN train error', markersize=3, colour='b')
    #krls.plot_errors(list_units, nn1_list_test_errors,label='HBF1 test', markersize=3, colour='c')
    nn_list_train_errors = [0.047651, 0.012382, 0.008429]
    krls.plot_errors(list_units, nn_list_train_errors,label='Binary Tree NN train error', markersize=3, colour='r')

    plt.legend()
    plt.show()

def display4D_2():
    list_units = [5*5, 10*5, 15*5]
    #list_units = 6*np.array([6,12,18])

    nn1_list_test_errors =  (8.971309392885537, 8.96969783241346, 8.971491006192313)
    krls.plot_errors(list_units, nn1_list_test_errors,label='NN train error', markersize=3, colour='b')
    #krls.plot_errors(list_units, nn1_list_test_errors,label='HBF1 test', markersize=3, colour='c')
    bt_multiple_experiment_results =  (8.96799373626709, 8.967977523803711, 8.968118667602539)
    krls.plot_errors(list_units, bt_multiple_experiment_results,label='Binary Tree NN train error', markersize=3, colour='r')

    plt.legend()
    plt.show()

def display_this():
    get_k = lambda a: 7*a + 4*3*2*a**2
    shallow = lambda k: 4*k+k+k
    bt = lambda f: 2*f+f +2*f*(2*f)+ 2*(2*f)

    nn1_list_test_errors = (0.024345317618429937, 0.003073343364860556, 0.0017753394536080345, 0.0011256201563784277, 0.009532295960287112, 0.007766131572159345)
    bt_multiple_experiment_results =  (0.0004508599522523582, 0.00015345749852713197, 1.3420359209703747e-05, 1.3420359209703747e-05, 1.3420359209703747e-05, 1.3420359209703747e-05)

    #list_units = [ bt()]
    list_units = [1,2, 3, 4, 5, 6]
    krls.plot_errors(list_units, nn1_list_test_errors,label='NN train error', markersize=3, colour='b')
    krls.plot_errors(list_units, bt_multiple_experiment_results,label='Binary Tree NN train error', markersize=3, colour='r')

    plt.legend()
    plt.show()



if __name__ == '__main__':
    #display_results_NN_xsinglog1_x()
    display_this()
